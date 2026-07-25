"""Google Cloud Batch executor for ``pimm submit --site gcloud``.

Cloud Batch is GCP's managed, queue-based batch service -- the closest analog to
a Slurm queue. A submitted job queues, provisions an A100 VM, runs the pimm dev
container image, and tears the VM down. Output artifacts are written to a GCS
bucket that Batch mounts on the VM via gcsfuse, so the training/checkpoint code
sees an ordinary local directory and needs no cloud-storage awareness.

The user writes ``paths.exp_root: gs://<bucket>/<prefix>`` in the site config;
this module derives the bucket, mounts it at ``gcs_mount_path`` on the VM, and
rewrites ``EXP_ROOT`` to the mounted local path before rendering the training
script. Everything else reuses the shared renderers in ``local.py``.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

from .local import build_train_sh_command, redact_script, render_script
from .utils import ROOT, as_bool, scheduler, shell_join, slurm_time_to_minutes, write_text

# Cloud Batch only auto-creates gcsfuse mount dirs under /mnt/disks/; mounting
# elsewhere fails with "mount: stat <path>: no such file or directory".
DEFAULT_MOUNT_PATH = "/mnt/disks/gcs"
DEFAULT_PROVISIONING_MODEL = "STANDARD"
DEFAULT_BOOT_DISK_GB = 200
# PyTorch DataLoader workers pass tensors to the main process through POSIX
# shared memory (/dev/shm). A container's /dev/shm defaults to 64 MB, which a
# multi-worker loader exhausts almost immediately -- the failure surfaces as
# `RuntimeError: unable to allocate shared memory (shm) ... (11)`. Share the
# host IPC namespace by default so shm is bounded by host RAM, not 64 MB; set
# `scheduler_options.shm_size` (e.g. "16g") to size an isolated /dev/shm instead.
DEFAULT_CONTAINER_OPTIONS = "--ipc=host"
# Open-file limit for the Batch container. Across one training epoch the rank-0
# process's open-fd count climbs well past the old 65536 cap (worker-shared
# tensors, gcsfuse read/cache handles, CUDA/NCCL) and the first checkpoint
# publish on the gcsfuse mount then trips `OSError: [Errno 24] Too many open
# files`. The same code runs fine on s3df, whose login shells give a 262144
# limit -- i.e. the working set fits under 262144 but not 65536. Set the
# container hard+soft limit high (1048576, the usual Linux fs.nr_open ceiling
# and a value docker `--ulimit` reliably accepts) so the VM matches/exceeds the
# s3df headroom; train.sh's `ulimit -n` and train.py's setrlimit then raise the
# soft limit to this hard cap.
DEFAULT_NOFILE_LIMIT = 1048576
# gcsfuse file cache. Training data is read straight off this mount as many
# small random-offset reads (one HDF5 event per __getitem__), which is slow
# over the network. Enabling the file cache pulls each object to local disk on
# first touch, so the rest of that shard's reads -- and every later epoch --
# hit local disk instead of GCS. `--cache-dir` is what turns the cache on (it
# is a host path: gcsfuse runs on the VM, not in the container, and must live
# under /mnt/disks on Batch VMs; gcsfuse creates it if missing).
# `--file-cache-max-size-mb=-1` lets gcsfuse fill the free space on that disk
# with LRU eviction, so it can't overflow the boot disk. To cache the full
# dataset raise `resources.scheduler_options.boot_disk_gb` accordingly (train
# alone is ~141 GB); disable with `scheduler_options.gcs_file_cache=false`.
DEFAULT_GCS_CACHE_DIR = "/mnt/disks/gcsfuse-cache"
DEFAULT_GCS_CACHE_MAX_SIZE_MB = -1
# Cap gcsfuse's per-host HTTP connections so its host-side fd budget (which the
# container ulimit cannot raise) is not exhausted by unbounded connections plus
# the checkpoint write burst. 100 matches gcsfuse's historical default and is
# ample for this workload; override via scheduler_options.gcs_max_conns_per_host
# (falsy leaves the gcsfuse default).
DEFAULT_GCS_MAX_CONNS_PER_HOST = 100
DEFAULT_CODE_PREFIX = "_pimm_code"
DEFAULT_STAGE_DIR = "/tmp/pimm_src"
# Files never worth shipping to GCS with the source (a Python regex for
# `gsutil rsync -x`): VCS/venv/caches, large data/checkpoint artifacts, and
# `.env` -- the local .env is site-specific (e.g. s3df paths) and train.sh
# would source it and clobber the gcloud site's env; set gcloud env in the site
# config instead.
#
# NOTE: `gsutil rsync -x` matches the pattern against the *relative* path with
# `re.match`, i.e. anchored at the start (see `gsutil help rsync`). So patterns
# for anything that can appear nested must allow a leading path prefix
# (`(.*/)?` for dirs, `.*` for suffixes) -- a bare `__pycache__/` or `\.pyc$`
# would only ever match at the repo root and silently ship nested copies.
STAGE_EXCLUDE = (
    r"(.*/)?(\.git|\.venv|__pycache__|\.pytest_cache|\.mypy_cache)/|"
    r"(exp|slurm_logs)/|"
    r"(.*/)?\.env$|"
    r".*\.(pyc|h5|pth)$"
)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI into ``(bucket, prefix)``.

    ``prefix`` has no leading/trailing slashes and may be empty (bucket root).
    """
    if not uri.startswith("gs://"):
        raise SystemExit(f"Expected a gs:// URI, got {uri!r}")
    rest = uri[len("gs://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise SystemExit(f"gs:// URI is missing a bucket: {uri!r}")
    return bucket, prefix.strip("/")


def sanitize_job_name(run_name: str) -> str:
    """Coerce a run name into a valid Cloud Batch job id.

    Batch job names must match ``^[a-z]([a-z0-9-]{0,61})?$`` (<=63 chars,
    lowercase, start with a letter). Run names contain uppercase, underscores
    (timestamp), and dots, so normalize them.
    """
    name = re.sub(r"[^a-z0-9-]", "-", run_name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    if not name or not name[0].isalpha():
        name = f"pimm-{name}".strip("-")
    return name[:63].rstrip("-")


def build_batch_job(
    cfg: dict[str, Any], run_name: str
) -> tuple[dict[str, Any], str, tuple[str, str] | None]:
    """Build a single-node A100 Batch job.

    Returns ``(job_spec, inner_script, stage_plan)`` where ``stage_plan`` is
    ``(local_source_root, gs_dest)`` when code staging is enabled (else None) --
    the caller performs the actual upload.
    """
    opts = cfg.get("resources", {}).get("scheduler_options", {}) or {}
    mount_path = str(opts.get("gcs_mount_path") or DEFAULT_MOUNT_PATH).rstrip("/")

    exp_root = str(cfg.get("paths", {}).get("exp_root", ""))
    bucket, prefix = parse_gs_uri(exp_root)
    # EXP_ROOT on the VM points into the gcsfuse mount, not the gs:// URI.
    exp_root_local = "/".join([mount_path, prefix]).rstrip("/") if prefix else mount_path

    # Cloud Batch is the container runner: render the *inner* training script
    # with no nested `docker run` (runtime="none") and the mounted EXP_ROOT.
    render_cfg = copy.deepcopy(cfg)
    render_cfg.setdefault("container", {})["runtime"] = "none"
    render_cfg.setdefault("paths", {})["exp_root"] = exp_root_local

    # Code staging: rsync the local checkout to the bucket at submit time, then
    # have the job copy it off the gcsfuse mount to a local dir and run from
    # there -- so edits take effect without rebuilding the image (the image then
    # supplies only the environment/deps). On by default; opt out per site.
    stage = as_bool(opts.get("stage_code", True))
    stage_plan: tuple[str, str] | None = None
    stage_dir = str(opts.get("stage_dir") or DEFAULT_STAGE_DIR).rstrip("/")
    code_mount = ""
    if stage:
        code_prefix = str(opts.get("code_prefix") or DEFAULT_CODE_PREFIX).strip("/")
        code_rel = f"{code_prefix}/{run_name}"
        code_mount = f"{mount_path}/{code_rel}"
        # Run from the staged copy and make it win over the baked-in install.
        render_cfg["paths"]["repo_root"] = stage_dir
        render_cfg.setdefault("env", {})["PYTHONPATH"] = stage_dir
        stage_plan = (str(ROOT), f"gs://{bucket}/{code_rel}")

    train_cmd = build_train_sh_command(render_cfg, run_name)
    script = render_script(render_cfg, train_cmd, run_name)

    if stage:
        # Copy the staged source off the (high-latency) gcsfuse mount to local
        # disk before cd-ing into it. Injected right before render_script's
        # `cd <repo_root>` line.
        cd_line = f"cd {shlex.quote(stage_dir)}"
        copy_block = "\n".join(
            [
                "echo '# staging pimm source from gcsfuse mount'",
                f"mkdir -p {shlex.quote(stage_dir)}",
                f"cp -a {shlex.quote(code_mount)}/. {shlex.quote(stage_dir)}/",
            ]
        )
        if cd_line not in script:
            raise SystemExit("could not inject code-staging step into rendered script")
        script = script.replace(cd_line, f"{copy_block}\n{cd_line}", 1)

    image = cfg.get("container", {}).get("image")
    provisioning = str(opts.get("provisioning_model") or DEFAULT_PROVISIONING_MODEL)
    boot_disk_gb = int(opts.get("boot_disk_gb") or DEFAULT_BOOT_DISK_GB)
    minutes = slurm_time_to_minutes(cfg.get("resources", {}).get("time", "24:00:00"))
    max_run_seconds = f"{minutes * 60}s"

    instance_policy: dict[str, Any] = {
        "machineType": opts["machine_type"],
        "provisioningModel": provisioning,
        "bootDisk": {"sizeGb": boot_disk_gb},
    }
    # a2-* machine types bundle their A100s automatically. For other machine
    # types (e.g. n1-*), require an explicit accelerator spec.
    accel_type = opts.get("accelerator_type")
    accel_count = opts.get("accelerator_count")
    if accel_type and accel_count:
        instance_policy["accelerators"] = [
            {"type": accel_type, "count": int(accel_count)}
        ]

    # DataLoader workers need more shared memory than a container's 64 MB
    # default; pass docker-run flags to the Batch container to raise it. An
    # explicit shm_size sizes an isolated /dev/shm; otherwise share host IPC.
    shm_size = opts.get("shm_size")
    shm_option = f"--shm-size={shm_size}" if shm_size else DEFAULT_CONTAINER_OPTIONS
    nofile = int(opts.get("nofile_limit") or DEFAULT_NOFILE_LIMIT)
    container_options = f"{shm_option} --ulimit nofile={nofile}:{nofile}"

    # Mount the bucket, optionally with the gcsfuse file cache turned on so the
    # dataset's random-offset reads are served from local disk after first
    # touch (see DEFAULT_GCS_CACHE_DIR). Batch passes mountOptions to gcsfuse as
    # CLI flags ("--flag value" strings).
    gcs_volume: dict[str, Any] = {
        "gcs": {"remotePath": bucket},
        "mountPath": mount_path,
    }
    mount_options: list[str] = []
    if as_bool(opts.get("gcs_file_cache", True)):
        cache_dir = str(opts.get("gcs_cache_dir") or DEFAULT_GCS_CACHE_DIR).rstrip("/")
        cache_max_mb = int(
            opts.get("gcs_cache_max_size_mb", DEFAULT_GCS_CACHE_MAX_SIZE_MB)
        )
        mount_options += [
            f"--cache-dir {cache_dir}",
            f"--file-cache-max-size-mb {cache_max_mb}",
        ]
    # Bound the gcsfuse process's HTTP connections per host. gcsfuse runs on the
    # VM host with its own (~1024) fd budget that the container `--ulimit` cannot
    # raise; unbounded connections plus the checkpoint write burst can exhaust it
    # (EMFILE on the mount). A finite cap keeps gcsfuse's socket fds bounded. Set
    # falsy to leave gcsfuse's default.
    max_conns = opts.get("gcs_max_conns_per_host", DEFAULT_GCS_MAX_CONNS_PER_HOST)
    if max_conns:
        mount_options.append(f"--max-conns-per-host {int(max_conns)}")
    if mount_options:
        gcs_volume["mountOptions"] = mount_options

    task_spec: dict[str, Any] = {
        "runnables": [
            {
                "container": {
                    "imageUri": image,
                    "entrypoint": "/bin/bash",
                    "commands": ["-lc", script],
                    # gcsfuse mount lives on the host; expose it to the container.
                    "volumes": [f"{mount_path}:{mount_path}"],
                    "options": container_options,
                }
            }
        ],
        "volumes": [gcs_volume],
        "maxRunDuration": max_run_seconds,
        "maxRetryCount": 0,
    }

    # W&B key: the key (--run.wandb-api-key, moved into env by finalize_config)
    # is rendered as an env export in the script. It must be a non-empty value:
    # the gcloud VM does not see the local .env, so a missing key means the run
    # cannot authenticate to W&B. Fail loudly at build time rather than letting
    # the job start and die. build_batch_job is the single chokepoint every
    # submit (and --dry-run render) goes through, so this guards them all.
    if not str(cfg.get("env", {}).get("WANDB_API_KEY", "")).strip():
        raise SystemExit(
            "WANDB_API_KEY is empty; a gcloud run needs a real W&B key.\n"
            "It is read automatically from a `WANDB_API_KEY=...` line in the "
            "repo `.env`; add that line, or pass --run.wandb-api-key explicitly."
        )

    allocation_instance: dict[str, Any] = {
        "policy": instance_policy,
        "installGpuDrivers": True,
    }
    network = opts.get("network")
    subnetwork = opts.get("subnetwork")
    allocation_policy: dict[str, Any] = {"instances": [allocation_instance]}
    if opts.get("service_account"):
        allocation_policy["serviceAccount"] = {"email": opts["service_account"]}
    if network or subnetwork:
        interface: dict[str, Any] = {}
        if network:
            interface["network"] = network
        if subnetwork:
            interface["subnetwork"] = subnetwork
        allocation_policy["network"] = {"networkInterfaces": [interface]}

    job_spec: dict[str, Any] = {
        "taskGroups": [{"taskCount": 1, "taskSpec": task_spec}],
        "allocationPolicy": allocation_policy,
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    return job_spec, script, stage_plan


def stage_code(local_root: str, gs_dest: str, *, dry_run: bool) -> None:
    """Mirror the local checkout to ``gs_dest`` with ``gsutil rsync``."""
    argv = [
        "gsutil", "-m", "rsync", "-r", "-x", STAGE_EXCLUDE, local_root, gs_dest,
    ]
    if dry_run:
        print(f"# would stage code: {shell_join(argv)}")
        return
    if shutil.which("gsutil") is None:
        raise SystemExit(
            "gsutil not found; it is required to stage code to GCS "
            "(install the Google Cloud SDK, or set scheduler_options.stage_code=false)"
        )
    print(f"# staging code -> {gs_dest}")
    rc = subprocess.run(argv).returncode
    if rc != 0:
        raise SystemExit(f"gsutil rsync failed with exit code {rc}")


def run_gcloud(
    cfg: dict[str, Any],
    run_name: str,
    *,
    dry_run: bool,
    output: str | None,
) -> int:
    """Render and submit a Google Cloud Batch job for a training run."""
    if scheduler(cfg) != "gcloud":
        raise SystemExit("run_gcloud requires resources.scheduler='gcloud'")

    opts = cfg.get("resources", {}).get("scheduler_options", {}) or {}
    project = opts["project"]
    location = opts["location"]

    job_spec, script, stage_plan = build_batch_job(cfg, run_name)
    job_name = sanitize_job_name(run_name)
    job_json = json.dumps(job_spec, indent=2)

    # Upload the local checkout before submitting (a no-op print under --dry-run).
    if stage_plan is not None:
        stage_code(stage_plan[0], stage_plan[1], dry_run=dry_run)

    # The inner training script is embedded as a JSON string, so its `export`
    # lines can't be redacted after serialization. Redact the script itself and
    # splice it into a display-only copy of the spec.
    display_spec = copy.deepcopy(job_spec)
    display_spec["taskGroups"][0]["taskSpec"]["runnables"][0]["container"][
        "commands"
    ][1] = redact_script(script)
    display_json = json.dumps(display_spec, indent=2)
    if output:
        path = write_text(output, display_json)
        print(f"# wrote Cloud Batch job spec: {path}")
    if dry_run:
        print(f"# gcloud batch jobs submit {job_name} "
              f"--project {project} --location {location} --config <spec>")
        print(display_json)
        return 0

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=f"{job_name}-", delete=False
    ) as handle:
        handle.write(job_json)
        config_path = handle.name

    argv = [
        "gcloud",
        "batch",
        "jobs",
        "submit",
        job_name,
        "--project",
        project,
        "--location",
        location,
        "--config",
        config_path,
    ]
    print(f"# submitting Cloud Batch job {job_name} to {project}/{location}")
    return subprocess.run(argv).returncode
