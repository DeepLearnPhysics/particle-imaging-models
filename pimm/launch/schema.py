"""Typed launch configuration used by the Tyro CLI."""

from dataclasses import asdict, dataclass, field, fields
from typing import Annotated, Any, Literal

import tyro


SuppressedDict = Annotated[dict[str, Any], tyro.conf.Suppress]


@dataclass(kw_only=True)
class Paths:
    repo_root: str = "."
    exp_root: str = "{repo_root}/exp"


@dataclass(kw_only=True)
class Resources:
    """Run topology and scheduler requests for every executor.

    ``pimm launch`` always runs locally and ignores scheduler-only fields.
    ``pimm submit`` requires ``scheduler='slurm'`` and consumes them.
    """

    scheduler: Literal["local", "slurm"] = "local"
    nnodes: int = 1
    nproc_per_node: Annotated[
        int | Literal["auto"],
        tyro.conf.arg(help="GPUs/processes per node; 'auto' is local-only."),
    ] = 4
    cpus_per_proc: int = 12
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    constraint: str | None = None
    dependency: str | None = None
    time: str | int | None = "24:00:00"
    mem: str | None = None
    output: str = "slurm_logs/slurm-%j.out"
    error: str | None = None
    gpu_directive: Literal["gres", "gpus-per-node"] = "gres"
    job_name: str | None = None
    signal_delay_s: int = 120
    scheduler_options: SuppressedDict = field(default_factory=dict)


@dataclass(kw_only=True)
class Run:
    name: str | None = None
    timestamp: bool = True
    wandb_name: str | None = None
    wandb_project: str | None = None
    wandb_api_key: str | None = None


@dataclass(kw_only=True)
class Train:
    config: str | None = None
    weight: str | None = None
    resume: bool = False
    code_copy: bool = True
    python: str | None = None
    options: SuppressedDict = field(default_factory=dict)


@dataclass(kw_only=True)
class Container:
    runtime: Literal["none", "singularity", "apptainer", "shifter", "docker"] = "none"
    image: str | None = None
    module: str | None = None
    binds: list[str] = field(default_factory=list)
    repo_mount: str | None = "/opt/pimm/src"
    unset_env: list[str] = field(default_factory=list)
    # Absolute interpreter path inside the image (e.g. /opt/conda/bin/python).
    # When set, used as `train.sh -p` so the image's python wins regardless of a
    # host conda leak. None -> fall back to the runtime default / sys.executable.
    interpreter: str | None = None


@dataclass(kw_only=True)
class Submit:
    host: str | None = None
    folder: str | None = None


@dataclass(kw_only=True)
class Chain:
    jobs: int = 1
    resume_first: bool = False
    wandb_group: str | None = None
    wandb_job_type: str = "train-job"


@dataclass(kw_only=True)
class Watchdog:
    """Supervision of chained *interactive* runs (see pimm/launch/watchdog.py).

    A chained `pimm submit --interactive` run installs a scron job that hosts its
    driver resiliently. These knobs shape that scron entry; defaults suit NERSC,
    whose `cron` QOS runs on the login nodes and caps walltime at 1 day (the
    driver is simply re-fired by scron, resuming from checkpoint, each cycle).
    """

    qos: str = "cron"
    interval_min: int = 5
    time: str = "1-00:00:00"


@dataclass(kw_only=True)
class Rendezvous:
    endpoint: str | None = None
    id: str | None = None
    backend: str | None = None


@dataclass(kw_only=True)
class LaunchConfig:
    site: str = "local"
    executor: Annotated[
        Literal["local", "batch", "interactive"],
        tyro.conf.arg(help="Execution mode; the launch/submit command sets this."),
    ] = "local"
    interactive: Annotated[
        bool,
        tyro.conf.arg(help="Use a blocking Slurm allocation for `pimm submit`."),
    ] = False
    paths: Paths = field(default_factory=Paths)
    resources: Resources = field(default_factory=Resources)
    run: Run = field(default_factory=Run)
    train: Train = field(default_factory=Train)
    container: Container = field(default_factory=Container)
    submit: Submit = field(default_factory=Submit)
    chain: Chain = field(default_factory=Chain)
    watchdog: Watchdog = field(default_factory=Watchdog)
    rdzv: Rendezvous = field(default_factory=Rendezvous)
    env: SuppressedDict = field(default_factory=dict)
    # Shell lines that bootstrap the environment (e.g. `mamba activate pimm`,
    # `module load ...`) before pimm runs. Applied in two places: on the compute
    # node before scripts/train.sh (inside the container, if one is configured),
    # and on the login host before a remote `pimm submit` (when submit.host is
    # set). Runs regardless of container.runtime.
    setup: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaunchConfig":
        """Build typed launcher config from merged YAML dictionaries."""
        from .compat import normalize_legacy_config

        if not isinstance(data, dict):
            raise SystemExit("Launch configuration must be a mapping")
        data = normalize_legacy_config(data, source="launch configuration")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SystemExit(f"Unknown launch setting(s): {', '.join(unknown)}")

        def group(name: str, group_type: type, *legacy_keys: str) -> dict[str, Any]:
            value = data.get(name, {})
            if not isinstance(value, dict):
                raise SystemExit(f"{name} must be a mapping")
            allowed = {item.name for item in fields(group_type)} | set(legacy_keys)
            unknown_keys = sorted(set(value) - allowed)
            if unknown_keys:
                paths = ", ".join(f"{name}.{key}" for key in unknown_keys)
                raise SystemExit(f"Unknown launch setting(s): {paths}")
            return dict(value)

        train_data = group("train", Train, "no_code_copy")
        if "no_code_copy" in train_data and "code_copy" in train_data:
            raise SystemExit("train.no_code_copy conflicts with train.code_copy")
        if "no_code_copy" in train_data:
            no_code_copy = train_data.pop("no_code_copy")
            if not isinstance(no_code_copy, bool):
                raise SystemExit("train.no_code_copy must be a boolean")
            train_data["code_copy"] = not no_code_copy
        env = data.get("env", {})
        if not isinstance(env, dict):
            raise SystemExit("env must be a mapping")
        setup = data.get("setup", [])
        if not isinstance(setup, list) or not all(
            isinstance(item, str) for item in setup
        ):
            raise SystemExit("setup must be a list of strings")
        interactive = data.get("interactive", False)
        if not isinstance(interactive, bool):
            raise SystemExit("interactive must be a boolean")
        return cls(
            site=data.get("site", "local"),
            executor=data.get("executor", "local"),
            interactive=interactive,
            paths=Paths(**group("paths", Paths)),
            resources=Resources(**group("resources", Resources)),
            run=Run(**group("run", Run)),
            train=Train(**train_data),
            container=Container(**group("container", Container)),
            submit=Submit(**group("submit", Submit)),
            chain=Chain(**group("chain", Chain)),
            watchdog=Watchdog(**group("watchdog", Watchdog)),
            rdzv=Rendezvous(**group("rdzv", Rendezvous)),
            env=dict(env),
            setup=list(setup),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the dictionary shape consumed by render/submit helpers."""
        data = asdict(self)
        train = data.setdefault("train", {})
        if "code_copy" in train:
            train["no_code_copy"] = not train.pop("code_copy")
        return data


@dataclass(kw_only=True)
class LaunchCommand(LaunchConfig):
    recipe: str | None = None
    dry_run: bool = False
    output: str | None = None

    @classmethod
    def from_config(
        cls,
        config: LaunchConfig,
        *,
        recipe: str | None = None,
    ) -> "LaunchCommand":
        """Build command defaults from a typed launch config."""
        return cls(
            site=config.site,
            executor=config.executor,
            interactive=config.interactive,
            paths=config.paths,
            resources=config.resources,
            run=config.run,
            train=config.train,
            container=config.container,
            submit=config.submit,
            chain=config.chain,
            watchdog=config.watchdog,
            rdzv=config.rdzv,
            env=config.env,
            setup=config.setup,
            recipe=recipe,
        )

    def launch_config_dict(self) -> dict[str, Any]:
        """Return only launch settings, excluding CLI-only fields."""
        data = self.to_dict()
        data.pop("recipe", None)
        data.pop("dry_run", None)
        data.pop("output", None)
        return data


@dataclass(kw_only=True)
class SubmitCommand(LaunchCommand):
    no_remote: bool = False

    @classmethod
    def from_config(
        cls,
        config: LaunchConfig,
        *,
        recipe: str | None = None,
    ) -> "SubmitCommand":
        """Build submit command defaults from a typed launch config."""
        return cls(
            site=config.site,
            executor=config.executor,
            interactive=config.interactive,
            paths=config.paths,
            resources=config.resources,
            run=config.run,
            train=config.train,
            container=config.container,
            submit=config.submit,
            chain=config.chain,
            watchdog=config.watchdog,
            rdzv=config.rdzv,
            env=config.env,
            setup=config.setup,
            recipe=recipe,
        )

    def launch_config_dict(self) -> dict[str, Any]:
        """Return only launch settings, excluding CLI-only fields."""
        data = super().launch_config_dict()
        data.pop("no_remote", None)
        return data
