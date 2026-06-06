"""Typed launch configuration used by the Tyro CLI."""

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any

import tyro


SuppressedDict = Annotated[dict[str, Any], tyro.conf.Suppress]


@dataclass(kw_only=True)
class Paths:
    repo_root: str = "."
    exp_root: str = "{repo_root}/exp"


@dataclass(kw_only=True)
class Resources:
    nnodes: int = 1
    nproc_per_node: int = 4
    cpus_per_proc: int = 12
    time: str | int | None = "24:00:00"
    mem: str | None = None


@dataclass(kw_only=True)
class Slurm:
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    constraint: str | None = None
    dependency: str | None = None
    output: str = "slurm_logs/%j_%x.txt"
    error: str | None = None
    gpu_directive: str = "gres"
    job_name: str | None = None
    image: str | None = None
    module: str | None = None
    signal_delay_s: int = 120
    additional_parameters: SuppressedDict = field(default_factory=dict)


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
    runtime: str | None = "none"
    image: str | None = None
    module: str | None = None
    binds: list[str] = field(default_factory=list)
    repo_mount: str | None = "/opt/pimm/src"
    unset_env: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class Submit:
    host: str | None = None
    folder: str | None = None
    setup: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class Chain:
    jobs: int = 1
    resume_first: bool = False
    wandb_group: str | None = None
    wandb_job_type: str = "train-job"


@dataclass(kw_only=True)
class Rendezvous:
    endpoint: str | None = None
    id: str | None = None
    backend: str | None = None


@dataclass(kw_only=True)
class LaunchConfig:
    site: str = "local"
    paths: Paths = field(default_factory=Paths)
    resources: Resources = field(default_factory=Resources)
    slurm: Slurm = field(default_factory=Slurm)
    run: Run = field(default_factory=Run)
    train: Train = field(default_factory=Train)
    container: Container = field(default_factory=Container)
    submit: Submit = field(default_factory=Submit)
    chain: Chain = field(default_factory=Chain)
    rdzv: Rendezvous = field(default_factory=Rendezvous)
    env: SuppressedDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaunchConfig":
        """Build typed launcher config from merged YAML dictionaries."""
        data = data or {}
        train_data = dict(data.get("train") or {})
        if "no_code_copy" in train_data and "code_copy" not in train_data:
            train_data["code_copy"] = not bool(train_data.pop("no_code_copy"))
        return cls(
            site=data.get("site", "local"),
            paths=Paths(**dict(data.get("paths") or {})),
            resources=Resources(**dict(data.get("resources") or {})),
            slurm=Slurm(**dict(data.get("slurm") or {})),
            run=Run(**dict(data.get("run") or {})),
            train=Train(**train_data),
            container=Container(**dict(data.get("container") or {})),
            submit=Submit(**dict(data.get("submit") or {})),
            chain=Chain(**dict(data.get("chain") or {})),
            rdzv=Rendezvous(**dict(data.get("rdzv") or {})),
            env=dict(data.get("env") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy dict shape consumed by render/submit helpers."""
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
            paths=config.paths,
            resources=config.resources,
            slurm=config.slurm,
            run=config.run,
            train=config.train,
            container=config.container,
            submit=config.submit,
            chain=config.chain,
            rdzv=config.rdzv,
            env=config.env,
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
            paths=config.paths,
            resources=config.resources,
            slurm=config.slurm,
            run=config.run,
            train=config.train,
            container=config.container,
            submit=config.submit,
            chain=config.chain,
            rdzv=config.rdzv,
            env=config.env,
            recipe=recipe,
        )

    def launch_config_dict(self) -> dict[str, Any]:
        """Return only launch settings, excluding CLI-only fields."""
        data = super().launch_config_dict()
        data.pop("no_remote", None)
        return data
