"""Allow `python -m pimm.cli ...` to behave like the `pimm` console script."""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
