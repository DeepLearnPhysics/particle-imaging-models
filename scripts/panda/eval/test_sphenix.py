"""Score sPHENIX: python scripts/panda/eval/test_sphenix.py --config CONFIG --checkpoint WEIGHTS --out DIR."""

if __package__:
    from .test_lartpc import main
else:
    from test_lartpc import main


if __name__ == "__main__":
    main("sphenix")
