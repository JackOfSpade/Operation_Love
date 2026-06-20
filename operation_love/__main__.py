"""CLI:  python -m operation_love [run|stats] [--config config.yaml]"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="operation_love")
    p.add_argument("command", nargs="?", default="run", choices=["run", "stats"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if args.command == "stats":
        from .stats import show
        show(args.config)
    else:
        from .supervisor import run
        run(args.config)


if __name__ == "__main__":
    main()
