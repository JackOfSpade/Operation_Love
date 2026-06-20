"""Entry point — delegates to the supervisor (one process, all enabled apps).

Kept as a thin wrapper for back-compat; the loop lives in supervisor.py now.
"""
from __future__ import annotations

from .supervisor import run


def main(config_path: str = "config.yaml") -> None:
    run(config_path)


if __name__ == "__main__":
    main()
