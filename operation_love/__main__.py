"""CLI:  python -m operation_love [run|stats|hub] [--config config.yaml]"""
from __future__ import annotations

import argparse
from pathlib import Path

from .private_files import load_private_dotenv


def main() -> None:
    # Load only this project's local .env; never walk parent directories where an
    # unrelated application's credentials could accidentally become active.
    #
    # ``load_private_dotenv`` preserves a missing file as a no-op, but rejects links and
    # hardlinks and tightens an existing repo-local file to 0600 before reading credentials.
    load_private_dotenv(Path.cwd() / ".env")
    from ._warnings import configure_warnings
    configure_warnings()

    p = argparse.ArgumentParser(prog="operation_love")
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "stats", "hub", "bugreport"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--port", type=int, default=8765, help="hub: port to serve on")
    p.add_argument("--no-browser", action="store_true", help="hub: don't auto-open the browser")
    p.add_argument("--make-launchers", action="store_true",
                   help="hub: write a double-click launcher for this OS, then exit")
    args = p.parse_args()

    if args.command == "stats":
        from .stats import show
        show(args.config)
    elif args.command == "bugreport":
        from .bugreport import build_report
        print(build_report(None, config_path=args.config))
    elif args.command == "hub":
        from .hub import make_launchers, serve
        if args.make_launchers:
            make_launchers(args.config)
        else:
            serve(args.config, port=args.port, open_browser=not args.no_browser)
    else:
        from .supervisor import run
        run(args.config)


if __name__ == "__main__":
    main()
