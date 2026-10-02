"""CLI:  python -m operation_love [run|stats|hub] [--config config.yaml]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .private_files import load_private_dotenv


def main() -> None:
    p = argparse.ArgumentParser(prog="operation_love")
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "stats", "hub", "bugreport"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--port", type=int, default=8765, help="hub: port to serve on")
    p.add_argument("--no-browser", action="store_true", help="hub: don't auto-open the browser")
    p.add_argument("--make-launchers", action="store_true",
                   help="hub: write a double-click launcher for this OS, then exit")
    args = p.parse_args()

    # These switches would otherwise be silently ignored for run/stats/bugreport (or while
    # writing a launcher). A command-line typo should fail before it looks like a successful
    # operation with unexpectedly different behavior.
    if args.command != "hub":
        if args.port != 8765:
            p.error("--port is only valid with the hub command")
        if args.no_browser:
            p.error("--no-browser is only valid with the hub command")
        if args.make_launchers:
            p.error("--make-launchers is only valid with the hub command")
    elif not 1 <= args.port <= 65535:
        p.error("--port must be an integer from 1 to 65535")
    elif args.make_launchers and args.no_browser:
        p.error("--no-browser cannot be used with --make-launchers")

    # A launcher is specifically the recovery/setup path for a fresh checkout, before runtime
    # extras (including python-dotenv) necessarily exist. It neither reads credentials nor
    # launches the application, so keep that path dependency-free.
    if args.command == "hub" and args.make_launchers:
        from .hub import make_launchers
        make_launchers(args.config)
        return

    # Load only this project's local .env; never walk parent directories where an
    # unrelated application's credentials could accidentally become active.
    #
    # ``load_private_dotenv`` preserves a missing file as a no-op, but rejects links and
    # hardlinks and tightens an existing repo-local file to 0600 before reading credentials.
    load_private_dotenv(Path.cwd() / ".env")
    from ._warnings import configure_warnings
    configure_warnings()

    if args.command == "stats":
        from .stats import show
        show(args.config)
    elif args.command == "bugreport":
        from .bugreport import build_report
        sys.stdout.write(build_report(None, config_path=args.config) + "\n")
    elif args.command == "hub":
        from .hub import serve
        serve(args.config, port=args.port, open_browser=not args.no_browser)
    else:
        from .supervisor import run
        run(args.config)


if __name__ == "__main__":
    main()
