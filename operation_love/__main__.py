"""CLI:  python -m operation_love [run|stats|hub] [--config config.yaml]"""
from __future__ import annotations

import argparse


def main() -> None:
    # Load only this project's local .env; never walk parent directories where an
    # unrelated application's credentials could accidentally become active.
    #
    # python-dotenv is a HARD dependency (see pyproject.toml: "python-dotenv>=1.0"), not an
    # optional extra -- its absence means the install itself is broken. This used to swallow
    # ImportError with a bare `pass`, on the theory that CLI diagnostics should stay usable
    # in a "partial/minimal install". That silence is exactly what once hid a real
    # GEMINI_API_KEY outage: python-dotenv was not installed, so .env was NEVER READ, and the
    # run failed several layers downstream with a misleading "GEMINI_API_KEY is required"
    # instead of pointing at the actual cause (the package missing). This project's rule is
    # fail loud, never silently degrade -- so a missing package now raises immediately, with
    # an actionable fix, instead of continuing into a confusing downstream failure.
    #
    # A MISSING ``.env`` FILE, by contrast, is NOT an error: env vars may legitimately come
    # from the real environment instead of a .env file, and load_dotenv() itself already
    # treats a missing file as a silent no-op (it returns False, it does not raise) -- so
    # that behavior is preserved untouched below; only the import is fail-loud now.
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is not installed, so .env cannot be loaded (GEMINI_API_KEY and "
            "any other .env-provided settings will be invisible to this process even if "
            "the file exists). python-dotenv is a required dependency of this project, not "
            "optional, so this means the install is broken -- fix it with "
            "`pip install python-dotenv`, or reinstall the project's dependencies "
            "(`pip install -e .`)."
        ) from exc
    from pathlib import Path
    load_dotenv(Path.cwd() / ".env")
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
