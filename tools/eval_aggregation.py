"""Terminal view of the leakage-free, identity-grouped ranker evaluation.

    python -m tools.eval_aggregation                      # uses config.yaml
    python -m tools.eval_aggregation --config x.yaml --splits 5

The SAME evaluation is shown live (and auto-updating) in the hub GUI's "model
quality" card — this is just the terminal version. The logic lives in
operation_love.ranker.evaluate so both share one implementation.
"""
from __future__ import annotations

import argparse

from operation_love.ranker.evaluate import evaluate, format_report, marginal_return_summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_aggregation",
                                 description="Leakage-free, identity-grouped CV of the ranker.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--splits", type=int, default=5, help="max CV folds (clamped to your data)")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="DBSCAN cosine-distance eps for identity grouping (0.5 = sim>=0.5)")
    args = ap.parse_args()

    from operation_love import config as cfg_mod
    from operation_love.ranker import make_store
    cfg = cfg_mod.load(args.config)
    store = make_store(cfg, ensure=False)   # read-only: don't create tables/bucket just to eval
    try:
        samples = store.load_labels()
    finally:
        store.close()
    print(f"[eval] loaded {len(samples)} label(s) from {cfg.storage.backend}\n")
    result = evaluate(samples, n_splits=args.splits, eps=args.eps)
    result = {
        **result,
        "marginal_return": marginal_return_summary(
            samples, n_splits=args.splits, eps=args.eps, eval_result=result
        ),
    }
    print(format_report(result))


if __name__ == "__main__":
    main()
