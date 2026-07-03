"""stats.show() — the `python -m operation_love stats` readout. No prior test file
exercised this module at all; it has real logic (load labels, train the model,
print a summary) worth pinning cheaply against a real (tmp) SQLite store."""
import yaml

from operation_love import stats
from operation_love.ranker.store import SQLiteStore


def _write_config(tmp_path, **overrides):
    cfg = {
        "enabled_apps": ["bumble"],
        "mode": "observe",
        "storage": {"backend": "sqlite"},
        "opener": {"enabled": False},
        "paths": {"data_dir": str(tmp_path), "db_file": str(tmp_path / "store.db")},
        **overrides,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_show_prints_storage_labels_and_ranker_readiness(tmp_path, capsys):
    path = _write_config(tmp_path, ranker={"min_labels_to_engage": 2})
    store = SQLiteStore(tmp_path / "store.db")
    store.add_label("r1", "bumble", True, [1.0, 1.0])
    store.add_label("r1", "bumble", False, [-1.0, -1.0])
    store.close()

    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Storage : sqlite" in out
    assert "Labels  : 2  (liked 1 / passed 1)" in out
    assert "Ranker  : ready=True" in out
    assert "Today (auto): bumble:" in out


def test_show_reports_labels_needed_when_not_ready(tmp_path, capsys):
    path = _write_config(tmp_path, ranker={"min_labels_to_engage": 40})
    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Labels  : 0  (liked 0 / passed 0)" in out
    assert "Ranker  : ready=False" in out
    assert "Seed ~40 more swipes in observe mode" in out
