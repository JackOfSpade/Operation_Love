"""A secondary evidence outage must not reopen an accepted opener's lifecycle."""

from operation_love.opener.service import (
    OpenerPick,
    OpenerService,
    _StagedOpenerRecord,
)


def test_evidence_failure_does_not_duplicate_or_discard_a_committed_opener(capsys):
    class Store:
        def __init__(self):
            self.openers = []

        def record_opener(self, *args, **kwargs):
            self.openers.append((args, kwargs))

        def record_opener_send_evidence(self, *args, **kwargs):
            raise OSError("evidence storage unavailable: private frame bytes")

    store = Store()
    service = OpenerService(None, None, store, "casual")
    pick = OpenerPick(
        "Where was that hike?",
        index=2,
        _staged_record=_StagedOpenerRecord(
            run_id="run",
            app="hinge",
            model="model",
            opener="Where was that hike?",
            referenced="hiking",
            angle="asking about a place",
            item_description="hiking photo",
            recent_entry={"opener": "Where was that hike?"},
            prompt_sha256=service.prompt_sha256,
        ),
    )

    assert service.commit_opener(pick, pre_send_evidence={"frame": b"png"})
    assert not service.commit_opener(pick)
    assert not service.discard_opener(pick)
    assert len(store.openers) == 1
    assert store.openers[0][1]["decision"] == "like"
    assert len(service.recent_openers_snapshot()) == 1
    warning = capsys.readouterr().out
    assert "opener was recorded" in warning
    assert "OSError" in warning
    assert "private frame bytes" not in warning


def test_primary_opener_failure_retains_draft_for_retry():
    class Store:
        fail = True
        written = 0

        def record_opener(self, *args, **kwargs):
            if self.fail:
                raise OSError("opener storage unavailable")
            self.written += 1

        def record_opener_send_evidence(self, *args, **kwargs):
            raise AssertionError("evidence should not run without its opener")

    store = Store()
    service = OpenerService(None, None, store, "casual")
    pick = OpenerPick(
        "Where was that hike?",
        index=2,
        _staged_record=_StagedOpenerRecord(
            run_id="run",
            app="hinge",
            model="model",
            opener="Where was that hike?",
            referenced="hiking",
            angle="asking about a place",
            item_description="hiking photo",
            recent_entry={"opener": "Where was that hike?"},
            prompt_sha256=service.prompt_sha256,
        ),
    )

    assert not service.commit_opener(pick, pre_send_evidence={"frame": b"png"})
    assert pick._staged_record is not None
    assert service.recent_openers_snapshot() == []
    store.fail = False
    assert service.commit_opener(pick)
    assert store.written == 1
    assert not service.commit_opener(pick)
