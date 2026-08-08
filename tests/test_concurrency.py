"""Two app workers run concurrently in one process sharing the store (offline)."""
import sys
import threading
import time
import types

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import DatingAppDriver
from operation_love.opener.opener import OpenerResult
from operation_love.opener.service import OpenerService
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"claude-opus-4-8": ModelPricing(input=5.0, output=25.0)}


class _Driver(DatingAppDriver):
    def __init__(self, n):
        self.n = n; self.i = 0; self.closed = False
    def open_session(self): pass
    def next_profile(self):
        if self.i >= self.n:
            return None
        self.i += 1; return Profile(photos=[b"x"])
    def out_of_profiles(self): return self.i >= self.n
    def like(self, opener=None): pass
    def dislike(self): pass
    def close(self): self.closed = True


class _Decider:
    def decide(self, profile):
        return Decision("dislike", 0.1, [0.1], "ranker")


class _Store:
    def __init__(self):
        self._lock = threading.Lock()
        self.decisions = 0
    def record_decision(self, *a, **k):
        with self._lock:
            self.decisions += 1
    def record_profile(self, *a, **k): pass
    def add_label(self, *a, **k): pass
    def count_today(self, app): return 0


class _Pacing:
    swipe_delay_s = 0.0


def test_two_workers_run_concurrently_and_share_store():
    store = _Store()
    stop = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    d1, d2 = _Driver(4), _Driver(6)
    w1 = Worker("bumble", d1, _Decider(), svc, store, "r", _Pacing(), stop, mode="auto")
    w2 = Worker("hinge", d2, _Decider(), svc, store, "r", _Pacing(), stop, mode="auto")

    w1.start(); w2.start()
    w1.join(timeout=10); w2.join(timeout=10)

    assert not w1.is_alive() and not w2.is_alive()
    assert d1.i == 4 and d2.i == 6 and d1.closed and d2.closed
    assert store.decisions == 10           # both workers' swipes recorded to the shared store


class _OpenerClient:
    """One opener costs $0.002 at the PRICING below (400 input tok * $5/MTok)."""
    def generate(self, profile, style):
        return OpenerResult(opener="hi", referenced="r",
                            usage=Usage(input_tokens=400), model="claude-opus-4-8")


class _LikeDecider:
    def decide(self, profile):
        return Decision("like", 0.9, [0.1], "ranker")


class _SpendStore(_Store):
    """Adds the spend/opener sinks the like path needs; records (app, decision) tuples."""
    def __init__(self):
        super().__init__()
        self.rows = []
    def record_decision(self, run_id, app, decision, score, source="auto"):
        with self._lock:
            self.rows.append((app, decision))
    def record_opener(self, *a, **k): pass
    def record_spend(self, *a, **k): pass


def test_one_worker_budget_exhaustion_stops_the_other_worker():
    """Headline multi-worker safety contract: one shared stop_event + one shared OpenerService.
    When worker A exhausts the GLOBAL budget (on_exhausted='stop'), it sets stop_requested ->
    the shared stop_event; worker B, parked in its loop, then halts WITHOUT swiping (and without
    spending more), because the supervisor hands every worker the SAME stop_event."""
    class _Spender(DatingAppDriver):
        def __init__(self, b_parked):
            self.b_parked = b_parked
            self.i = 0; self.likes = []; self.dislikes = 0; self.closed = False
        def open_session(self): pass
        def next_profile(self):
            self.b_parked.wait(timeout=5)         # act only once B is parked (determinism)
            if self.i >= 1:
                return None
            self.i += 1; return Profile(photos=[b"x"])
        def out_of_profiles(self): return False
        def like(self, opener=None, item_index=0): self.likes.append(opener)
        def dislike(self): self.dislikes += 1
        def close(self): self.closed = True

    class _Gated(DatingAppDriver):
        def __init__(self, stop):
            self.stop = stop; self.parked = threading.Event()
            self.likes = []; self.dislikes = 0; self.closed = False
        def open_session(self): pass
        def next_profile(self):
            self.parked.set()                     # signal B is in its loop, waiting
            self.stop.wait(timeout=5)             # released only when A exhausts -> stop set
            return Profile(photos=[b"x"])         # returns AFTER stop; worker breaks before acting
        def out_of_profiles(self): return False
        def like(self, opener=None, item_index=0): self.likes.append(opener)
        def dislike(self): self.dislikes += 1
        def close(self): self.closed = True

    stop = threading.Event()
    store = _SpendStore()
    # budget 0.001 < one opener's $0.002 -> A's first like exhausts the shared budget.
    svc = OpenerService(_OpenerClient(), CostTracker(PRICING, run_budget_usd=0.001), store,
                        "s", on_exhausted="stop")
    b = _Gated(stop)
    a = _Spender(b.parked)
    wa = Worker("hinge", a, _LikeDecider(), svc, store, "r", _Pacing(), stop, mode="auto")
    wb = Worker("bumble", b, _LikeDecider(), svc, store, "r", _Pacing(), stop, mode="auto")

    wb.start()
    assert b.parked.wait(2)                        # B is in its loop, waiting on the shared stop
    wa.start()
    wa.join(timeout=10); wb.join(timeout=10)

    assert not wa.is_alive() and not wb.is_alive()
    assert stop.is_set()                           # A's exhaustion propagated to the shared event
    assert len(a.likes) == 1                       # A liked exactly once, then the budget blew
    assert b.likes == [] and b.dislikes == 0       # B halted WITHOUT swiping
    assert a.closed and b.closed
    assert store.rows == [("hinge", "like")]       # only A's single decision recorded


# --- Embedder._ensure() concurrency -------------------------------------------------
#
# The tests below exercise the REAL Embedder._ensure() (double-checked locking, sentinel
# check, and commit ordering all run unmodified). They stub only the model-CONSTRUCTION
# boundary: best_device(), the lazily-imported open_clip/onnxruntime modules (injected via
# sys.modules so this file needs neither package actually installed), and
# Embedder._build_arc (which itself lazily imports insightface -- stubbing the method
# avoids having to fake that import chain too). _select_onnx_providers and
# _set_arc_providers are NOT stubbed: they're cheap and their real behavior -- including
# the commit-ordering guarantee this file cares about -- is exactly what's under test.
#
# A prior version of the first test below built a real Embedder() and then overwrote the
# bound method with a hand-written replica of the locking ("mirror the real double-checked
# locking"), so only the replica ever ran under test. Proof of the gap: gutting the real
# _ensure() (deleting the fast-path sentinel check AND the `with self._lock:` double check)
# left that test at "1 passed". None of the tests here touch operation_love/vision/embed.py.

import operation_love.vision.embed as embed_mod  # noqa: E402
from operation_love.vision.embed import Embedder  # noqa: E402


def _install_fake_open_clip(monkeypatch, create_model_and_transforms):
    """Inject a fake `open_clip` module via sys.modules (not a real import, and not a
    monkeypatch of an already-imported real module) so this needs neither the real
    package installed nor any particular import order -- works on a bare machine."""
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "open_clip", module)


def _install_fake_onnxruntime(monkeypatch, providers=("CPUExecutionProvider",)):
    module = types.ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


class _FakeClipModel:
    """Cheap stand-in for the real open_clip model object: only `.to().eval()` is used
    by _ensure() before the model is stashed on the embedder."""

    def to(self, device):
        return self

    def eval(self):
        return self


def test_ensure_builds_models_exactly_once_under_concurrent_access(monkeypatch):
    """N threads racing on the REAL _ensure() build the (stubbed) heavy models exactly
    once. The stand-ins count invocations and sleep briefly so callers genuinely overlap
    instead of trivially serializing -- a real double-init race, guarded by the real lock."""
    monkeypatch.setattr(embed_mod, "best_device", lambda: "cpu")

    clip_calls = []

    def fake_create(name, pretrained=None):
        clip_calls.append(1)
        time.sleep(0.02)             # slow enough that racers genuinely queue on the lock
        return _FakeClipModel(), None, (lambda img: img)

    _install_fake_open_clip(monkeypatch, fake_create)
    _install_fake_onnxruntime(monkeypatch)

    arc_calls = []

    def fake_build_arc(self, providers):
        arc_calls.append(1)
        time.sleep(0.02)
        return object()

    monkeypatch.setattr(Embedder, "_build_arc", fake_build_arc)

    embedder = Embedder()
    n = 8
    barrier = threading.Barrier(n)

    def run():
        barrier.wait(timeout=5)      # all threads hit _ensure() at essentially the same instant
        embedder._ensure()

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not any(t.is_alive() for t in threads)
    assert len(clip_calls) == 1, f"CLIP built {len(clip_calls)} times (expected 1 -> double-init race!)"
    assert len(arc_calls) == 1, f"ArcFace built {len(arc_calls)} times (expected 1 -> double-init race!)"
    assert embedder._arc is not None and embedder._clip is not None


def test_ensure_never_exposes_a_half_initialized_embedder_to_a_racing_reader(monkeypatch):
    """_ensure() deliberately builds CLIP first and commits self._arc -- the sentinel its
    own fast path (and every other caller) short-circuits on -- LAST, so a thread that
    ever observes self._arc set can safely assume self._clip is also ready. A background
    thread polls the embedder's raw attributes, unsynchronized -- exactly like the fast
    path's read at the top of _ensure() -- for the whole duration of the concurrent build,
    and fails the test the instant it ever catches _arc set while _clip is still None.
    This asserts on state a racing thread actually observed mid-build, not just the final
    state, which the old replica-based test never exercised at all."""
    monkeypatch.setattr(embed_mod, "best_device", lambda: "cpu")

    def fake_create(name, pretrained=None):
        time.sleep(0.03)
        return _FakeClipModel(), None, (lambda img: img)

    _install_fake_open_clip(monkeypatch, fake_create)
    _install_fake_onnxruntime(monkeypatch)

    def fake_build_arc(self, providers):
        time.sleep(0.03)             # widen the arc-not-yet-committed window for the watcher
        return object()

    monkeypatch.setattr(Embedder, "_build_arc", fake_build_arc)

    embedder = Embedder()
    violations = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            arc, clip = embedder._arc, embedder._clip
            if arc is not None and clip is None:
                violations.append((arc, clip))
        arc, clip = embedder._arc, embedder._clip   # one last look after the builders joined
        if arc is not None and clip is None:
            violations.append((arc, clip))

    watcher = threading.Thread(target=watch)
    watcher.start()

    n = 5
    barrier = threading.Barrier(n)

    def run():
        barrier.wait(timeout=5)
        embedder._ensure()

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    stop.set()
    watcher.join(timeout=5)

    assert violations == [], (
        f"a racing reader observed self._arc set while self._clip was still None: {violations}"
    )
    assert embedder._arc is not None and embedder._clip is not None


def test_ensure_stays_retryable_for_a_concurrent_caller_after_clip_load_fails(monkeypatch):
    """Concurrent companion to test_vision.py's single-threaded
    test_ensure_leaves_embedder_retryable_if_clip_load_fails (read there for the single-
    call contract; not duplicated here). Two threads race on the real lock via the real
    _ensure(): whichever wins hits a simulated CLIP load failure and -- because self._arc
    is only committed after CLIP succeeds -- leaves nothing committed, so the other
    thread, a genuinely concurrent SECOND caller (not a manual retry sequenced by the
    test) rather than the same caller retrying, picks the build back up once it gets the
    lock and completes it."""
    monkeypatch.setattr(embed_mod, "best_device", lambda: "cpu")

    create_calls = []

    def fake_create(name, pretrained=None):
        create_calls.append(1)
        time.sleep(0.02)
        if len(create_calls) == 1:
            raise RuntimeError("simulated CLIP weight download failure")
        return _FakeClipModel(), None, (lambda img: img)

    _install_fake_open_clip(monkeypatch, fake_create)
    _install_fake_onnxruntime(monkeypatch)

    build_arc_calls = []

    def fake_build_arc(self, providers):
        build_arc_calls.append(1)
        return object()

    monkeypatch.setattr(Embedder, "_build_arc", fake_build_arc)

    embedder = Embedder()
    errors = []
    barrier = threading.Barrier(2)

    def run():
        barrier.wait(timeout=5)
        try:
            embedder._ensure()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert len(create_calls) == 2       # first attempt failed; the concurrent 2nd caller retried
    assert len(build_arc_calls) == 1    # only the successful attempt reached arc construction
    assert embedder._arc is not None and embedder._clip is not None   # fully committed by the retry
