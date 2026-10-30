"""Run the actual polling functions against deliberately overlapping responses."""

import json
import re
import shutil
import subprocess

import pytest

from operation_love.hub.page import _PAGE


NODE_BIN = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE_BIN is None, reason="node is not available")


def _js_function(name):
    match = re.search(
        rf"^(?:async )?function {name}\([^\n]*?\)\s*{{",
        _PAGE,
        re.MULTILINE,
    )
    assert match is not None
    depth = 0
    for index in range(match.end() - 1, len(_PAGE)):
        if _PAGE[index] == "{":
            depth += 1
        elif _PAGE[index] == "}":
            depth -= 1
            if depth == 0:
                return _PAGE[match.start() : index + 1]
    raise AssertionError(f"unbalanced function {name}")


def _polling_script(channel):
    function = {
        "status": "tick",
        "eval": "tickEval",
        "logs": "tickLogs",
        "training": "tickTrainingCheckpoint",
    }[channel]
    functions = [function] + (["_nearBottom"] if channel == "logs" else [])
    source = [_js_function(name) for name in functions]
    declarations = re.findall(
        r"^let _(?:status|eval|logs|training)(?:Applied)?Request = 0;",
        _PAGE,
        re.MULTILINE,
    )
    declarations += re.findall(
        r"^let _training(?:PollContext|Checkpoint) = [^;]+;", _PAGE, re.MULTILINE
    )
    declarations += re.findall(r"^let _hubConnectionLost = false;", _PAGE, re.MULTILINE)
    training_stub = (
        "" if channel == "training" else "function tickTrainingCheckpoint(){}\n"
    )
    poll = (
        "()=>tickTrainingCheckpoint(JSON.parse(JSON.stringify(snapshot)))"
        if channel == "training"
        else function
    )
    return (
        "\n".join(declarations + source)
        + "\nconst renders=[], resolvers=[], rejecters=[];\n"
        "function getJSON(){return new Promise((resolve,reject)=>{resolvers.push(resolve);rejecters.push(reject);});}\n"
        "function renderGlobal(s){renders.push(s.version);}\n"
        "function renderRunStatus(){}\n"
        + training_stub
        + "const snapshot={running:true,status:{run_id:'run',apps:{hinge:{app:'hinge',mode:'training'}}}};\n"
        "function selectRunApps(status){return Object.values(status.apps||{});}\n"
        "function activeTrainingCheckpoint(payload){return payload;}\n"
        "function latestTrainingResult(){return null;}\n"
        "function renderTrainingCheckpointFromPoll(s){_trainingCheckpoint=s;renders.push(s.version);}\n"
        "function renderTrainingCheckpoint(s){_trainingCheckpoint=s||null;renders.push(s?s.version:null);}\n"
        "function renderTrainingFeedback(){renders.push('stopping');}\n"
        "function clearHubConnectionLost(){}\n"
        "function renderHubConnectionLost(){renders.push('disconnected');}\n"
        "function renderEval(s){renders.push(s.version);}\n"
        "const logs={scrollHeight:1,scrollTop:0,clientHeight:1,set textContent(v){renders.push(v);}};\n"
        "function $(selector){return selector==='#logs'?logs:{};}\n"
        f"const poll={poll};\n"
        "function payload(version){return {version,lines:[version]};}\n"
    )


def _run(script):
    result = subprocess.run(
        [NODE_BIN, "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("channel", ["status", "eval", "logs", "training"])
def test_slow_polls_make_progress_while_newer_requests_are_pending(channel):
    script = (
        _polling_script(channel)
        + """
(async()=>{
  let pending=poll();
  for(let index=0;index<3;index++){
    const next=poll();
    resolvers[index](payload(String(index+1)));
    await pending;
    pending=next;
  }
  resolvers[3](payload('4'));
  await pending;
  console.log(JSON.stringify(renders));
})();
"""
    )
    expected = ["1", "2", "3", "4"]
    assert _run(script) == ([None] + expected if channel == "training" else expected)


@pytest.mark.parametrize("channel", ["status", "eval", "logs", "training"])
def test_old_responses_never_replace_a_newer_completed_response(channel):
    script = (
        _polling_script(channel)
        + """
(async()=>{
  const first=poll(), second=poll();
  resolvers[1](payload('new'));
  await second;
  resolvers[0](payload('old'));
  await first;
  console.log(JSON.stringify(renders));
})();
"""
    )
    assert _run(script) == ([None, "new"] if channel == "training" else ["new"])


def test_old_status_failure_does_not_hide_a_newer_success():
    script = (
        _polling_script("status")
        + """
(async()=>{
  const first=poll(), second=poll();
  resolvers[1](payload('new'));
  await second;
  rejecters[0](new Error('old failure'));
  await first;
  console.log(JSON.stringify(renders));
})();
"""
    )
    assert _run(script) == ["new"]


def test_old_status_success_does_not_clear_a_newer_connection_failure():
    script = (
        _polling_script("status")
        + """
(async()=>{
  const first=poll(), second=poll();
  rejecters[1](new Error('new failure'));
  await second;
  resolvers[0](payload('old'));
  await first;
  console.log(JSON.stringify(renders));
})();
"""
    )
    assert _run(script) == ["disconnected"]


@pytest.mark.parametrize(
    "replace_context",
    [
        "snapshot.status.run_id='new-run';",
        "snapshot.status.apps={bumble:{app:'bumble',mode:'training'}};",
    ],
)
def test_training_context_change_immediately_invalidates_old_responses(replace_context):
    script = (
        _polling_script("training")
        + "(async()=>{const first=poll();"
        + replace_context
        + """
  const second=poll();
  resolvers[0](payload('old'));
  await first;
  const before=renders.slice();
  resolvers[1](payload('new'));
  await second;
  console.log(JSON.stringify({before,after:renders}));
})();
"""
    )
    assert _run(script) == {"before": [None, None], "after": [None, None, "new"]}


@pytest.mark.parametrize(
    ("retire_context", "expected"),
    [
        ("snapshot.status.apps={};", None),
        ("snapshot.status.stopping=true;", "stopping"),
    ],
)
def test_training_stop_or_completion_cannot_be_replaced_by_a_late_checkpoint(
    retire_context, expected
):
    script = (
        _polling_script("training")
        + "(async()=>{const first=poll();"
        + retire_context
        + """
  await poll();
  resolvers[0](payload('late actionable checkpoint'));
  await first;
  console.log(JSON.stringify(renders));
})();
"""
    )
    assert _run(script) == [None, expected]


def test_completed_run_with_retained_status_immediately_retires_its_checkpoint():
    script = (
        _polling_script("training")
        + """
(async()=>{
  const first=poll();
  snapshot.running=false; // completed Hub snapshots retain their run_id and app statuses
  const completed=poll();
  resolvers[0](payload('late actionable checkpoint'));
  await first;
  const before=renders.slice();
  resolvers[1](payload('completed run checkpoint'));
  await completed;
  console.log(JSON.stringify({before,checkpoint:_trainingCheckpoint,renders}));
})();
"""
    )
    assert _run(script) == {
        "before": [None, None],
        "checkpoint": None,
        "renders": [None, None, None],
    }


def test_status_disconnect_invalidates_a_pending_checkpoint_until_recovery():
    functions = "\n".join(
        _js_function(name)
        for name in ("tick", "renderHubConnectionLost", "clearHubConnectionLost")
    )
    script = (
        _polling_script("training")
        + functions
        + "\nfunction $(selector){return null;}\n"
        + """
(async()=>{
  const checkpoint=tickTrainingCheckpoint(snapshot);
  const status=tick();
  rejecters[1](new Error('status unavailable'));
  await status;
  resolvers[0](payload('stale actionable checkpoint'));
  await checkpoint;
  const lost={renders:renders.slice(),checkpoint:_trainingCheckpoint,latch:_hubConnectionLost};
  const recovered=tick();
  resolvers[2](snapshot);
  await recovered;
  resolvers[3](payload('fresh checkpoint'));
  await Promise.resolve();
  console.log(JSON.stringify({lost,latest:_trainingCheckpoint.version,latch:_hubConnectionLost}));
})();
"""
    )
    assert _run(script) == {
        "lost": {"renders": [None, None], "checkpoint": None, "latch": True},
        "latest": "fresh checkpoint",
        "latch": False,
    }
