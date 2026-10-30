"""Safety tests for the archived ZIP-code batch utility."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _legacy_module(monkeypatch):
    """Load the standalone legacy script without requiring its optional requests extra."""
    fake_requests = types.SimpleNamespace(
        RequestException=type("RequestException", (Exception,), {})
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    path = (
        Path(__file__).parents[1]
        / "legacy/city_name_to_zip_code/city_name_to_zip_code.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_zip_code_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_requests


def test_legacy_google_key_has_no_source_default(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    module, _requests = _legacy_module(monkeypatch)

    assert module.google_api_key == ""


def test_legacy_zip_output_accepts_only_public_postal_code_tokens(monkeypatch):
    module, _requests = _legacy_module(monkeypatch)

    assert module._public_postal_code("02139") == "02139"
    assert module._public_postal_code("02139-1234") == "02139-1234"
    assert module._public_postal_code("api_key=not-for-output") == "Not Found"
    assert module._public_postal_code(None) == "Not Found"


def test_legacy_zip_lookup_rejects_arbitrary_api_response_text(monkeypatch):
    module, fake_requests = _legacy_module(monkeypatch)

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"places": [{"post code": "token=not-for-output"}]}

    fake_requests.get = lambda *_args, **_kwargs: Response()

    assert module.get_zip_code("Cambridge", "Massachusetts", "api-key") == "Not Found"


def test_legacy_main_keeps_public_postal_codes_in_stdout_and_output_file(
    monkeypatch, tmp_path, capsys
):
    module, _requests = _legacy_module(monkeypatch)
    monkeypatch.setattr(module, "cities_input", ["Cambridge, Massachusetts"])
    monkeypatch.setattr(module, "get_public_postal_token", lambda *_args: "02139")
    monkeypatch.chdir(tmp_path)

    module.main()

    assert capsys.readouterr().out == "['02139']\n"
    assert (tmp_path / "zip_codes.txt").read_text() == "['02139']"


def test_legacy_zip_helper_alias_keeps_existing_callers_compatible(monkeypatch):
    module, _requests = _legacy_module(monkeypatch)

    assert module.get_zip_code is module.get_public_postal_token


def test_legacy_zip_lookup_falls_back_after_transport_failure(monkeypatch):
    module, fake_requests = _legacy_module(monkeypatch)
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise fake_requests.RequestException("primary service is unavailable")
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {
                "results": [
                    {
                        "address_components": [
                            {"types": ["postal_code"], "long_name": "02139"}
                        ]
                    }
                ]
            },
        )

    fake_requests.get = get
    assert module.get_zip_code("Cambridge", "Massachusetts", "api-key") == "02139"
    assert len(calls) == 2
    assert all(kwargs["timeout"] == module._REQUEST_TIMEOUT_S for _, kwargs in calls)
    assert calls[1][1]["params"] == {"address": "Cambridge,MA", "key": "api-key"}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "error",
        {"places": None},
        {"places": [None]},
        {"results": [None, {"address_components": [None, {}]}]},
    ],
)
def test_legacy_zip_lookup_tolerates_malformed_payloads(monkeypatch, payload):
    module, fake_requests = _legacy_module(monkeypatch)
    fake_requests.get = lambda *_args, **_kwargs: types.SimpleNamespace(
        status_code=200, json=lambda: payload
    )
    assert module.get_zip_code("Cambridge", "Massachusetts", "api-key") == "Not Found"


@pytest.mark.parametrize("failure", ["network", "json"])
def test_legacy_zip_lookup_failure_keeps_credentials_out_of_output(
    monkeypatch, capsys, failure
):
    module, fake_requests = _legacy_module(monkeypatch)

    def fail():
        error = fake_requests.RequestException if failure == "network" else ValueError
        raise error("private-api-key must remain private")

    fake_requests.get = lambda *_args, **_kwargs: (
        fail()
        if failure == "network"
        else types.SimpleNamespace(status_code=200, json=fail)
    )
    assert (
        module.get_zip_code("Cambridge", "Massachusetts", "private-api-key")
        == "Not Found"
    )
    assert capsys.readouterr() == ("", "")


def test_legacy_zip_lookup_uses_a_valid_postal_code_after_malformed_places(monkeypatch):
    module, fake_requests = _legacy_module(monkeypatch)
    fake_requests.get = lambda *_args, **_kwargs: types.SimpleNamespace(
        status_code=200,
        json=lambda: {"places": [None, {}, {"post code": "02139"}]},
    )
    assert module.get_zip_code("Cambridge", "Massachusetts", "") == "02139"
