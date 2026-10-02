"""Safety tests for the archived ZIP-code batch utility."""

import importlib.util
import sys
import types
from pathlib import Path


def _legacy_module(monkeypatch):
    """Load the standalone legacy script without requiring its optional requests extra."""
    fake_requests = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    path = Path(__file__).parents[1] / "legacy/city_name_to_zip_code/city_name_to_zip_code.py"
    spec = importlib.util.spec_from_file_location("legacy_zip_code_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, fake_requests


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
