"""Privacy checks for the reference-only persistent browser driver."""
from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from operation_love.drivers.web.playwright_base import PlaywrightDriver
from operation_love.private_files import UnsafePrivatePathError


class _Page:
    def set_default_timeout(self, _timeout):
        pass

    def goto(self, _url, *, wait_until):
        assert wait_until == "domcontentloaded"


class _Context:
    def __init__(self):
        self.pages = [_Page()]

    def close(self):
        pass


class _Playwright:
    def __init__(self, on_start):
        self._on_start = on_start

    def start(self):
        self._on_start()
        return self

    def stop(self):
        pass


class _Driver(PlaywrightDriver):
    def __init__(self, user_data_dir, on_start):
        cfg = SimpleNamespace(apps={"test": {"user_data_dir": str(user_data_dir)}})
        super().__init__(
            cfg,
            config_key="test",
            default_url="https://example.invalid",
            default_user_data_dir=str(user_data_dir),
            default_debug_dir=str(user_data_dir.parent / "debug"),
        )
        self._fake_context = _Context()
        self._on_start = on_start

    def _import_playwright(self):
        return (lambda: _Playwright(self._on_start)), "test"

    def _launch_context(self, _launch_kwargs):
        return self._fake_context

    def next_profile(self, *, should_stop=None):
        return None

    def like(self, opener=None, item_index=None, *, model_item_index=None, should_stop=None):
        pass

    def dislike(self):
        pass

    def out_of_profiles(self):
        return False


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_open_session_tightens_only_persistent_profile_leaf_before_browser_start(tmp_path):
    broad_parent = tmp_path / "shared"
    profile = broad_parent / "browser-profile"
    broad_parent.mkdir(mode=0o755)
    profile.mkdir(mode=0o755)
    broad_parent.chmod(0o755)
    profile.chmod(0o755)

    observed_modes = []
    driver = _Driver(profile, lambda: observed_modes.append(stat.S_IMODE(profile.stat().st_mode)))
    try:
        driver.open_session()
    finally:
        driver.close()

    assert observed_modes == [0o700]
    assert stat.S_IMODE(profile.stat().st_mode) == 0o700
    assert stat.S_IMODE(broad_parent.stat().st_mode) == 0o755


@pytest.mark.skipif(os.name != "posix", reason="symlink semantics")
def test_open_session_rejects_symlink_profile_before_browser_start(tmp_path):
    target = tmp_path / "unrelated-profile"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    profile_link = tmp_path / "browser-profile"
    profile_link.symlink_to(target, target_is_directory=True)
    starts = []

    with pytest.raises(UnsafePrivatePathError):
        _Driver(profile_link, lambda: starts.append(True)).open_session()

    assert starts == []
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
