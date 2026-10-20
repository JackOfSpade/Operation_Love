"""Exercise the real zsh launcher without installing models or starting the hub."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh launcher needs zsh")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_uses_existing_environment_and_preserves_exit_status(
    tmp_path, exit_code
):
    source = Path(__file__).resolve().parents[1] / "Operation Love.command"
    launcher = tmp_path / source.name
    shutil.copy2(source, launcher)
    (tmp_path / "pyproject.toml").write_text("# fixture metadata\n")
    environment = tmp_path / ".venv"
    (environment / "bin").mkdir(parents=True)
    interpreter = environment / "bin/python"
    interpreter.write_text(
        '#!/bin/sh\n[ "$*" = "-m operation_love hub" ] || exit 99\n'
        f'printf "hub entry point reached\\n"\nexit {exit_code}\n'
    )
    interpreter.chmod(0o755)
    stamp = environment / ".oplove-deps-stamp"
    stamp.touch()
    os.utime(tmp_path / "pyproject.toml", (1, 1))
    result = subprocess.run(
        [shutil.which("zsh"), str(launcher)],
        check=False,
        input="",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert "hub entry point reached" in result.stdout
    assert "Dependencies already up to date" in result.stdout
