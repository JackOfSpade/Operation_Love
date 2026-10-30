"""Offline failure and compatibility checks for the archived screenshot analyzers."""

import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from legacy.beauty_and_BMI_analysis import analyze_beauty, analyze_bmi, evaluate_images


@pytest.fixture
def analysis_paths(monkeypatch, tmp_path):
    screenshots = tmp_path / "Screenshots"
    screenshots.mkdir()
    result = tmp_path / "result.txt"
    result.write_text("like")
    monkeypatch.setattr(evaluate_images, "SCREENSHOTS_DIR", screenshots)
    monkeypatch.setattr(evaluate_images, "RESULT_PATH", result)
    return screenshots, result


def test_unavailable_bmi_never_fabricates_a_score_or_renames_files(tmp_path):
    cropped, original = tmp_path / "crop.png", tmp_path / "original.png"
    cropped.write_bytes(b"crop")
    original.write_bytes(b"original")
    with pytest.raises(analyze_bmi.BMIUnavailableError, match="unavailable"):
        analyze_bmi.main(cropped, original)
    assert cropped.read_bytes() == b"crop"
    assert original.read_bytes() == b"original"


def test_unavailable_bmi_cli_fails_without_loading_ml_dependencies():
    completed = subprocess.run(
        [sys.executable, "-S", analyze_bmi.__file__, "crop.png", "original.png"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "inference is unavailable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_run_in_env_uses_direct_argv_and_propagates_subprocess_failure(
    monkeypatch, tmp_path
):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.CalledProcessError(2, argv, output="", stderr="model failed")

    monkeypatch.setattr(evaluate_images.subprocess, "run", run)
    args = [tmp_path / "analyze script.py", "a 'quoted'; image.png", "original.png"]
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_images.run_in_env(args, tmp_path / "model env")
    argv, kwargs = calls[0]
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    assert argv == [str(tmp_path / "model env" / executable), *map(str, args)]
    assert kwargs["check"] is True
    assert kwargs["timeout"] == evaluate_images.ANALYSIS_TIMEOUT_S
    assert kwargs["cwd"] == evaluate_images.SCRIPT_DIR
    assert "shell" not in kwargs


@pytest.mark.parametrize("args", ["python analyze.py", b"python analyze.py", []])
def test_run_in_env_refuses_command_strings(args):
    with pytest.raises(ValueError, match="argument sequence"):
        evaluate_images.run_in_env(args, "env")


@pytest.mark.parametrize("image_count", [0, 2])
def test_empty_or_ambiguous_screenshot_input_retires_stale_decision(
    analysis_paths, image_count
):
    screenshots, result = analysis_paths
    for index in range(image_count):
        (screenshots / f"{index}.png").write_bytes(b"image")
    with pytest.raises(ValueError, match="exactly one screenshot"):
        evaluate_images.main()
    assert result.read_text() == "error"


def test_failed_analyzer_never_publishes_like_or_overwrites_source(
    monkeypatch, analysis_paths
):
    screenshots, result = analysis_paths
    source = screenshots / "original.png"
    source.write_bytes(b"original")
    monkeypatch.setattr(evaluate_images, "require_bmi_available", lambda: None)
    monkeypatch.setattr(
        evaluate_images,
        "detect_and_crop_face_retinaface",
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
    )

    def fail(*args):
        raise subprocess.CalledProcessError(1, ["analyzer"])

    monkeypatch.setattr(evaluate_images, "run_in_env", fail)
    with pytest.raises(subprocess.CalledProcessError):
        evaluate_images.main()
    assert result.read_text() == "error"
    assert source.read_bytes() == b"original"


def test_failed_crop_write_stops_before_analyzer(monkeypatch, tmp_path):
    import cv2

    source = tmp_path / "original.png"
    source.write_bytes(b"original")
    monkeypatch.setattr(evaluate_images, "require_bmi_available", lambda: None)
    monkeypatch.setattr(
        evaluate_images,
        "detect_and_crop_face_retinaface",
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *a: False)
    with pytest.raises(OSError, match="could not save"):
        evaluate_images.analyze_image(source)
    assert source.read_bytes() == b"original"


def test_unavailable_bmi_stops_before_crop_writes_and_beauty_inference(
    monkeypatch, analysis_paths
):
    screenshots, result = analysis_paths
    source = screenshots / "original.png"
    source.write_bytes(b"original")
    monkeypatch.setattr(
        evaluate_images,
        "detect_and_crop_face_retinaface",
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    with pytest.raises(analyze_bmi.BMIUnavailableError):
        evaluate_images.main()
    assert result.read_text() == "error"
    assert source.read_bytes() == b"original"
    assert not (screenshots / "cropped").exists()


def test_no_face_result_preserves_legacy_token(monkeypatch, analysis_paths):
    screenshots, result = analysis_paths
    (screenshots / "original.png").write_bytes(b"original")
    monkeypatch.setattr(
        evaluate_images, "detect_and_crop_face_retinaface", lambda _: None
    )
    evaluate_images.main()
    assert result.read_text() == "No face detected"
    assert (screenshots / "original No Face Detected.png").read_bytes() == b"original"


@pytest.mark.parametrize(
    ("beauty", "bmi", "expected"),
    [
        (4, 18, "super_like"),
        (3, 22, "like"),
        (1, 35, "dislike"),
        (1.5, 18, "like"),
        (5, 28, "like"),
    ],
)
def test_valid_scores_preserve_archived_decision_policy(beauty, bmi, expected):
    assert evaluate_images.decide(beauty, bmi) == expected


@pytest.mark.parametrize(
    ("beauty", "bmi"), [(math.nan, 22), (3, math.inf), (3, -1), (0, 22), (6, 22)]
)
def test_invalid_scores_never_produce_decisions(beauty, bmi):
    with pytest.raises(ValueError):
        evaluate_images.decide(beauty, bmi)


@pytest.mark.parametrize(
    "output", ["", "BMI: nan", "BMI: 22\nmissing.png", "BMI: 22\nBMI: 23"]
)
def test_missing_malformed_and_nonfinite_analyzer_outputs_fail(output):
    with pytest.raises(ValueError):
        evaluate_images._analysis_output(output, "BMI", 1)


def _beauty_predictor(monkeypatch, score):
    monkeypatch.setitem(
        sys.modules,
        "ComboLoss.main.inference",
        SimpleNamespace(
            FacialBeautyPredictor=lambda _: SimpleNamespace(
                infer=lambda _: {"beauty": score}
            )
        ),
    )


@pytest.mark.parametrize("score", [math.nan, math.inf, 0, 6])
def test_beauty_analyzer_rejects_invalid_scores_without_renaming(
    monkeypatch, tmp_path, score
):
    _beauty_predictor(monkeypatch, score)
    cropped, original = tmp_path / "crop.png", tmp_path / "original.png"
    cropped.write_bytes(b"crop")
    original.write_bytes(b"original")
    with pytest.raises(ValueError, match="outside"):
        analyze_beauty.main(cropped, original)
    assert cropped.read_bytes() == b"crop"
    assert original.read_bytes() == b"original"


def test_beauty_analyzer_rolls_back_first_rename_when_second_fails(
    monkeypatch, tmp_path
):
    _beauty_predictor(monkeypatch, 3)
    cropped, original = tmp_path / "crop.png", tmp_path / "original.png"
    cropped.write_bytes(b"crop")
    original.write_bytes(b"original")
    rename = Path.rename

    def fail_original(path, target):
        if path == original:
            raise PermissionError("read-only original")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_original)
    with pytest.raises(PermissionError):
        analyze_beauty.main(cropped, original)
    assert cropped.read_bytes() == b"crop"
    assert original.read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "crop.png",
        "original.png",
    ]


def test_successful_analyzer_protocol_moves_only_the_current_image(
    monkeypatch, analysis_paths
):
    screenshots, result = analysis_paths
    source = screenshots / "original.png"
    source.write_bytes(b"original")
    monkeypatch.setattr(evaluate_images, "require_bmi_available", lambda: None)
    monkeypatch.setattr(
        evaluate_images,
        "detect_and_crop_face_retinaface",
        lambda _: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    stages = []

    def analyze(args, env):
        stages.append(args[0].name)
        cropped, original = map(Path, args[1:])
        assert cropped.is_file() and original.is_file()
        if args[0].name == "analyze_beauty.py":
            return f"Beauty Score: 3.00\n{cropped}\n{original}\n"
        return f"BMI: 22.00\n{original}\n"

    monkeypatch.setattr(evaluate_images, "run_in_env", analyze)
    evaluate_images.main()
    assert stages == ["analyze_beauty.py", "analyze_bmi.py"]
    assert result.read_text() == "like"
    assert not source.exists()
    assert (screenshots / "like" / "original.png").read_bytes() == b"original"


def test_beauty_wrapper_preserves_its_score_and_two_path_protocol(
    monkeypatch, tmp_path, capsys
):
    _beauty_predictor(monkeypatch, 3)
    cropped, original = tmp_path / "crop.png", tmp_path / "original.png"
    cropped.write_bytes(b"crop")
    original.write_bytes(b"original")
    analyze_beauty.main(cropped, original)
    score, paths = evaluate_images._analysis_output(
        capsys.readouterr().out, "Beauty Score", 2
    )
    assert score == 3
    assert [path.read_bytes() for path in paths] == [b"crop", b"original"]
    assert not cropped.exists() and not original.exists()


def test_beauty_wrapper_does_not_overwrite_existing_annotated_image(
    monkeypatch, tmp_path
):
    _beauty_predictor(monkeypatch, 3)
    cropped, original = tmp_path / "crop.png", tmp_path / "original.png"
    cropped.write_bytes(b"crop")
    original.write_bytes(b"original")
    existing = original.with_stem("original  Beauty Score 3.00")
    existing.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        analyze_beauty.main(cropped, original)
    assert cropped.read_bytes() == b"crop"
    assert original.read_bytes() == b"original"
    assert existing.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "box", [(-100, -100, -50, -50), (200, 200, 250, 250), (10, 10, 5, 5)]
)
def test_face_detector_rejects_out_of_frame_and_reversed_boxes(monkeypatch, box):
    import cv2

    monkeypatch.setattr(
        cv2, "imread", lambda _: np.zeros((100, 100, 3), dtype=np.uint8)
    )
    monkeypatch.setitem(
        sys.modules,
        "retinaface",
        SimpleNamespace(
            RetinaFace=SimpleNamespace(
                detect_faces=lambda _: {"face": {"facial_area": box}}
            )
        ),
    )
    with pytest.raises(ValueError, match="face"):
        evaluate_images.detect_and_crop_face_retinaface("image.png")


def test_face_detector_clips_partial_edge_box_without_wrapped_coordinates(monkeypatch):
    import cv2

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda _: image)
    monkeypatch.setitem(
        sys.modules,
        "retinaface",
        SimpleNamespace(
            RetinaFace=SimpleNamespace(
                detect_faces=lambda _: {"face": {"facial_area": (-10, -10, 20, 20)}}
            )
        ),
    )
    crop = evaluate_images.detect_and_crop_face_retinaface("image.png", margin=0)
    assert crop.shape == (20, 20, 3)
