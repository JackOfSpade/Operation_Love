"""Archived screenshot analysis, with explicit failures and no shell command strings.

The BMI entry point currently reports unavailable inference, so this pipeline stops
without publishing an actionable decision until a verified implementation exists.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

if __package__:
    from .analyze_bmi import require_available as require_bmi_available
else:
    from analyze_bmi import require_available as require_bmi_available


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_PATH = SCRIPT_DIR / "beauty_and_BMI_analysis_result.txt"
SCREENSHOTS_DIR = SCRIPT_DIR.parent / "Screenshots"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".gif", ".webp"}
ANALYSIS_TIMEOUT_S = 300


def run_in_env(script_args, env_path):
    """Run an argv sequence with the environment's own interpreter, without activation."""
    if isinstance(script_args, (str, bytes)) or not script_args:
        raise ValueError("script_args must be a nonempty argument sequence")
    env = Path(env_path)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return subprocess.run(
        [str(python), *(str(arg) for arg in script_args)],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=ANALYSIS_TIMEOUT_S,
        check=True,
    ).stdout


def detect_and_crop_face_retinaface(image_path, margin=0.5):
    import cv2
    from retinaface import RetinaFace

    if not math.isfinite(margin) or margin < 0:
        raise ValueError("face margin must be finite and nonnegative")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Image not loaded properly.")
    faces = RetinaFace.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not faces:
        return None
    face = next(iter(faces.values()))
    x, y, x_end, y_end = map(int, face["facial_area"])
    if x_end <= x or y_end <= y:
        raise ValueError("detector returned an empty face box")
    x_margin, y_margin = int((x_end - x) * margin), int((y_end - y) * margin)
    x_start, y_start = max(x - x_margin, 0), max(y - y_margin, 0)
    x_end = min(x_end + x_margin, image.shape[1])
    y_end = min(y_end + y_margin, image.shape[0])
    if x_end <= x_start or y_end <= y_start:
        raise ValueError("detector returned a face outside the image")
    face = image[y_start:y_end, x_start:x_end]
    if not face.size:
        raise ValueError("detector returned a face outside the image")
    return face


def _analysis_output(output, label, path_count):
    """Read the archived analyzer's score followed by its renamed image paths."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    matches = [i for i, line in enumerate(lines) if line.startswith(f"{label}:")]
    if len(matches) != 1:
        raise ValueError(f"analyzer must return exactly one {label} score")
    index = matches[0]
    score = float(lines[index].split(":", 1)[1].strip())
    if not math.isfinite(score) or score <= 0:
        raise ValueError(f"{label} score must be positive and finite")
    paths = [Path(line) for line in lines[index + 1 :]]
    if len(paths) != path_count or not all(path.is_file() for path in paths):
        raise ValueError(
            f"{label} analyzer did not return {path_count} existing image paths"
        )
    return score, paths


def decide(beauty_score, bmi_score):
    """Preserve the archived decision policy only for valid analyzer results."""
    if (
        not math.isfinite(beauty_score)
        or not 1 <= beauty_score <= 5
        or not math.isfinite(bmi_score)
        or bmi_score <= 0
    ):
        raise ValueError("analysis scores are outside their valid ranges")
    baseline_beauty, baseline_bmi = 2.25, 27.39
    beauty_diff = abs((beauty_score - baseline_beauty) / baseline_beauty)
    bmi_diff = 3 * abs((bmi_score - baseline_bmi) / baseline_bmi)
    if beauty_score >= 3.5 and bmi_score <= 18.5:
        return "super_like"
    if (
        (beauty_score >= baseline_beauty and bmi_score <= baseline_bmi)
        or (
            beauty_score < baseline_beauty
            and bmi_score < baseline_bmi
            and bmi_diff >= beauty_diff
        )
        or (
            bmi_score > baseline_bmi
            and beauty_score > baseline_beauty
            and beauty_diff >= bmi_diff
        )
    ):
        return "like"
    return "dislike"


def analyze_image(image_path):
    import cv2

    face = detect_and_crop_face_retinaface(image_path)
    if face is None:
        target = image_path.with_stem(f"{image_path.stem} No Face Detected")
        if target.exists():
            raise FileExistsError(target)
        image_path.rename(target)
        return "No face detected"
    # Do not load a second model or rename source images when the required BMI
    # stage is already known to be unavailable.
    require_bmi_available()
    cropped_dir = image_path.parent / "cropped"
    cropped_dir.mkdir(exist_ok=True)
    cropped = cropped_dir / f"Cropped {uuid.uuid4()}{image_path.suffix}"
    if not cv2.imwrite(str(cropped), face):
        raise OSError(f"could not save face crop: {cropped}")
    beauty_output = run_in_env(
        [SCRIPT_DIR / "analyze_beauty.py", cropped, image_path],
        SCRIPT_DIR / "analyze_beauty.venv",
    )
    beauty, (cropped, original) = _analysis_output(beauty_output, "Beauty Score", 2)
    bmi_output = run_in_env(
        [SCRIPT_DIR / "analyze_bmi.py", cropped, original],
        SCRIPT_DIR / "analyze_bmi.venv",
    )
    bmi, (original,) = _analysis_output(bmi_output, "BMI", 1)
    decision = decide(beauty, bmi)
    destination = image_path.parent / decision
    destination.mkdir(exist_ok=True)
    if (destination / original.name).exists():
        raise FileExistsError(destination / original.name)
    shutil.move(str(original), str(destination))
    return decision


def main():
    # A failed/empty run must never leave a previous profile's LIKE ready for AHK to read.
    RESULT_PATH.write_text("error", encoding="utf-8")
    image_paths = sorted(
        path
        for path in SCREENSHOTS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if len(image_paths) != 1:
        raise ValueError(f"expected exactly one screenshot; found {len(image_paths)}")
    decision = analyze_image(image_paths[0])
    RESULT_PATH.write_text(decision, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)
