"""Archived beauty-model wrapper; paths and scores are validated before publication."""

import argparse
import math
from pathlib import Path


def main(cropped_image_path, uncropped_image_path):
    from ComboLoss.main.inference import FacialBeautyPredictor

    cropped, original = Path(cropped_image_path), Path(uncropped_image_path)
    if (
        not cropped.is_file()
        or not original.is_file()
        or cropped.resolve() == original.resolve()
    ):
        raise ValueError("analysis requires two distinct existing image files")
    model_path = (
        Path(__file__).resolve().parent / "ComboLoss/models/ComboNet_SCUTFBP5500.pth"
    )
    predictor = FacialBeautyPredictor(str(model_path))
    score = float(predictor.infer(str(cropped))["beauty"])
    if not math.isfinite(score) or not 1 <= score <= 5:
        raise ValueError("beauty model returned a score outside [1, 5]")
    new_cropped = cropped.with_stem(f"{cropped.stem}  Beauty Score {score:.2f}")
    new_original = original.with_stem(f"{original.stem}  Beauty Score {score:.2f}")
    if new_cropped.exists() or new_original.exists():
        raise FileExistsError("annotated image destination already exists")
    cropped.rename(new_cropped)
    try:
        original.rename(new_original)
    except OSError:
        new_cropped.rename(cropped)
        raise
    print(f"Beauty Score: {score:.2f}")
    print(new_cropped.as_posix())
    print(new_original.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cropped_image_path")
    parser.add_argument("uncropped_image_path")
    args = parser.parse_args()
    main(args.cropped_image_path, args.uncropped_image_path)
