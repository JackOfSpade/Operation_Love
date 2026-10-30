"""Archived BMI entry point, retained only to report its unavailable model honestly."""

import argparse


class BMIUnavailableError(RuntimeError):
    """The archived pipeline has no configured, verified BMI model."""


def require_available():
    raise BMIUnavailableError(
        "BMI inference is unavailable: this archived pipeline has no configured, "
        "verified model. No score or decision was produced."
    )


def main(cropped_image_path, uncropped_image_path):
    require_available()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cropped_image_path")
    parser.add_argument("uncropped_image_path")
    args = parser.parse_args()
    try:
        main(args.cropped_image_path, args.uncropped_image_path)
    except BMIUnavailableError as exc:
        parser.exit(1, f"{exc}\n")
