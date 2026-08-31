#!/usr/bin/env python3
"""Predict AIGC probabilities for every readable image in a directory.

The output is a JSON array with exactly two fields per image:
``image_path`` and ``pred``. ``pred`` is a calibrated probability in [0, 1].

Example:
    uv run python scripts/predict_directory.py ./images \
        --checkpoint artifacts/model.pt --output predictions.json
"""

from aigc_detector.predict import main


if __name__ == "__main__":
    main()
