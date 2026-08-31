import json
import tempfile
import unittest
from pathlib import Path

from aigc_detector.dashboard import load_results, resolve_image


class DashboardContractTests(unittest.TestCase):
    def test_exact_submission_schema_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            output.write_text(json.dumps([{"image_path": "images/a.png", "pred": 0.25}]))
            self.assertEqual(load_results(output)[0]["pred"], 0.25)

    def test_extra_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            output.write_text(json.dumps([{"image_path": "a.png", "pred": 0.25, "label": "real"}]))
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_results(output)

    def test_image_filename_can_resolve_under_images_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            expected = images / "a.png"
            expected.touch()
            self.assertEqual(resolve_image("a.png", root, images), expected.resolve())


if __name__ == "__main__":
    unittest.main()
