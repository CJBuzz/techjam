import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.streaming_cache import (
    bounded_stream_sample,
    audit_stream_cache_metadata,
    completed_cache_state,
    load_stream_feature_cache,
    paired_views,
    repair_stream_cache_metadata,
    save_feature_chunk,
)
from aigc_detector.train import merge_balanced_feature_sets


class StreamingCacheTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict]:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "gray").save(buffer, format="PNG")
        image_data = buffer.getvalue()
        return [
            {
                "image_data": image_data, "image_name": f"a-{index}.png", "label": 1,
                "model_name": "model-a", "real_source": None, "subset": "train",
                "architecture": "diffusion",
            }
            for index in range(5)
        ] + [
            {
                "image_data": image_data, "image_name": f"b-{index}.png", "label": 1,
                "model_name": "model-b", "real_source": None, "subset": "train",
                "architecture": "diffusion",
            }
            for index in range(5)
        ] + [
            {
                "image_data": image_data, "image_name": f"r-{index}.png", "label": 0,
                "model_name": None, "real_source": f"source-{index % 2}", "subset": "train",
                "architecture": None,
            }
            for index in range(6)
        ]

    def test_bounded_sampler_and_generator_quota(self) -> None:
        selected = list(bounded_stream_sample(self.rows(), 4, 4, 2, 2))
        self.assertEqual(sum(row["label"] == 0 for row in selected), 4)
        self.assertEqual(sum(row["label"] == 1 for row in selected), 4)
        self.assertLessEqual(sum(row["model_name"] == "model-a" for row in selected), 2)

    def test_sampler_yields_early_and_reports_progress(self) -> None:
        consumed = 0
        reports = []

        def rows():
            nonlocal consumed
            for row in self.rows():
                consumed += 1
                yield row

        sampler = bounded_stream_sample(
            rows(), 1, 1, 1, 1, progress_every=1, progress_callback=reports.append
        )
        first = next(sampler)
        self.assertEqual(first["label"], 1)
        self.assertEqual(consumed, 1)
        self.assertEqual(reports[-1]["inspected"], 1)
        self.assertEqual(reports[-1]["accepted_fake"], 1)
        self.assertIn("rejected_real_quota", reports[-1])
        self.assertIn("rejected_fake_quota", reports[-1])

    def test_smoke_counts_complete_with_one_real_source(self) -> None:
        rows = self.rows()[:10] + [
            {**row, "image_name": f"one-source-{index}.png", "real_source": "only-source"}
            for index, row in enumerate(self.rows()[-6:] + self.rows()[-4:])
        ]
        selected = list(bounded_stream_sample(rows, 10, 10, 5, 10))
        self.assertEqual(sum(row["label"] == 0 for row in selected), 10)
        self.assertEqual(sum(row["label"] == 1 for row in selected), 10)

    def test_real_quota_relaxes_and_records_state(self) -> None:
        fake_rows = self.rows()[:10]
        real_template = self.rows()[-1]
        real_rows = [
            {**real_template, "image_name": f"real-{index}.png", "real_source": "only-source"}
            for index in range(20)
        ]
        state = {}
        selected = list(bounded_stream_sample(
            fake_rows + real_rows, 10, 10, 10, 5,
            relax_after_no_progress=3, max_inspected_rows=100, sampler_state=state,
        ))
        self.assertEqual(sum(row["label"] == 0 for row in selected), 10)
        self.assertTrue(state["real_quota_relaxed"])
        self.assertGreaterEqual(state["rejected_real_quota"], 3)
        self.assertEqual(state["real_source_counts"], {"only-source": 10})
        self.assertEqual(state["fake_model_counts"], {"model-a": 5, "model-b": 5})
        self.assertTrue(state["complete"])
        json.dumps({"sampling_state": state})

    def test_impossible_stream_stops_at_inspection_guard(self) -> None:
        fake = self.rows()[0]
        rows = [{**fake, "image_name": f"fake-{index}.png"} for index in range(20)]
        with self.assertRaisesRegex(RuntimeError, "max-inspected-rows=10"):
            list(bounded_stream_sample(
                rows, 1, 1, 20, 1, relax_after_no_progress=2, max_inspected_rows=10
            ))

    def test_cache_has_paired_metadata_and_no_raw_images(self) -> None:
        selected = next(bounded_stream_sample(self.rows(), 1, 1, 1, 1))
        self.assertNotIn("image", selected)
        images, metadata = paired_views(selected, robust_views=1, seed=42)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(image.mode == "RGB" for image in images))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = {"forensic_mode": "laplacian_fft"}
            (directory / "manifest.json").write_text(
                '{"model_config":{"forensic_mode":"laplacian_fft"},"robust_views":1}',
                encoding="utf-8",
            )
            save_feature_chunk(directory / "chunk-00000.pt", torch.zeros(2, 4), metadata)
            payload = torch.load(directory / "chunk-00000.pt", weights_only=True)
            self.assertNotIn("image", payload)
            self.assertTrue(all("image" not in row and "image_data" not in row for row in payload["metadata"]))
            self.assertEqual(payload["metadata"][0]["image_name"], selected["image_name"])
            self.assertEqual(payload["metadata"][0]["architecture"], "diffusion")
            features, labels, groups, indices, originals, _ = load_stream_feature_cache(directory, config)
            self.assertEqual(tuple(features.shape), (2, 4))
            self.assertEqual((len(labels), len(groups), len(indices), originals), (2, 2, 2, 1))
            self.assertEqual(groups[0], "clean")
            self.assertEqual(completed_cache_state(directory, robust_views=1), (1, 1))

    def test_smoke_uses_small_shuffle_buffer(self) -> None:
        script = (Path(__file__).parents[1] / "scripts" / "run_track5_overnight.sh").read_text(
            encoding="utf-8"
        )
        smoke = script.split("--smoke)", 1)[1].split(';;', 1)[0]
        self.assertIn("--buffer-size 64", smoke)
        self.assertIn("--max-real-per-source 10", smoke)
        self.assertNotIn("--buffer-size 10000", smoke)

    def test_streamed_features_are_merged_into_training_features(self) -> None:
        merged_x, merged_y, groups, indices, originals = merge_balanced_feature_sets(
            torch.tensor([[1.0], [2.0], [11.0], [12.0]]),
            torch.tensor([0, 1, 0, 1]), ["clean", "clean", "jpeg", "jpeg"], 2,
            torch.tensor([[3.0], [13.0]]), torch.tensor([1, 1]), ["clean", "jpeg"], 1,
        )
        self.assertEqual(merged_x.flatten().tolist(), [1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
        self.assertEqual(merged_y.tolist(), [0, 1, 1, 0, 1, 1])
        self.assertEqual(groups, ["clean", "clean", "clean", "jpeg", "jpeg", "jpeg"])
        self.assertEqual(indices.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(originals, 3)

    def test_metadata_repair_excludes_colliding_complete_view_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = {"forensic_mode": "laplacian_fft"}
            (directory / "manifest.json").write_text(json.dumps({
                "model_config": config, "robust_views": 1,
            }), encoding="utf-8")
            metadata = [
                {"original_id": "collision", "repeat": repeat, "label": 1,
                 "transform_group": "clean" if repeat == 0 else "jpeg_q90"}
                for repeat in (0, 1, 0, 1)
            ]
            torch.save({"features": torch.arange(8.0).view(4, 2), "metadata": metadata},
                       directory / "chunk-00000.pt")
            audit = audit_stream_cache_metadata(directory)
            self.assertEqual(audit["invalid_selected_originals"], 2)
            self.assertEqual(audit["invalid_records"], 4)
            repair = repair_stream_cache_metadata(directory)
            self.assertEqual(repair["excluded_original_ids"], ["collision"])
            self.assertEqual(len(list(directory.glob("manifest.backup-pre-metadata-repair-*.json"))), 1)
            with self.assertRaisesRegex(ValueError, "No completed feature chunks"):
                load_stream_feature_cache(directory, config)


if __name__ == "__main__":
    unittest.main()
