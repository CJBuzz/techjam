import tempfile
import unittest
from types import SimpleNamespace

import torch

from aigc_detector.phase3.r6 import (
    HEAD_MODES,
    LocalHead,
    PatchDetector,
    build_patch_detector,
    extract_global_and_patches,
    select_candidate,
    topk_count,
)


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(patch_size=4); self.proj = torch.nn.Linear(3, 6)
    def forward(self, pixel_values, output_hidden_states=False, return_dict=True, interpolate_pos_encoding=False):
        batch, _, height, width = pixel_values.shape
        count = (height // 4) * (width // 4)
        base = self.proj(pixel_values.mean((2, 3)))
        tokens = torch.cat((base[:, None], base[:, None].repeat(1, count, 1)), 1)
        return SimpleNamespace(pooler_output=base, last_hidden_state=tokens)


class R6Tests(unittest.TestCase):
    def test_patch_and_global_extraction(self):
        global_representation, patches = extract_global_and_patches(ToyBackbone(), torch.randn(2, 3, 8, 8))
        self.assertEqual(global_representation.shape, (2, 6)); self.assertEqual(patches.shape, (2, 4, 6))

    def test_topk_edge_cases(self):
        self.assertEqual(topk_count(1, 0.1), 1); self.assertEqual(topk_count(10, 1.0), 10)
        with self.assertRaises(ValueError): topk_count(0, 0.1)
        with self.assertRaises(ValueError): topk_count(4, 0.0)

    def test_attention_weights_normalize(self):
        head = LocalHead(6, "attention_pool"); _, _, details = head(torch.randn(3, 5, 6))
        self.assertTrue(torch.allclose(details["patch_weights"].sum(1), torch.ones(3)))

    def test_global_only_reproduces_baseline(self):
        backbone = ToyBackbone(); baseline = torch.nn.Module()
        baseline.backbone = backbone; baseline.classifier = torch.nn.Linear(6, 1)
        state = {"state_dict": baseline.state_dict()}
        rebuilt = build_patch_detector(ToyBackbone(), 6, state, "global_only", "topk_patch", 0.1)
        pixels = torch.randn(2, 3, 8, 8)
        expected = baseline.classifier(extract_global_and_patches(baseline.backbone, pixels)[0]).squeeze(1)
        self.assertTrue(torch.allclose(rebuilt(pixels), expected))

    def test_all_heads_small_and_no_target_input(self):
        for mode in HEAD_MODES:
            model = PatchDetector(ToyBackbone(), 6, mode)
            self.assertEqual(model(torch.randn(2, 3, 8, 8)).shape, (2,))
            head_parameters = sum(parameter.numel() for parameter in model.classifier.parameters())
            self.assertLess(head_parameters, 1000)
        self.assertNotIn("label", PatchDetector.forward.__code__.co_varnames)
        self.assertNotIn("target", PatchDetector.forward.__code__.co_varnames)

    def test_validation_only_selection(self):
        summaries = []
        for mode, worst in (("global_only", .90), ("mean_patch", .91), ("topk_patch", .92)):
            row = {"candidate_id": mode, "head_mode": mode, "status": "succeeded",
                   "clean_validation_balanced_accuracy": .965,
                   "worst_transformed_validation_balanced_accuracy": worst,
                   "mean_transformed_validation_balanced_accuracy": worst + .01,
                   "inference_multiplier": 1, "total_deployment_parameter_count": 100}
            summaries.append({"selection_split": "validation", "final_test_evaluated": False, "results": [row]})
        self.assertEqual(select_candidate(summaries, .9681)["selected_head_mode"], "topk_patch")
        summaries[0]["selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"): select_candidate(summaries, .9681)

    def test_artifacts_do_not_cache_tokens(self):
        # The production metadata contract explicitly records online extraction.
        self.assertNotIn("save", extract_global_and_patches.__code__.co_names)


if __name__ == "__main__": unittest.main()
