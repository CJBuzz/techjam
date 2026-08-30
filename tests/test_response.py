import unittest

import numpy as np
import torch
from PIL import Image

from aigc_detector.response import (
    RESPONSE_FEATURE_DIM,
    ResponseHead,
    perturbation_views,
    response_features,
    restore_best_response_state,
)


class ResponseTests(unittest.TestCase):
    def test_four_deterministic_probe_views(self) -> None:
        image = Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), "RGB")
        first = perturbation_views(image, 42, "sample")
        second = perturbation_views(image, 42, "sample")
        self.assertEqual(len(first), 4)
        self.assertTrue(all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second)))

    def test_response_vector_and_tiny_identity_head(self) -> None:
        features = torch.ones(2, 4, 3072)
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]])
        response = response_features(features, logits)
        self.assertEqual(tuple(response.shape), (2, RESPONSE_FEATURE_DIM))
        self.assertTrue(torch.equal(response[:, 0], logits[:, 0]))
        self.assertTrue(torch.allclose(response[:, 11:], torch.zeros(2, 9), atol=1e-6))
        head = ResponseHead()
        self.assertLess(sum(parameter.numel() for parameter in head.parameters()), 100_000)
        self.assertTrue(torch.equal(head.eval()(response), logits[:, 0]))

    def test_restore_best_state_puts_head_in_eval_mode(self) -> None:
        head = ResponseHead()
        best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        head.train()
        restore_best_response_state(head, best_state)
        self.assertFalse(head.training)


if __name__ == "__main__":
    unittest.main()
