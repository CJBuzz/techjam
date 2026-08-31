import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "kaggle" / "extract_wildfake.py"
SPEC = importlib.util.spec_from_file_location("extract_kaggle_wildfake", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class KaggleWildFakeExtractionTests(unittest.TestCase):
    def test_view_groups_are_clean_one_then_two_operations(self) -> None:
        groups = [MODULE.view_group("one-image", repeat, 42) for repeat in range(3)]
        self.assertEqual(groups[0], "clean")
        self.assertEqual(len(groups[1].split("+")), 1)
        self.assertEqual(len(groups[2].split("+")), 2)
        self.assertEqual(groups, [MODULE.view_group("one-image", repeat, 42) for repeat in range(3)])

    def test_diverse_selector_round_robins_groups(self) -> None:
        rows = [{"generator": name, "real_source": name} for name in ("a", "a", "b", "b", "c", "c")]
        chosen = MODULE.select_diverse(rows, target=3, key="generator", cap=2, seed=42)
        self.assertEqual({row["generator"] for row in chosen}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
