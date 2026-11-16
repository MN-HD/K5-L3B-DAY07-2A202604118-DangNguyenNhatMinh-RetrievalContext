import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from day7_pipeline import metadata_matches, normalize_evidence, NEW_TEST_QUESTIONS, UNANSWERABLE_CASES


class Day7PipelineTests(unittest.TestCase):
    def test_metadata_filter(self):
        chunk = {"metadata": {"chapter_id": "ch14", "page_id": "p14"}}
        self.assertTrue(metadata_matches(chunk, {"chapter_id": "ch14"}))
        self.assertFalse(metadata_matches(chunk, {"chapter_id": "ch15"}))

    def test_filter_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            metadata_matches({"metadata": {}}, {"password": "x"})

    def test_day6_evidence_whitespace_normalization(self):
        self.assertEqual(normalize_evidence("one\n  two\tthree"), "one two three")

    def test_fresh_test_question_set(self):
        self.assertEqual(len(NEW_TEST_QUESTIONS), 5)
        self.assertEqual(len({q["id"] for q in NEW_TEST_QUESTIONS}), 5)
        self.assertTrue(all(q["split"] == "test" and q["gold"] for q in NEW_TEST_QUESTIONS))

    def test_unanswerable_split(self):
        self.assertEqual(sum(c["split"] == "dev" for c in UNANSWERABLE_CASES), 2)
        self.assertEqual(sum(c["split"] == "test" for c in UNANSWERABLE_CASES), 2)
        self.assertTrue(all(c["answerable"] is False for c in UNANSWERABLE_CASES))


if __name__ == "__main__":
    unittest.main()
