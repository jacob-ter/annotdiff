"""annotdiff —— 自检测试。

按样本比对两批标注，列出分歧点、一致率与分歧最集中的类别。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import annotdiff  # noqa: E402


class TestAnnotDiff(unittest.TestCase):
    def test_identical_annotations_have_full_agreement(self):
        a = [{"id": "s1", "label": "x"}, {"id": "s2", "label": "y"}]
        b = [{"id": "s1", "label": "x"}, {"id": "s2", "label": "y"}]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertEqual(r["divergence_count"], 0)
        self.assertAlmostEqual(r["agreement"]["exact"], 1.0)

    def test_divergence_is_located(self):
        a = [{"id": "s1", "label": "x", "category": "c1"},
             {"id": "s2", "label": "y", "category": "c2"}]
        b = [{"id": "s1", "label": "x", "category": "c1"},
             {"id": "s2", "label": "z", "category": "c2"}]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertEqual(r["divergence_count"], 1)
        self.assertEqual(r["divergences"][0]["id"], "s2")

    def test_divergence_category_is_ranked(self):
        a = [{"id": f"s{i}", "label": "x", "category": "退款"} for i in range(3)]
        b = [{"id": f"s{i}", "label": "y", "category": "退款"} for i in range(3)]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertEqual(r["categories"][0]["category"], "退款")
        self.assertEqual(r["categories"][0]["divergences"], 3)

    def test_kappa_is_computed(self):
        a = [{"id": f"s{i}", "label": "x"} for i in range(20)]
        b = [{"id": f"s{i}", "label": "x"} for i in range(20)]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertIsNotNone(r["agreement"]["cohen_kappa"])

    def test_multilabel_jaccard(self):
        a = [{"id": "s1", "label": ["x", "y"]}]
        b = [{"id": "s1", "label": ["x", "y"]}]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertAlmostEqual(r["agreement"]["jaccard"], 1.0)

    def test_only_in_one_batch_is_reported(self):
        a = [{"id": "s1", "label": "x"}]
        b = [{"id": "s2", "label": "x"}]
        r = annotdiff.find_divergences(a, b, "id", "label", "category")
        self.assertEqual(r["only_a"], ["s1"])
        self.assertEqual(r["only_b"], ["s2"])
