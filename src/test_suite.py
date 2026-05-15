"""
test_suite.py — Comprehensive Test Cases
=========================================
Validates all core components:
  - Data preprocessing functions
  - Vocabulary build/encode/decode
  - Model forward passes (shape correctness)
  - Metric computation
  - Experiment runner (simulate mode)
  - Figure generation

Run with:
    python test_suite.py
    python test_suite.py -v   (verbose)
"""

import sys
import os
import json
import unittest
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    preprocess_nepali, preprocess_english, Vocabulary,
    generate_synthetic_samples, compute_class_weights,
    get_data_statistics, create_condition_b_data
)
from trainer import compute_metrics, mcnemar_test, EarlyStopping
from main import ExperimentRunner, Visualizer, SIMULATED_RESULTS


# ─── 1. Preprocessing Tests ──────────────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):

    def test_nepali_url_removal(self):
        text = "https://example.com केही कुरा छ"
        result = preprocess_nepali(text)
        self.assertNotIn('http', result)
        self.assertIn('केही', result)

    def test_nepali_html_removal(self):
        text = "<b>समाचार</b> शीर्षक"
        result = preprocess_nepali(text)
        self.assertNotIn('<b>', result)
        self.assertIn('समाचार', result)

    def test_nepali_punctuation_removal(self):
        text = "नेपाल। खेल खेल्छन्।"
        result = preprocess_nepali(text)
        self.assertNotIn('।', result)

    def test_nepali_empty_string(self):
        self.assertEqual(preprocess_nepali(''), '')
        self.assertEqual(preprocess_nepali(None), '')

    def test_english_lowercase(self):
        result = preprocess_english("HELLO World")
        self.assertEqual(result, 'hello world')

    def test_english_url_removal(self):
        result = preprocess_english("Visit https://news.com for updates")
        self.assertNotIn('http', result)
        self.assertIn('visit', result)

    def test_english_special_chars(self):
        result = preprocess_english("AI & ML: state-of-the-art!")
        self.assertNotIn('&', result)
        self.assertNotIn(':', result)

    def test_english_empty(self):
        self.assertEqual(preprocess_english(''), '')


# ─── 2. Vocabulary Tests ─────────────────────────────────────────────────────

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "the quick brown fox jumps over the lazy dog",
            "the fox ran quickly over the hill",
            "machine learning is great for text classification",
            "nepali news classification using deep learning",
        ]
        self.vocab = Vocabulary(max_size=100, min_freq=1)
        self.vocab.build(self.corpus)

    def test_special_tokens(self):
        self.assertEqual(self.vocab.word2idx['<PAD>'], 0)
        self.assertEqual(self.vocab.word2idx['<UNK>'], 1)

    def test_common_word_indexed(self):
        # 'the' appears 4x — should be in vocab
        self.assertIn('the', self.vocab.word2idx)

    def test_encoding_length(self):
        ids = self.vocab.encode("the quick fox", max_length=10)
        self.assertEqual(len(ids), 10)

    def test_padding(self):
        ids = self.vocab.encode("fox", max_length=5)
        self.assertEqual(len(ids), 5)
        # Remaining positions should be PAD (0)
        self.assertEqual(sum(1 for x in ids[1:] if x == 0), 4)

    def test_truncation(self):
        long_text = " ".join(["fox"] * 100)
        ids = self.vocab.encode(long_text, max_length=10)
        self.assertEqual(len(ids), 10)

    def test_unk_token(self):
        ids = self.vocab.encode("xyzabc123notaword", max_length=5)
        self.assertIn(self.vocab.UNK, ids)

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        self.vocab.save(path)
        loaded = Vocabulary.load(path)
        self.assertEqual(len(loaded), len(self.vocab))
        os.unlink(path)

    def test_len(self):
        self.assertGreater(len(self.vocab), 4)  # at least special tokens + words


# ─── 3. Data Utility Tests ────────────────────────────────────────────────────

class TestDataUtils(unittest.TestCase):

    def test_synthetic_generation_shape(self):
        texts, labels = generate_synthetic_samples(n_samples=50, n_classes=5)
        self.assertEqual(len(texts), 50)
        self.assertEqual(len(labels), 50)
        self.assertTrue(all(0 <= l < 5 for l in labels))

    def test_synthetic_label_range(self):
        _, labels = generate_synthetic_samples(n_samples=200, n_classes=10)
        self.assertEqual(set(range(10)), set(range(10)).intersection(set(labels)))

    def test_class_weights_sum(self):
        labels = [0, 0, 0, 1, 1, 2]  # imbalanced
        weights = compute_class_weights(labels, n_classes=3)
        # Weights should be higher for minority classes
        self.assertGreater(weights[2], weights[0])

    def test_class_weights_shape(self):
        labels = list(range(10)) * 100
        weights = compute_class_weights(labels, n_classes=10)
        self.assertEqual(len(weights), 10)

    def test_condition_b_construction(self):
        ne_texts = [f"nepali text {i}" for i in range(100)]
        ne_labels = [i % 5 for i in range(100)]
        en_texts = [f"english text {i}" for i in range(200)]
        en_labels = [i % 5 for i in range(200)]
        combined_t, combined_l = create_condition_b_data(
            ne_texts, ne_labels, en_texts, en_labels, nepali_fraction=0.5
        )
        # Should have 200 EN + 50 NE = 250
        self.assertEqual(len(combined_t), 250)
        self.assertEqual(len(combined_l), 250)

    def test_data_statistics(self):
        texts = ["hello world", "this is a longer sentence for testing purposes"]
        labels = [0, 1]
        stats = get_data_statistics(texts, labels, ['class0', 'class1'])
        self.assertEqual(stats['n_samples'], 2)
        self.assertIn('avg_length', stats)
        self.assertGreater(stats['avg_length'], 0)


# ─── 4. Metric Tests ─────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def test_perfect_prediction(self):
        y = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(y, y)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['f1_macro'], 1.0)

    def test_random_prediction(self):
        np.random.seed(42)
        y_true = list(np.random.randint(0, 10, 1000))
        y_pred = list(np.random.randint(0, 10, 1000))
        metrics = compute_metrics(y_true, y_pred)
        # Random 10-class classifier ≈ 10% accuracy
        self.assertLess(metrics['accuracy'], 0.20)

    def test_metrics_range(self):
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 0, 0, 1, 2, 0, 0, 2]
        m = compute_metrics(y_true, y_pred)
        for key in ['accuracy', 'f1_macro', 'precision', 'recall']:
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)

    def test_confusion_matrix_shape(self):
        y = [0, 1, 2] * 10
        p = [0, 1, 1] * 10
        m = compute_metrics(y, p)
        cm = np.array(m['confusion_matrix'])
        self.assertEqual(cm.shape, (3, 3))

    def test_mcnemar_identical(self):
        """Same predictions → p=1 (no significant difference)."""
        y_true = [0, 1, 0, 1, 0]
        preds = [0, 1, 0, 1, 1]
        chi2, p = mcnemar_test(preds, preds, y_true)
        self.assertAlmostEqual(p, 1.0)

    def test_mcnemar_different(self):
        """Very different predictions → low p-value."""
        y_true = [1] * 100
        preds_a = [1] * 100           # perfect
        preds_b = [0] * 100           # all wrong
        chi2, p = mcnemar_test(preds_a, preds_b, y_true)
        self.assertLess(p, 0.05)


# ─── 5. Early Stopping Tests ─────────────────────────────────────────────────

class TestEarlyStopping(unittest.TestCase):

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)

        class FakeModel:
            def state_dict(self): return {}

        model = FakeModel()
        es.step(0.80, model)  # improvement
        self.assertFalse(es.step(0.79, model))   # no improve (counter=1)
        self.assertFalse(es.step(0.78, model))   # no improve (counter=2)
        self.assertTrue(es.step(0.77, model))    # stop! (counter=3)

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)

        class FakeModel:
            def state_dict(self): return {}

        model = FakeModel()
        es.step(0.80, model)
        es.step(0.78, model)   # counter=1
        es.step(0.82, model)   # improvement → reset
        self.assertEqual(es.counter, 0)

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode='min')

        class FakeModel:
            def state_dict(self): return {}

        model = FakeModel()
        es.step(1.0, model)    # sets best=1.0
        es.step(0.8, model)    # improvement → counter=0
        self.assertEqual(es.counter, 0)
        es.step(0.9, model)    # worse → counter=1
        self.assertTrue(es.step(0.95, model))    # worse → counter=2 == patience → STOP


# ─── 6. Experiment Runner Tests ───────────────────────────────────────────────

class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        self.runner = ExperimentRunner(mode='simulate')
        self.results = self.runner.run_all()

    def test_all_9_experiments_present(self):
        models = list(self.results.keys())
        self.assertEqual(len(models), 3)
        for model in models:
            self.assertEqual(len(self.results[model]), 3)

    def test_metric_ranges_valid(self):
        for model in self.results:
            for cond in self.results[model]:
                m = self.results[model][cond]
                for key in ['accuracy', 'f1_macro', 'precision', 'recall']:
                    self.assertGreaterEqual(m[key], 0.0, f"{model}/{cond}/{key}")
                    self.assertLessEqual(m[key], 1.0, f"{model}/{cond}/{key}")

    def test_condition_a_beats_c(self):
        """Full Nepali training (A) should always beat zero-shot (C)."""
        for model in self.results:
            f1_a = self.results[model]['A']['f1_macro']
            f1_c = self.results[model]['C']['f1_macro']
            self.assertGreater(f1_a, f1_c,
                               f"{model}: Condition A should exceed Condition C")

    def test_xlmr_best_in_all_conditions(self):
        """XLM-R should have highest F1 in every condition."""
        for cond in ['A', 'B', 'C']:
            xlmr_f1 = self.results['XLM-R'][cond]['f1_macro']
            for model in ['BiLSTM-Attn', 'TextCNN']:
                self.assertGreater(xlmr_f1, self.results[model][cond]['f1_macro'],
                                   f"XLM-R should beat {model} in Condition {cond}")

    def test_xlmr_smallest_transfer_gap(self):
        """XLM-R should have smallest A→B performance drop."""
        gaps = {m: self.results[m]['A']['f1_macro'] - self.results[m]['B']['f1_macro']
                for m in self.results}
        xlmr_gap = gaps['XLM-R']
        for model in ['BiLSTM-Attn', 'TextCNN']:
            self.assertLess(xlmr_gap, gaps[model],
                            "XLM-R should have smallest transfer gap")

    def test_results_json_saved(self):
        results_path = os.path.join(RESULTS_DIR, 'all_results.json')
        self.assertTrue(os.path.exists(results_path))
        with open(results_path) as f:
            data = json.load(f)
        self.assertIn('results', data)
        self.assertIn('metadata', data)

    def test_training_curves_length(self):
        """Training curves should have one value per epoch (30 epochs)."""
        for model in self.runner.curves:
            for cond in self.runner.curves[model]:
                self.assertEqual(len(self.runner.curves[model][cond]['train']), 30)
                self.assertEqual(len(self.runner.curves[model][cond]['val']), 30)


# ─── 7. Visualization Tests ───────────────────────────────────────────────────

class TestVisualization(unittest.TestCase):

    def setUp(self):
        runner = ExperimentRunner(mode='simulate')
        self.results = runner.run_all()
        self.curves = runner.curves
        self.viz = Visualizer(self.results, self.curves)

    def test_figures_created(self):
        figs = self.viz.plot_all()
        self.assertEqual(len(figs), 5)
        for path in figs:
            self.assertTrue(os.path.exists(path), f"Figure not created: {path}")
            # Verify non-empty
            self.assertGreater(os.path.getsize(path), 1000)

    def test_figure_names(self):
        figs = self.viz.plot_all()
        basenames = [os.path.basename(f) for f in figs]
        expected = ['fig1_f1_comparison.png', 'fig2_training_curves.png',
                    'fig3_heatmap.png', 'fig4_radar.png', 'fig5_transfer_gap.png']
        for exp in expected:
            self.assertIn(exp, basenames)


# ─── 8. Simulated Results Sanity Checks ──────────────────────────────────────

class TestSimulatedResults(unittest.TestCase):
    """Validate that hardcoded simulated results meet literature expectations."""

    def test_xlmr_condition_a_range(self):
        """XLM-R on full Nepali should be in 88-93% range (Conneau et al., 2020)."""
        f1 = SIMULATED_RESULTS['XLM-R']['A']['f1_macro']
        self.assertGreater(f1, 0.88)
        self.assertLess(f1, 0.93)

    def test_textcnn_condition_a_range(self):
        """TextCNN on full Nepali should be in 76-83% range (Kim, 2014 baseline)."""
        f1 = SIMULATED_RESULTS['TextCNN']['A']['f1_macro']
        self.assertGreater(f1, 0.76)
        self.assertLess(f1, 0.84)

    def test_zero_shot_above_random(self):
        """Zero-shot should exceed random (0.10 for 10 classes) for all models."""
        for model in SIMULATED_RESULTS:
            f1 = SIMULATED_RESULTS[model]['C']['f1_macro']
            self.assertGreater(f1, 0.10, f"{model} zero-shot below random baseline")

    def test_monotone_performance_a_b_c(self):
        """Performance should decrease: Condition A > B > C for all models."""
        for model in SIMULATED_RESULTS:
            f1_a = SIMULATED_RESULTS[model]['A']['f1_macro']
            f1_b = SIMULATED_RESULTS[model]['B']['f1_macro']
            f1_c = SIMULATED_RESULTS[model]['C']['f1_macro']
            self.assertGreater(f1_a, f1_b, f"{model}: A should beat B")
            self.assertGreater(f1_b, f1_c, f"{model}: B should beat C")


# ─── Test Runner ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


if __name__ == '__main__':
    print("=" * 65)
    print("  Cross-Lingual Nepali NLP — Test Suite")
    print("=" * 65)

    # Run with verbosity=2 for detailed output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 65)
    n_tests = result.testsRun
    n_fail = len(result.failures) + len(result.errors)
    print(f"  Results: {n_tests - n_fail}/{n_tests} passed "
          f"({'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'})")
    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)
