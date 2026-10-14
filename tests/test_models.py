"""Unit tests for sequence preprocessing, GAF encoding, and evaluation metrics."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import create_sequences, create_gaf, chronological_split
from src.evaluation import calculate_metrics, backtest_strategy, create_walk_forward_folds


class TestDeepLearningPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.sample_returns = np.random.normal(0.0005, 0.01, 200)

    def test_create_sequences_shape(self):
        lookback = 20
        horizon = 5
        X, y = create_sequences(self.sample_returns, lookback=lookback, horizon=horizon)
        expected_len = len(self.sample_returns) - lookback - horizon + 1
        self.assertEqual(len(X), expected_len)
        self.assertEqual(X.shape[1], lookback)
        self.assertEqual(y.shape[1], horizon)

    def test_chronological_split(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, test_size=0.25)
        self.assertEqual(len(X_tr), 75)
        self.assertEqual(len(X_te), 25)
        self.assertTrue(np.all(X_tr[-1] < X_te[0]))

    def test_create_gaf_properties(self):
        seq = np.linspace(-0.05, 0.05, 15)
        gaf = create_gaf(seq)
        self.assertEqual(gaf.shape, (15, 15))
        self.assertTrue(np.all(gaf >= -1.0 - 1e-6) and np.all(gaf <= 1.0 + 1e-6))
        # Gramian matrix must be symmetric
        self.assertTrue(np.allclose(gaf, gaf.T))

    def test_calculate_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertGreater(metrics['R2'], 0.9)

    def test_backtest_strategy(self):
        preds = np.array([1, 0, 1, 1, 0])
        realized = np.array([0.02, -0.01, 0.015, -0.005, 0.03])
        res = backtest_strategy(preds, realized)
        self.assertIn('Total Return', res)
        self.assertIn('Sharpe Ratio', res)
        self.assertEqual(len(res['Equity Curve']), len(realized))

    def test_walk_forward_folds(self):
        folds = create_walk_forward_folds(n_samples=1000, train_size=400, test_size=200)
        self.assertGreater(len(folds), 0)
        for tr_idx, te_idx in folds:
            self.assertEqual(len(tr_idx), 400)
            self.assertEqual(len(te_idx), 200)
            self.assertEqual(max(tr_idx) + 1, min(te_idx))


if __name__ == '__main__':
    unittest.main()
