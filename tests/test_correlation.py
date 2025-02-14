"""Filename: test_imbalance.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestCorrelation(unittest.TestCase):
    """Test correlation metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

    def test_iris(self):
        """Test correlation metrics with Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.C1(), 1.0, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.0, atol=1e-06)

    def test_iris_100(self):
        """Test correlation metrics with Iris dataset first 100 rows."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.C1(), 1.0, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0, atol=1e-06)


class TestCorrelation2(unittest.TestCase):
    """Test correlation metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

    def test_cancer(self):
        """Test correlation metrics with cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.C1(), 0.952635, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.12196, atol=1e-06)

    def test_cancer_100(self):
        """Test with first 100 rows of cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.C1(), 0.934068, atol=1e-06)
        numpy.testing.assert_allclose(model.C2(), 0.165138, atol=1e-06)
