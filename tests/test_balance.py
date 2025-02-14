"""Filename: test_balance.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestBalance(unittest.TestCase):
    """Test Balance metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

    def test_iris(self):
        """Test balance metrics using Iris dataset."""
        model = ComplexityProfile(measures=["B1", "B2"])
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.B1(), 2.220446e-16, atol=0)
        numpy.testing.assert_allclose(model.B2(), 0.0, atol=0)

    def test_iris_100(self):
        """Test balance metrics using Iris dataset first 100 rows."""
        model = ComplexityProfile(measures=["B1", "B2"])
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.B1(), 0.0, atol=0)
        numpy.testing.assert_allclose(model.B2(), 0.0, atol=0)


class TestBalance2(unittest.TestCase):
    """Test Balance metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

    def test_cancer(self):
        """Test balance metrics with cancer dataset."""
        model = ComplexityProfile(measures=["B1", "B2"])
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.B1(), 0.047365, atol=1e-4)
        numpy.testing.assert_allclose(model.B2(), 0.12196, atol=1e-4)

    def test_cancer_100(self):
        """Test balance metrics with first 100 rows on cancer dataset."""
        model = ComplexityProfile(measures=["B1", "B2"])
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.B1(), 0.07, atol=1e-2)
        numpy.testing.assert_allclose(model.B2(), 0.165138, atol=1e-6)
