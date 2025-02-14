"""Filename: test_linearity.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestLinearity(unittest.TestCase):
    """Test linearity metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

    def test_iris(self):
        """Test using the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.L1(), 0.01, atol=1e-2)

    def test_iris_100(self):
        """Test using the first 100 rows of the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.L1(), 0.0, atol=0)


class TestLinearity2(unittest.TestCase):
    """Test linearity metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

    def test_cancer(self):
        """Test using cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.L1(), 0.032313, atol=1e-2)

    def test_cancer_100(self):
        """Test with first 100 rows of cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.L1(), 0.009901, atol=1e-2)
