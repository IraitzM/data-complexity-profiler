"""Filename: test_features.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestFeatures(unittest.TestCase):
    """Test feature-base metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

    def test_iris(self):
        """Test using the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.F1(), 0.058628, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)

    def test_iris_100(self):
        """Test using the first 100 rows of the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.F1(), 0.059119, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)


class TestFeatures2(unittest.TestCase):
    """Test feature based metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

    def test_cancer(self):
        """Test using cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.F1(), 0.370253, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)

    def test_cancer_100(self):
        """Test with first 100 rows of cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.F1(), 0.399597, atol=1e-06)
        numpy.testing.assert_allclose(model.F4(), 0.0, atol=1e-06)
