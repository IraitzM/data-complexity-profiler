"""Filename: test_smoothness.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestSmoothness(unittest.TestCase):
    """Test smoothness."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

        self.measures = ["S1", "S2", "S3", "S4"]

    def test_iris(self):
        """Test using the Iris dataset."""
        model = ComplexityProfile(self.measures)
        model.fit(self.features, self.target)

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.036891, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.524347, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.031914, atol=1e-4)

    def test_iris_100(self):
        """Test using the first 100 rows of the Iris dataset."""
        model = ComplexityProfile(self.measures)
        model.fit(self.features[:100], self.target[:100])

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.006723, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.523209, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.0, atol=1e-4)


class TestSmoothness2(unittest.TestCase):
    """Test smoothness metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

        self.measures = ["S1", "S2", "S3"]

    def test_cancer(self):
        """Test using cancer dataset."""
        model = ComplexityProfile(self.measures)
        model.fit(self.features, self.target)

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.290675, atol=1e-4)
        numpy.testing.assert_allclose(result["S2"], 0.844673, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.356, atol=1e-4)

    def test_cancer_100(self):
        """Test with first 100 rows of cancer dataset."""
        model = ComplexityProfile(self.measures)
        model.fit(self.features[:100], self.target[:100])

        result = model.transform()

        numpy.testing.assert_allclose(result["S1"], 0.300428, atol=1e-2)
        numpy.testing.assert_allclose(result["S2"], 0.853594, atol=1e-4)
        numpy.testing.assert_allclose(result["S3"], 0.308994, atol=1e-4)
