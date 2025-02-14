"""Filename: test_neighborhood.py."""

import unittest

import numpy
from sklearn.datasets import load_breast_cancer, load_iris

from dcp import ComplexityProfile


class TestNeighborhood(unittest.TestCase):
    """Test neighborhood metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_iris(return_X_y=True)

        self.features = features
        self.target = target

    def test_iris(self):
        """Test using the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.N1(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.N3(), 0.053333, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.7, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.106667, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.804356, atol=1e-06)

    def test_iris_100(self):
        """Test using the first 100 rows of the Iris dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.N1(), 0.01, atol=0)
        numpy.testing.assert_allclose(model.N3(), 0.0, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.5, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.03, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.5066, atol=1e-06)


class TestNeighborhood2(unittest.TestCase):
    """Test neighborhood metrics."""

    def setUp(self):
        """Setup."""
        features, target = load_breast_cancer(return_X_y=True)

        self.features = features
        self.target = target

    def test_cancer(self):
        """Test using cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features, self.target)

        numpy.testing.assert_allclose(model.N1(), 0.086116, atol=1e-06)
        numpy.testing.assert_allclose(model.N3(), 0.091388, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.534271, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.003515, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.912293, atol=1e-06)

    def test_cancer_100(self):
        """Test with first 100 rows of cancer dataset."""
        model = ComplexityProfile()
        model.fit(self.features[:100], self.target[:100])

        numpy.testing.assert_allclose(model.N1(), 0.1, atol=0)
        numpy.testing.assert_allclose(model.N3(), 0.08, atol=1e-06)
        numpy.testing.assert_allclose(model.N4(), 0.5, atol=1e-01)
        numpy.testing.assert_allclose(model.N5(), 0.02, atol=1e-06)
        numpy.testing.assert_allclose(model.N6(), 0.7953, atol=1e-06)
