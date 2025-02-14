"""Main entrypoint for data complexity module."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from . import utils


class ComplexityProfile(BaseEstimator):
    """ComplexityProfile class.

    Args:
        BaseEstimator (_type_): _description_
    """

    def __init__(self, measures="all"):
        """Class initialization.

        Args:
            measures (str, optional): Measures to be computed.
                Defaults to "all".
        """
        # Select measures
        self.measures = self.ls_measures(measures)

        # Initialized
        self.x = None
        self.y = None
        self.d = None

        # Extra
        self.data = None
        self.dst = None
        self.dst_matrix = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """Fitting function.

        Args:
            x (pd.DataFrame): Features
            y (pd.DataFrame): Target

        Raises:
            ValueError: Number of examples in the minority
                class should be >= 2.
            ValueError: Label attribute needs to be numeric.
            ValueError: x and y must have same number of rows.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        y = pd.Series(y)
        if y.value_counts().min() < 2:  # noqa: PLR2004
            raise ValueError(
                "Number of examples in the minority class should be >= 2"
            )

        if isinstance(y, pd.Categorical):
            raise ValueError("label attribute needs to be numeric")

        if len(x) != len(y):
            raise ValueError("x and y must have same number of rows")

        x.columns = [f"feat_{i}" for i in range(x.shape[1])]

        self.data = pd.concat([utils.binarize(x), y.rename("class")], axis=1)
        self.dst = pdist(x)
        self.dst_matrix = squareform(self.dst)

        x = utils.normalize(x)
        y = utils.normalize(pd.DataFrame(y)).iloc[:, 0]

        sorted_indices = np.argsort(y)
        self.x = x.iloc[sorted_indices].reset_index(drop=True)
        self.y = y.iloc[sorted_indices].reset_index(drop=True)

        self.d = squareform(pdist(x))

    def transform(self, return_type: str = "dict") -> dict | pd.DataFrame:
        """Main function that returns the required metrics.

        Args:
            return_type (str, optional): How to produce the output information.
                Defaults to "dict".

        Returns:
            dict | pd.DataFrame: Metric structures
        """
        result = {}
        for measure in self.measures:
            method = getattr(self, f"{measure}")
            measure_result = method()
            result[measure] = float(np.mean(measure_result))

        if return_type == "df":
            return pd.DataFrame.from_records([result])

        return result

    @staticmethod
    def ls_measures(measures: list[str] | str) -> list[str]:
        """List available measures.

        Args:
            measures (list[str]|str): List of measures or
            family of measures

        Returns:
            list[str]: Returns the list of available measures
        """
        measure_dict = {
            # Feature based
            "features": ["F1", "F1v", "F2", "F3"],
            # Linearity
            "linear": ["L1"],
            # Neighborhood
            "neighborhood": ["N1", "N2", "N3", "N4"],  # "N5", "N6"
            # Balance
            "balance": ["B1", "B2"],
            # Smoothness
            "smoothness": ["S1", "S2", "S3", "S4"],
            # Correlation
            "correlation": ["C1", "C2"],
            # Dimensionality
            "dimensionality": ["T2", "T3", "T4"],
        }

        # Check list of measures
        all_measures = [m for m_list in measure_dict.values() for m in m_list]
        if isinstance(measures, list):
            if set(measures).issubset(set(all_measures)):
                return measures

        if measures in measure_dict:
            return measure_dict[measures]
        elif isinstance(measures, str) and measures in all_measures:
            return [measures]

        return all_measures

    # Feature based
    def branch(self, j: int) -> pd.DataFrame:
        """Check the data for a given class.

        Args:
            j (int): Class identifier

        Returns:
            DataFrame: Rows for a given class
        """
        return self.data[self.data["class"] == j].drop("class", axis=1)

    def F1(self):
        """Maximum Fisher's Discriminant Ratio (F1).

        Returns:
            float: F1
        """
        features = self.data.drop("class", axis=1)
        overall_mean = features.mean()

        # Numerator
        numerator = sum(
            len(self.branch(clss))
            * (self.branch(clss).mean() - overall_mean) ** 2
            for clss in self.data["class"].unique()
        )

        # Denominator
        denominator = sum(
            ((self.branch(clss) - self.branch(clss).mean()) ** 2).sum()
            for clss in self.data["class"].unique()
        )

        # Get max of all fi
        max_ri = 0.0
        for n, d in zip(numerator, denominator):
            if d == 0.0:
                max_ri = np.inf
            elif n / d > max_ri:
                max_ri = n / d

        return 1 / (max_ri + 1)

    def F1v(self):
        """The Directional-vector Maximum Fisher's Discriminant Ratio.

        Uses one-vs-one for multiclass problems.
        """

        def dvector(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            c1 = a.mean()
            c2 = b.mean()

            w = (len(a) / len(data)) * a.cov() + (len(b) / len(data)) * b.cov()
            b2 = np.outer(c1 - c2, c1 - c2)
            d = np.linalg.pinv(w) @ (c1 - c2)

            return (d.T @ b2 @ d) / (d.T @ w @ d)

        ovo_data = utils.ovo(self.data)
        f1v = [dvector(data) for data in ovo_data]
        return 1 / (np.array(f1v) + 1)

    def F2(self):
        """Value of overlapping region."""

        def region_over(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            maxmax = np.maximum(a.max(), b.max())
            minmin = np.minimum(a.min(), b.min())

            over = np.maximum(
                np.minimum(maxmax, b.max()) - np.maximum(minmin, a.min()), 0
            )
            rang = maxmax - minmin
            return np.prod(over / rang)

        ovo_data = utils.ovo(self.data)
        return [region_over(data) for data in ovo_data]

    def F3(self):
        def non_overlap(data):
            classes = data["class"].unique()
            a = self.branch(classes[0])
            b = self.branch(classes[1])

            minmax = np.minimum(a.max(), b.max())
            maxmin = np.maximum(a.min(), b.min())

            return (
                (data.drop("class", axis=1) < maxmin)
                | (data.drop("class", axis=1) > minmax)
            ).sum() / len(data)

        ovo_data = utils.ovo(self.data)
        f3 = [non_overlap(data) for data in ovo_data]
        return 1 - np.max(f3, axis=0)

    def F4(self):
        def removing(data):
            while True:
                non_overlap = (
                    (
                        data.drop("class", axis=1)
                        < data.drop("class", axis=1).min()
                    )
                    | (
                        data.drop("class", axis=1)
                        > data.drop("class", axis=1).max()
                    )
                ).sum()
                col = non_overlap.idxmax()
                data = data[data[col] == False].drop(col, axis=1)  # noqa: E712

                if (
                    len(data) == 0
                    or len(data.columns) == 1
                    or len(data["class"].unique()) == 1
                ):
                    break

            return data

        ovo_data = utils.ovo(self.data)
        return [len(removing(data)) / len(data) for data in ovo_data]

    # Smoothness

    def S1(self):
        g = nx.from_numpy_array(self.d)
        tree = nx.minimum_spanning_tree(g)
        edges = list(tree.edges())
        aux = np.abs(
            self.y.iloc[[e[0] for e in edges]].values
            - self.y.iloc[[e[1] for e in edges]].values
        )
        return aux / (aux + 1)

    def S2(self):
        pred = self.d[range(len(self.d) - 1), range(1, len(self.d))]
        return pred / (pred + 1)

    def S3(self):
        np.fill_diagonal(self.d, np.inf)
        pred = self.y.iloc[np.argmin(self.d, axis=1)].values
        aux = (pred - self.y.values) ** 2
        return aux / (aux + 1)

    def S4(self):
        test = self.r_generate()
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(self.x, self.y)
        pred = knn.predict(test.iloc[:, :-1])
        aux = (pred - test.iloc[:, -1].values) ** 2
        return aux / (aux + 1)

    def r_interpolation(self, i):
        aux = self.x.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        for j in range(len(self.x.columns)):
            if np.issubdtype(self.x.iloc[:, j].dtype, np.number):
                aux.iloc[0, j] += (aux.iloc[1, j] - aux.iloc[0, j]) * rnd
            else:
                aux.iloc[0, j] = np.random.choice(aux.iloc[:, j])

        tmp = self.y.iloc[(i - 1) : i + 1].copy()
        rnd = np.random.uniform()
        tmp.iloc[0] = tmp.iloc[0] * rnd + tmp.iloc[1] * (1 - rnd)

        return pd.concat([aux.iloc[0], pd.Series([tmp.iloc[0]], index=["y"])])

    def r_generate(self):
        n = len(self.x)

        tmp = pd.DataFrame([self.r_interpolation(i) for i in range(1, n)])
        tmp.columns = [*list(self.x.columns), "y"]
        return tmp

    # Correlation
    def C1(self):
        n = len(self.data["class"])
        class_proportion = self.data["class"].value_counts()

        nc = 0
        proportions = 0.0
        for _, v in class_proportion.items():
            proportions += (v / n) * np.log2(v / n)
            nc += 1

        # Calculate entropy
        return -(1 / np.log2(nc)) * proportions

    def C2(self):
        n = len(self.data["class"])
        class_proportion = self.data["class"].value_counts()

        nc = 0
        sumation = 0.0
        for _, v in class_proportion.items():
            sumation += v / (n - v)
            nc += 1

        ir = (nc - 1) / nc * sumation
        return 1 - 1 / ir

    # Linearity
    def L1(self) -> float:
        """Sum of error distance by linear programming.

        Returns:
            float: L1 metric
        """
        ovo_data = utils.ovo(self.data)

        error_dist = []
        for data in ovo_data:
            features = data.drop("class", axis=1)
            target = data["class"]

            clf = svm.SVC(kernel="linear")
            clf.fit(features, target)

            error_dist.append(
                sum(clf.predict(features) != target) / len(target)
            )

        return 1 - (1 / (1 + sum(error_dist)))

    def L2(self):
        raise NotImplementedError

    def L3(self):
        raise NotImplementedError

    # Neighborhood
    def N1(self):
        graph = nx.Graph(self.dst_matrix)
        mst = nx.minimum_spanning_tree(graph)
        edges = list(mst.edges())
        different_class = sum(
            self.data.iloc[u]["class"] != self.data.iloc[v]["class"]
            for u, v in edges
        )
        return different_class / self.data.shape[0]

    def intra(self, i):
        same_class = self.data[
            self.data["class"] == self.data.iloc[i]["class"]
        ].index
        return np.min(self.dst_matrix[i, list(set(same_class) - {i})])

    def inter(self, i):
        diff_class = self.data[
            self.data["class"] != self.data.iloc[i]["class"]
        ].index
        return np.min(self.dst_matrix[i, diff_class])

    def N2(self):
        intra_distances = np.array(
            [self.intra(i) for i in range(self.data.shape[0])]
        )
        inter_distances = np.array(
            [self.inter(i) for i in range(self.data.shape[0])]
        )
        return 1 - (1 / ((intra_distances / inter_distances) + 1))

    def knn(self, k):
        indices = np.argsort(self.dst_matrix, axis=1)[
            :, 1 : k + 1
        ]  # exclude self
        return np.array(
            [self.data.iloc[idx]["class"].mode()[0] for idx in indices]
        )

    def N3(self):
        knn_classes = self.knn(2)
        return np.mean(knn_classes != self.data["class"])

    def c_generate(self, n):
        new_data = []
        for _ in range(n):
            sample = self.data.sample(n=1)
            new_instance = sample.iloc[0].copy()
            new_instance["class"] = np.random.choice(
                self.data["class"].unique()
            )
            new_data.append(new_instance)
        return pd.DataFrame(new_data)

    def N4(self):
        generated_data = self.c_generate(self.data.shape[0])
        combined_data = pd.concat(
            [self.data, generated_data], ignore_index=True
        )

        combined_dst = pdist(combined_data.drop("class", axis=1))
        combined_dst_matrix = squareform(combined_dst)

        test_dst = combined_dst_matrix[
            self.data.shape[0] :, : self.data.shape[0]
        ]

        knn_classes = np.array(
            [self.data.iloc[np.argmin(dist)]["class"] for dist in test_dst]
        )
        return np.mean(knn_classes != generated_data["class"])

    def radios(self, i):
        di = self.inter(i)
        j = np.argmin(
            self.dst_matrix[
                i,
                self.data[
                    self.data["class"] != self.data.iloc[i]["class"]
                ].index,
            ]
        )

        _ = self.inter(j)
        k = np.argmin(
            self.dst_matrix[
                j,
                self.data[
                    self.data["class"] != self.data.iloc[j]["class"]
                ].index,
            ]
        )

        if i == k:
            return di / 2
        else:
            return di - self.radios(j)

    def hypersphere(self):
        return np.array([self.radios(i) for i in range(self.data.shape[0])])

    def translate(self, r):
        return self.dst_matrix < r[:, np.newaxis]

    def N5(self):
        r = self.hypersphere()
        adh = self.translate(r)
        h, _ = utils.adherence(adh)
        return len(h) / self.data.shape[0]

    def N6(self):
        r = np.array([self.inter(i) for i in range(self.data.shape[0])])
        adh = self.translate(r)
        return 1 - np.sum(adh) / (self.data.shape[0] ** 2)

    def T1(self):
        raise NotImplementedError

    def LSC(self):
        raise NotImplementedError

    # Balance

    def B1(self) -> float:
        """Class balance.

        Value of the entropy associated with the label.

        Returns:
            float: B1 metric.
        """
        c = -1 / np.log(self.y.nunique())
        i = self.y.value_counts(normalize=True)
        return 1 + c * entropy(i)

    def B2(self):
        ii = self.y.value_counts()
        nc = len(ii)
        aux = ((nc - 1) / nc) * np.sum(ii / (len(self.y) - ii))
        return 1 - (1 / aux)

    # Dimension
    def pca_variance(self):
        """PCA variance aggregator.

        It gets the number of
        components to those representing 95% of
        the variance.
        """
        pca = PCA()
        pca.fit(self.x)
        cumsum = np.cumsum(pca.explained_variance_ratio_)

        # Find number of components needed for 95% variance
        n_components = np.argmax(cumsum >= 0.95) + 1  # noqa: PLR2004
        return n_components

    def T2(self):
        """Ratio of number of features to number of instances."""
        return float(self.x.shape[1]) / float(self.x.shape[0])

    def T3(self):
        """Ratio of PCA components to number of instances."""
        return self.pca_variance() / float(self.x.shape[0])

    def T4(self):
        """Ratio of PCA components to number of features."""
        return self.pca_variance() / self.x.shape[1]
