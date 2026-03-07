from enum import Enum
import numpy as np

class DistanceCalculator:

    class Method(Enum):
        ThreeDBroadcasting = 1
        CrossProduct = 2

    @staticmethod 
    def calc_distance_sq(A: np.ndarray, B: np.ndarray, method=Method.ThreeDBroadcasting) -> np.ndarray:
        if method == DistanceCalculator.Method.ThreeDBroadcasting:
            return DistanceCalculator.calc_distance_sq_3d_broadcasting(A, B)

        if method == DistanceCalculator.Method.CrossProduct:
            return DistanceCalculator.calc_distance_sq_cross_prod(A, B)
        
        raise RuntimeError("Unrecognized Method passed")

    @staticmethod
    def calc_distance_sq_3d_broadcasting(x: np.ndarray, other: np.ndarray) -> np.ndarray:
        distances = np.sum((x[:, None, :] - other[None, :, :]) **2, axis=2)
        return distances
    
    @staticmethod
    def calc_distance_sq_cross_prod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        ''' doesn't rely on 3d-broadcasting '''
        cross_prod = -2.0 * A @ B.T
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        distances = cross_prod + A_sq + B_sq.T
        return distances


class CustomVectorizedKNeighborsClassifier:
    ''' - Brute force approach;
    - uses partition (np.argpartition) to find k nearest neighbours, does not sort them in order, 
    which makes it more efficient for a single value of k'''

    def __init__(self, n_neighbors: int) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.y_max = y_train.max()
        self.y_min = y_train.min()

    @staticmethod
    def calc_distance(x: np.ndarray, other: np.ndarray, method) -> np.ndarray:
        return DistanceCalculator.calc_distance_sq(x, other, method)

    def predict(self, x: np.ndarray, method=DistanceCalculator.Method.CrossProduct) -> np.ndarray:
        distances = self.calc_distance(x, self.x_train, method)
        
        min_distances_args = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        classes_of_nearest_neighbours = self.y_train[min_distances_args]

        counts = np.apply_along_axis(
            lambda x: np.bincount(x,  minlength=(self.y_max-self.y_min + 1)),
            axis = 1,
            arr = classes_of_nearest_neighbours
        )

        pred = self.y_min + np.argmax(counts, axis=1)

        return pred

    
class CustomVectorizedSortedKNeighborsClassifier:
    ''' - Brute force approach;
    - sorts neighbours by distance, which can be reused for multiple values of K '''

    def __init__(self, n_neighbors: int) -> None:
        self.n_neighbours = n_neighbors

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.y_max = y_train.max()
        self.y_min = y_train.min()

    @staticmethod
    def calc_distance(x: np.ndarray, other: np.ndarray, method) -> np.ndarray:
        return DistanceCalculator.calc_distance_sq(x, other, method)

    def predict(self, x: np.ndarray, method=DistanceCalculator.Method.CrossProduct) -> np.ndarray:
        distances = self.calc_distance(x, self.x_train, method)
        
        min_distances_args = np.argsort(distances, axis=1)[:, :self.n_neighbours]
        classes_of_nearest_neighbours = self.y_train[min_distances_args]

        counts = np.apply_along_axis(
            lambda x: np.bincount(x,  minlength=(self.y_max-self.y_min + 1)),
            axis = 1,
            arr = classes_of_nearest_neighbours
        )

        pred = self.y_min + np.argmax(counts, axis=1)

        return pred


class Utils:
    @staticmethod
    def calc_accuracy_score(y_test: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
        if y_test.shape != y_pred.shape:
            raise ValueError("Shapes of y_test and y_pred must match.")
        accuracy = (y_pred == y_test).sum()
        return accuracy / y_pred.shape[0] if normalize else accuracy

    @staticmethod
    def calc_accuracy_score_2(y_test: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
        if y_test.shape != y_pred.shape:
            raise ValueError("Shapes of y_test and y_pred must match.")
        if normalize:
            return np.mean(y_test == y_pred)
        return np.sum(y_test == y_pred)