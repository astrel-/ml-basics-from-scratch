# k-Nearest Neighbours (kNN)

This folder contains a **from-scratch implementation of k-Nearest Neighbours** used to explore algorithmic and performance trade-offs between different implementations.

The goal of this exercise is not to build a production classifier, but to understand:

* algorithmic complexity of kNN
* vectorization techniques
* top-k selection strategies
* performance differences between Python and C++

---

# Benchmark example

Example timings (microseconds per run):

| Algorithm | Language | k = 3 | k = 20 |
|---|---|---|---|
| Naive | C++ | 289 µs | 384 µs |
| VectorizedHeap | C++ | **97 µs** | 197 µs |
| VectorizedSort | C++ | 653 µs | 640 µs |
| VectorizedPartition | C++ | 119 µs | **146 µs** |
| scikit-learn | Python | 594 µs | 677 µs |
| VectorizedSort | Python | 408 µs | **399 µs** |
| VectorizedPartition | Python | **403 µs** | 432 µs |

Notes:
* All C++ benchmarks were compiled in **Release mode** using OpenBLAS.
* Python benchmarks were run using NumPy / scikit-learn implementations.
* Number of features: 2
* Train Sample: 227
* Test Sample: 98

Observations:

* `VectorizedSort` is consistently the slowest because it performs unnecessary sorting.
* `VectorizedHeap` performs well for very small `k`.
* `VectorizedPartition` tends to perform best for moderate `k`.

---


# Folder structure

```
methods/kNN/
    cpp/      C++ implementations and benchmarks
    python/   Python implementations (scikit-learn, custom)
    data/     data preparation + datasets exported as .npz
```

### `cpp/`

Contains a C++ implementation of kNN with several algorithmic variants.

The implementation focuses on:

* explicit memory layout
* vectorized distance computation
* different top-k selection algorithms

Dependencies:

* **OpenBLAS** (via `vcpkg`)
* **cnpy** for reading `.npz` files

---

### `python/`

Contains the reference implementation and a notebook used to:

The Python version is primarily used for correctness verification.

---

### `data/`

Training and test datasets exported from the Python notebook.

Expected files:

```
train_data.npz
test_data.npz
```

Each file contains:

```
X  → feature matrix
y  → labels
```

Contains notebook used to produce train and test sets from raw data:

* load the dataset
* split train/test data
* export datasets to `.npz` format for the C++ code

---

# C++ Implementations

Four kNN implementations are provided in the C++ code.

### Naive

Direct nested loops:

```
for each test sample
    create max-heap with maxSize of k to track distances
    for each train sample
        compute distance to each training sample
        push distance to the maxHeap to keep track of k closest neighbours 
```

#### Pros
- Low memory usage: only `k` elements are stored per row
- Works well for very small `k`
- Does not require storing the full distance matrix
#### Cons
- Lower performance due to lack of vectorization

---

### VectorizedHeap

Pairwise Distances are computed in a vectorized manner and the **k smallest elements** are maintained using a max-heap.

Efficient when **k is small**.

---

### VectorizedSort

Pairwise Distances are computed vectorized, then distances from test point to each train point are sorted.

This approach is simple but inefficient because it does unnecessary sorting of all the distances.

---

### VectorizedPartition

Distances are computed vectorized and `std::nth_element` is used to select the top-k neighbours.

This method avoids full sorting and is typically the fastest for moderate values of `k`.

---

# Vectorized distance computation

Distances are computed using the identity:

```
||a - b||² = ||a||² + ||b||² − 2 a·b^T
```

This allows the expensive part to be expressed as a **matrix multiplication**:

```
-2 * X_test * X_train^T
```

which is computed using **BLAS**.

The squared norms of the vectors are then added to rows and columns of the resulting matrix.

---

# Building the C++ implementation

The project uses **CMake**.

Example build:

```
cmake -B build
cmake --build build --config Release
```

For performance measurements the project **must be built in Release mode**.

---

# Python Implementations


### Scikit-learn implementation

Benchmarks also include the reference implementation from **scikit-learn**:

```python
sklearn.neighbors.KNeighborsClassifier
```

### VectorizedPartition

This implementation computes the full distance matrix and then uses **`np.argpartition`** to find the `k` nearest neighbours.

`np.argpartition` performs a partial ordering and returns the indices of the `k` smallest elements without sorting them fully.

**Pros**
- Faster for a single value of `k`
- Avoids the cost of full sorting
- Complexity roughly `O(n)` for neighbour selection

**Cons**
- Neighbours are **not sorted**
- Cannot reuse the ordering if predictions for multiple values of `k` are needed

### VectorizedSort

This implementation computes the full distance matrix and then uses **`np.argsort`** to fully sort neighbours by distance.

The first `k` neighbours are then selected from the sorted list.

**Pros**
- Neighbours are fully sorted by distance
- Can reuse the sorted order for **multiple values of `k`**

**Cons**
- More expensive than partition-based selection
- Complexity `O(n log n)` per row

---

# Purpose of this experiment

This implementation explores several practical lessons:

* separating **algorithmic complexity** from **constant factors**
* benefits of **vectorized linear algebra**
* trade-offs between **heap**, **partition**, and **sorting** for top-k selection
* the importance of measuring performance rather than assuming theoretical results.

---

This project is part of the broader **ML-from-scratch experiments** in this repository.
