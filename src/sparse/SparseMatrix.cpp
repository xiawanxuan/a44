#include "sparse/SparseMatrix.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>

namespace fem {
namespace sparse {

COOMatrix::COOMatrix() : rows(0), cols(0), nnz(0) {}

void COOMatrix::clear() {
    rows = 0;
    cols = 0;
    nnz = 0;
    rowIndices.clear();
    colIndices.clear();
    values.clear();
}

void COOMatrix::reserve(Index count) {
    rowIndices.reserve(count);
    colIndices.reserve(count);
    values.reserve(count);
}

void COOMatrix::addEntry(Index row, Index col, Scalar value) {
    rowIndices.push_back(row);
    colIndices.push_back(col);
    values.push_back(value);
    nnz++;
    if (row >= rows) rows = row + 1;
    if (col >= cols) cols = col + 1;
}

Index COOMatrix::nonZeros() const {
    return nnz;
}

CSRMatrix::CSRMatrix() : rows(0), cols(0), nnz(0) {}

void CSRMatrix::clear() {
    rows = 0;
    cols = 0;
    nnz = 0;
    rowPointers.clear();
    colIndices.clear();
    values.clear();
}

Index CSRMatrix::nonZeros() const {
    return nnz;
}

Scalar CSRMatrix::memoryUsageMB() const {
    size_t bytes = rowPointers.size() * sizeof(Index) +
                   colIndices.size() * sizeof(Index) +
                   values.size() * sizeof(Scalar);
    return static_cast<Scalar>(bytes) / (1024.0 * 1024.0);
}

double CSRMatrix::sparsityRatio() const {
    if (rows == 0 || cols == 0) return 0.0;
    double total = static_cast<double>(rows) * static_cast<double>(cols);
    return 1.0 - static_cast<double>(nnz) / total;
}

SparseConverter::SparseConverter() = default;
SparseConverter::~SparseConverter() = default;

CSRMatrix SparseConverter::denseToCSR(const DenseMatrix& dense, Scalar tolerance) const {
    COOMatrix coo = denseToCOO(dense, tolerance);
    return cooToCSR(coo);
}

COOMatrix SparseConverter::denseToCOO(const DenseMatrix& dense, Scalar tolerance) const {
    COOMatrix coo;
    coo.rows = dense.rows();
    coo.cols = dense.cols();
    coo.nnz = 0;

    Index estimatedNNZ = dense.rows() * dense.cols() / 10;
    coo.reserve(std::max(estimatedNNZ, Index(100)));

    for (Index j = 0; j < dense.cols(); ++j) {
        for (Index i = 0; i < dense.rows(); ++i) {
            if (std::abs(dense(i, j)) > tolerance) {
                coo.addEntry(i, j, dense(i, j));
            }
        }
    }

    return coo;
}

CSRMatrix SparseConverter::cooToCSR(const COOMatrix& coo) const {
    CSRMatrix csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;

    if (coo.nnz == 0) {
        csr.rowPointers.resize(csr.rows + 1, 0);
        return csr;
    }

    csr.rowPointers.resize(csr.rows + 1, 0);
    csr.colIndices.resize(coo.nnz);
    csr.values.resize(coo.nnz);

    for (Index k = 0; k < coo.nnz; ++k) {
        Index row = coo.rowIndices[k];
        if (row < csr.rows) {
            csr.rowPointers[row + 1]++;
        }
    }

    for (Index i = 0; i < csr.rows; ++i) {
        csr.rowPointers[i + 1] += csr.rowPointers[i];
    }

    std::vector<Index> rowCounts(csr.rows + 1, 0);
    for (Index k = 0; k < coo.nnz; ++k) {
        Index row = coo.rowIndices[k];
        if (row >= csr.rows) continue;

        Index pos = csr.rowPointers[row] + rowCounts[row];
        csr.colIndices[pos] = coo.colIndices[k];
        csr.values[pos] = coo.values[k];
        rowCounts[row]++;
    }

    for (Index i = 0; i < csr.rows; ++i) {
        Index start = csr.rowPointers[i];
        Index end = csr.rowPointers[i + 1];

        std::vector<Index> indices(end - start);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                  [&](Index a, Index b) {
                      return csr.colIndices[start + a] < csr.colIndices[start + b];
                  });

        std::vector<Index> tempCols(end - start);
        std::vector<Scalar> tempVals(end - start);
        for (Index k = 0; k < end - start; ++k) {
            tempCols[k] = csr.colIndices[start + indices[k]];
            tempVals[k] = csr.values[start + indices[k]];
        }

        for (Index k = 0; k < end - start; ++k) {
            csr.colIndices[start + k] = tempCols[k];
            csr.values[start + k] = tempVals[k];
        }
    }

    return csr;
}

COOMatrix SparseConverter::csrToCOO(const CSRMatrix& csr) const {
    COOMatrix coo;
    coo.rows = csr.rows;
    coo.cols = csr.cols;
    coo.nnz = csr.nnz;

    coo.rowIndices.resize(csr.nnz);
    coo.colIndices.resize(csr.nnz);
    coo.values.resize(csr.nnz);

    Index idx = 0;
    for (Index i = 0; i < csr.rows; ++i) {
        for (Index j = csr.rowPointers[i]; j < csr.rowPointers[i + 1]; ++j) {
            coo.rowIndices[idx] = i;
            coo.colIndices[idx] = csr.colIndices[j];
            coo.values[idx] = csr.values[j];
            idx++;
        }
    }

    return coo;
}

DenseMatrix SparseConverter::csrToDense(const CSRMatrix& csr) const {
    DenseMatrix dense = DenseMatrix::Zero(csr.rows, csr.cols);

    for (Index i = 0; i < csr.rows; ++i) {
        for (Index j = csr.rowPointers[i]; j < csr.rowPointers[i + 1]; ++j) {
            dense(i, csr.colIndices[j]) = csr.values[j];
        }
    }

    return dense;
}

DenseMatrix SparseConverter::cooToDense(const COOMatrix& coo) const {
    DenseMatrix dense = DenseMatrix::Zero(coo.rows, coo.cols);

    for (Index k = 0; k < coo.nnz; ++k) {
        dense(coo.rowIndices[k], coo.colIndices[k]) += coo.values[k];
    }

    return dense;
}

SparseMatrix SparseConverter::eigenSparseFromCSR(const CSRMatrix& csr) const {
    std::vector<Triplet> triplets;
    triplets.reserve(csr.nnz);

    for (Index i = 0; i < csr.rows; ++i) {
        for (Index j = csr.rowPointers[i]; j < csr.rowPointers[i + 1]; ++j) {
            triplets.emplace_back(i, csr.colIndices[j], csr.values[j]);
        }
    }

    SparseMatrix mat(csr.rows, csr.cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();

    return mat;
}

CSRMatrix SparseConverter::eigenSparseToCSR(const SparseMatrix& eigenSparse) const {
    CSRMatrix csr;
    csr.rows = eigenSparse.rows();
    csr.cols = eigenSparse.cols();
    csr.nnz = eigenSparse.nonZeros();

    if (eigenSparse.IsRowMajor) {
        csr.rowPointers.resize(csr.rows + 1);
        csr.colIndices.resize(csr.nnz);
        csr.values.resize(csr.nnz);

        const int* outerPtr = eigenSparse.outerIndexPtr();
        const int* innerPtr = eigenSparse.innerIndexPtr();
        const Scalar* valPtr = eigenSparse.valuePtr();

        for (Index i = 0; i <= csr.rows; ++i) {
            csr.rowPointers[i] = outerPtr[i];
        }
        for (Index k = 0; k < csr.nnz; ++k) {
            csr.colIndices[k] = innerPtr[k];
            csr.values[k] = valPtr[k];
        }
    } else {
        COOMatrix coo;
        coo.rows = csr.rows;
        coo.cols = csr.cols;
        coo.nnz = csr.nnz;
        coo.reserve(csr.nnz);

        for (Index k = 0; k < eigenSparse.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(eigenSparse, k); it; ++it) {
                coo.addEntry(it.row(), it.col(), it.value());
            }
        }

        csr = cooToCSR(coo);
    }

    return csr;
}

void SparseConverter::sortCOO(COOMatrix& coo) const {
    std::vector<Index> indices(coo.nnz);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](Index a, Index b) {
                  if (coo.rowIndices[a] != coo.rowIndices[b]) {
                      return coo.rowIndices[a] < coo.rowIndices[b];
                  }
                  return coo.colIndices[a] < coo.colIndices[b];
              });

    COOMatrix sorted;
    sorted.rows = coo.rows;
    sorted.cols = coo.cols;
    sorted.nnz = coo.nnz;
    sorted.reserve(coo.nnz);

    for (Index k = 0; k < coo.nnz; ++k) {
        sorted.rowIndices.push_back(coo.rowIndices[indices[k]]);
        sorted.colIndices.push_back(coo.colIndices[indices[k]]);
        sorted.values.push_back(coo.values[indices[k]]);
    }

    coo = sorted;
}

void SparseConverter::removeDuplicatesCOO(COOMatrix& coo, Scalar tolerance) const {
    if (coo.nnz == 0) return;

    sortCOO(coo);

    COOMatrix result;
    result.rows = coo.rows;
    result.cols = coo.cols;
    result.reserve(coo.nnz);

    Index currRow = coo.rowIndices[0];
    Index currCol = coo.colIndices[0];
    Scalar currVal = coo.values[0];

    for (Index k = 1; k < coo.nnz; ++k) {
        if (coo.rowIndices[k] == currRow && coo.colIndices[k] == currCol) {
            currVal += coo.values[k];
        } else {
            if (std::abs(currVal) > tolerance) {
                result.addEntry(currRow, currCol, currVal);
            }
            currRow = coo.rowIndices[k];
            currCol = coo.colIndices[k];
            currVal = coo.values[k];
        }
    }

    if (std::abs(currVal) > tolerance) {
        result.addEntry(currRow, currCol, currVal);
    }

    coo = result;
}

bool SparseConverter::compareCOOEntries(Index i, Index j,
                                        const std::vector<Index>& rows,
                                        const std::vector<Index>& cols) {
    if (rows[i] != rows[j]) {
        return rows[i] < rows[j];
    }
    return cols[i] < cols[j];
}

} // namespace sparse
} // namespace fem
