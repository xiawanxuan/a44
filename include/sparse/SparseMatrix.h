#ifndef FEM_SPARSE_SPARSE_MATRIX_H
#define FEM_SPARSE_SPARSE_MATRIX_H

#include "common/Types.h"
#include <vector>
#include <cstddef>

namespace fem {
namespace sparse {

struct COOMatrix {
    Index rows;
    Index cols;
    std::vector<Index> rowIndices;
    std::vector<Index> colIndices;
    std::vector<Scalar> values;
    Index nnz;

    COOMatrix();
    void clear();
    void reserve(Index count);
    void addEntry(Index row, Index col, Scalar value);
    Index nonZeros() const;
};

struct CSRMatrix {
    Index rows;
    Index cols;
    std::vector<Index> rowPointers;
    std::vector<Index> colIndices;
    std::vector<Scalar> values;
    Index nnz;

    CSRMatrix();
    void clear();
    Index nonZeros() const;
    Scalar memoryUsageMB() const;
    double sparsityRatio() const;
};

class SparseConverter {
public:
    SparseConverter();
    ~SparseConverter();

    CSRMatrix denseToCSR(const DenseMatrix& dense, Scalar tolerance = 1e-15) const;
    COOMatrix denseToCOO(const DenseMatrix& dense, Scalar tolerance = 1e-15) const;

    CSRMatrix cooToCSR(const COOMatrix& coo) const;
    COOMatrix csrToCOO(const CSRMatrix& csr) const;

    DenseMatrix csrToDense(const CSRMatrix& csr) const;
    DenseMatrix cooToDense(const COOMatrix& coo) const;

    SparseMatrix eigenSparseFromCSR(const CSRMatrix& csr) const;
    CSRMatrix eigenSparseToCSR(const SparseMatrix& eigenSparse) const;

    void sortCOO(COOMatrix& coo) const;
    void removeDuplicatesCOO(COOMatrix& coo, Scalar tolerance = 1e-15) const;

private:
    static bool compareCOOEntries(Index i, Index j,
                                  const std::vector<Index>& rows,
                                  const std::vector<Index>& cols);
};

} // namespace sparse
} // namespace fem

#endif // FEM_SPARSE_SPARSE_MATRIX_H
