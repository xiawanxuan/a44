#include "sparse/SparseMatrix.h"
#include <cmath>

using namespace fem;
using namespace fem::sparse;

TEST_CASE(dense_to_coo_basic) {
    DenseMatrix dense = DenseMatrix::Zero(4, 4);
    dense(0, 0) = 1.0;
    dense(1, 2) = 2.0;
    dense(3, 1) = 3.0;

    SparseConverter converter;
    COOMatrix coo = converter.denseToCOO(dense);

    ASSERT_EQ(coo.rows, 4);
    ASSERT_EQ(coo.cols, 4);
    ASSERT_EQ(coo.nnz, 3);
}

TEST_CASE(dense_to_csr_basic) {
    DenseMatrix dense = DenseMatrix::Zero(3, 3);
    dense(0, 0) = 5.0;
    dense(0, 2) = 6.0;
    dense(1, 1) = 7.0;
    dense(2, 0) = 8.0;

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    ASSERT_EQ(csr.rows, 3);
    ASSERT_EQ(csr.cols, 3);
    ASSERT_EQ(csr.nnz, 4);
    ASSERT_EQ(csr.rowPointers.size(), 4);
}

TEST_CASE(coo_to_csr_roundtrip) {
    DenseMatrix dense = DenseMatrix::Zero(5, 5);
    dense(0, 0) = 1.0;
    dense(0, 3) = 2.0;
    dense(2, 1) = 3.0;
    dense(2, 4) = 4.0;
    dense(4, 2) = 5.0;
    dense(3, 3) = 6.0;

    SparseConverter converter;
    COOMatrix coo = converter.denseToCOO(dense);
    CSRMatrix csr = converter.cooToCSR(coo);
    DenseMatrix reconstructed = converter.csrToDense(csr);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_NEAR(dense(i, j), reconstructed(i, j), 1e-15);
        }
    }
}

TEST_CASE(csr_to_coo_roundtrip) {
    DenseMatrix dense = DenseMatrix::Zero(4, 4);
    dense(0, 1) = 10.0;
    dense(1, 2) = 20.0;
    dense(2, 3) = 30.0;
    dense(3, 0) = 40.0;

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);
    COOMatrix coo = converter.csrToCOO(csr);
    DenseMatrix reconstructed = converter.cooToDense(coo);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_NEAR(dense(i, j), reconstructed(i, j), 1e-15);
        }
    }
}

TEST_CASE(sparse_matrix_memory_calculation) {
    DenseMatrix dense = DenseMatrix::Zero(100, 100);
    for (int i = 0; i < 100; ++i) {
        dense(i, i) = 1.0;
    }

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    ASSERT_GT(csr.memoryUsageMB(), 0.0);
    ASSERT_LT(csr.memoryUsageMB(), 1.0);
}

TEST_CASE(sparsity_ratio_diagonal_matrix) {
    DenseMatrix dense = DenseMatrix::Zero(100, 100);
    for (int i = 0; i < 100; ++i) {
        dense(i, i) = 1.0;
    }

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    double sparsity = csr.sparsityRatio();
    ASSERT_GT(sparsity, 0.9);
    ASSERT_LT(sparsity, 1.0);
}

TEST_CASE(sparsity_ratio_dense_matrix) {
    DenseMatrix dense = DenseMatrix::Ones(10, 10);

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    double sparsity = csr.sparsityRatio();
    ASSERT_NEAR(sparsity, 0.0, 1e-10);
}

TEST_CASE(zero_matrix_conversion) {
    DenseMatrix dense = DenseMatrix::Zero(5, 5);

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    ASSERT_EQ(csr.nnz, 0);
    ASSERT_EQ(csr.rowPointers.size(), 6);
    for (const auto& ptr : csr.rowPointers) {
        ASSERT_EQ(ptr, 0);
    }
}

TEST_CASE(coo_sort_and_remove_duplicates) {
    COOMatrix coo;
    coo.rows = 3;
    coo.cols = 3;
    coo.addEntry(2, 1, 2.0);
    coo.addEntry(0, 0, 1.0);
    coo.addEntry(1, 2, 3.0);
    coo.addEntry(0, 0, 1.0);

    SparseConverter converter;
    converter.removeDuplicatesCOO(coo);

    ASSERT_EQ(coo.nnz, 3);
}

TEST_CASE(eigen_sparse_conversion) {
    DenseMatrix dense = DenseMatrix::Zero(5, 5);
    dense(0, 0) = 1.0;
    dense(1, 2) = 2.5;
    dense(3, 1) = -1.0;
    dense(4, 4) = 3.0;

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);
    SparseMatrix eigenSparse = converter.eigenSparseFromCSR(csr);
    CSRMatrix back = converter.eigenSparseToCSR(eigenSparse);

    ASSERT_EQ(back.nnz, csr.nnz);

    DenseMatrix reconstructed = converter.csrToDense(back);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ASSERT_NEAR(dense(i, j), reconstructed(i, j), 1e-15);
        }
    }
}
