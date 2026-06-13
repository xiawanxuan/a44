#include "io/SparseIO.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace fem {
namespace io {

SparseIO::SparseIO() = default;
SparseIO::~SparseIO() = default;

bool SparseIO::writeCSRBinary(const sparse::CSRMatrix& matrix,
                              const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t magic = MAGIC_NUMBER;
    uint32_t version = VERSION;
    uint32_t rows = static_cast<uint32_t>(matrix.rows);
    uint32_t cols = static_cast<uint32_t>(matrix.cols);
    uint32_t nnz = static_cast<uint32_t>(matrix.nnz);
    uint32_t isDouble = 1;

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    file.write(reinterpret_cast<const char*>(&isDouble), sizeof(isDouble));

    uint64_t rowPtrSize = matrix.rowPointers.size();
    file.write(reinterpret_cast<const char*>(&rowPtrSize), sizeof(rowPtrSize));
    file.write(reinterpret_cast<const char*>(matrix.rowPointers.data()),
               rowPtrSize * sizeof(Index));

    uint64_t colSize = matrix.colIndices.size();
    file.write(reinterpret_cast<const char*>(&colSize), sizeof(colSize));
    file.write(reinterpret_cast<const char*>(matrix.colIndices.data()),
               colSize * sizeof(Index));

    uint64_t valSize = matrix.values.size();
    file.write(reinterpret_cast<const char*>(&valSize), sizeof(valSize));
    file.write(reinterpret_cast<const char*>(matrix.values.data()),
               valSize * sizeof(Scalar));

    return true;
}

sparse::CSRMatrix SparseIO::readCSRBinary(const std::string& filename) const {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint32_t magic, version, rows, cols, nnz, isDouble;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC_NUMBER) {
        throw std::runtime_error("Invalid magic number in sparse matrix file");
    }

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != VERSION) {
        throw std::runtime_error("Unsupported file version");
    }

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    file.read(reinterpret_cast<char*>(&isDouble), sizeof(isDouble));

    sparse::CSRMatrix matrix;
    matrix.rows = static_cast<Index>(rows);
    matrix.cols = static_cast<Index>(cols);
    matrix.nnz = static_cast<Index>(nnz);

    uint64_t rowPtrSize;
    file.read(reinterpret_cast<char*>(&rowPtrSize), sizeof(rowPtrSize));
    matrix.rowPointers.resize(rowPtrSize);
    file.read(reinterpret_cast<char*>(matrix.rowPointers.data()),
              rowPtrSize * sizeof(Index));

    uint64_t colSize;
    file.read(reinterpret_cast<char*>(&colSize), sizeof(colSize));
    matrix.colIndices.resize(colSize);
    file.read(reinterpret_cast<char*>(matrix.colIndices.data()),
              colSize * sizeof(Index));

    uint64_t valSize;
    file.read(reinterpret_cast<char*>(&valSize), sizeof(valSize));
    matrix.values.resize(valSize);
    file.read(reinterpret_cast<char*>(matrix.values.data()),
              valSize * sizeof(Scalar));

    return matrix;
}

bool SparseIO::writeCOOASCII(const sparse::COOMatrix& matrix,
                             const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# COO Sparse Matrix\n";
    file << "# Rows: " << matrix.rows << "\n";
    file << "# Cols: " << matrix.cols << "\n";
    file << "# NNZ: " << matrix.nnz << "\n";

    for (Index k = 0; k < matrix.nnz; ++k) {
        file << matrix.rowIndices[k] << " "
             << matrix.colIndices[k] << " "
             << matrix.values[k] << "\n";
    }

    return true;
}

sparse::COOMatrix SparseIO::readCOOASCII(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    sparse::COOMatrix matrix;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        Index row, col;
        Scalar value;

        if (iss >> row >> col >> value) {
            matrix.addEntry(row, col, value);
        }
    }

    return matrix;
}

bool SparseIO::writeMatrixMarket(const SparseMatrix& matrix,
                                 const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "%\n";
    file << matrix.rows() << " " << matrix.cols() << " "
         << matrix.nonZeros() << "\n";

    for (Index k = 0; k < matrix.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
            file << it.row() + 1 << " " << it.col() + 1 << " "
                 << it.value() << "\n";
        }
    }

    return true;
}

bool SparseIO::writeDenseMatrixASCII(const DenseMatrix& matrix,
                                     const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << matrix.rows() << " " << matrix.cols() << "\n";
    for (Index i = 0; i < matrix.rows(); ++i) {
        for (Index j = 0; j < matrix.cols(); ++j) {
            if (j > 0) file << " ";
            file << matrix(i, j);
        }
        file << "\n";
    }

    return true;
}

DenseMatrix SparseIO::readDenseMatrixASCII(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    Index rows, cols;
    file >> rows >> cols;

    DenseMatrix matrix(rows, cols);
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }

    return matrix;
}

} // namespace io
} // namespace fem
