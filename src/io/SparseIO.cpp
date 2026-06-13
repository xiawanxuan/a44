#include "io/SparseIO.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fem {
namespace io {

SparseIO::SparseIO() = default;
SparseIO::~SparseIO() = default;

bool SparseIO::writeCSRBinary(const sparse::CSRMatrix& matrix,
                              const std::string& filename) const {
    if (matrix.rows < 0 || matrix.cols < 0 || matrix.nnz < 0) {
        std::cerr << "SparseIO: Invalid matrix dimensions (negative)" << std::endl;
        return false;
    }

    if (!matrix.rowPointers.empty() && matrix.rowPointers.size() != static_cast<size_t>(matrix.rows) + 1) {
        std::cerr << "SparseIO: rowPointers size mismatch, expected "
                  << (matrix.rows + 1) << ", got " << matrix.rowPointers.size() << std::endl;
        return false;
    }

    if (matrix.colIndices.size() != static_cast<size_t>(matrix.nnz)) {
        std::cerr << "SparseIO: colIndices size mismatch, expected "
                  << matrix.nnz << ", got " << matrix.colIndices.size() << std::endl;
        return false;
    }
    if (matrix.values.size() != static_cast<size_t>(matrix.nnz)) {
        std::cerr << "SparseIO: values size mismatch, expected "
                  << matrix.nnz << ", got " << matrix.values.size() << std::endl;
        return false;
    }

    if (matrix.nnz > 0 && !matrix.rowPointers.empty()) {
        if (matrix.rowPointers[0] != 0) {
            std::cerr << "SparseIO: rowPointers[0] must be 0, got "
                      << matrix.rowPointers[0] << std::endl;
            return false;
        }
        if (matrix.rowPointers[matrix.rows] != matrix.nnz) {
            std::cerr << "SparseIO: rowPointers[rows] != nnz, expected "
                      << matrix.nnz << ", got " << matrix.rowPointers[matrix.rows] << std::endl;
            return false;
        }
        for (Index i = 0; i < matrix.rows; ++i) {
            if (matrix.rowPointers[i + 1] < matrix.rowPointers[i]) {
                std::cerr << "SparseIO: rowPointers not monotonic at row " << i << std::endl;
                return false;
            }
        }
    }

    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "SparseIO: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    const uint32_t magic = MAGIC_NUMBER;
    const uint32_t version = VERSION;
    const uint64_t rows = static_cast<uint64_t>(matrix.rows);
    const uint64_t cols = static_cast<uint64_t>(matrix.cols);
    const uint64_t nnz = static_cast<uint64_t>(matrix.nnz);
    const uint8_t  isDouble = (std::is_same<Scalar, double>::value) ? 1 : 0;
    const uint8_t  indexByteSize = static_cast<uint8_t>(sizeof(Index));
    const uint8_t  reserved1 = 0;
    const uint8_t  reserved2 = 0;

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    file.write(reinterpret_cast<const char*>(&isDouble), sizeof(isDouble));
    file.write(reinterpret_cast<const char*>(&indexByteSize), sizeof(indexByteSize));
    file.write(reinterpret_cast<const char*>(&reserved1), sizeof(reserved1));
    file.write(reinterpret_cast<const char*>(&reserved2), sizeof(reserved2));

    const uint64_t rowPtrSize = static_cast<uint64_t>(matrix.rowPointers.size());
    file.write(reinterpret_cast<const char*>(&rowPtrSize), sizeof(rowPtrSize));
    if (rowPtrSize > 0) {
        file.write(reinterpret_cast<const char*>(matrix.rowPointers.data()),
                   static_cast<std::streamsize>(rowPtrSize * sizeof(Index)));
    }

    const uint64_t colSize = static_cast<uint64_t>(matrix.colIndices.size());
    file.write(reinterpret_cast<const char*>(&colSize), sizeof(colSize));
    if (colSize > 0) {
        file.write(reinterpret_cast<const char*>(matrix.colIndices.data()),
                   static_cast<std::streamsize>(colSize * sizeof(Index)));
    }

    const uint64_t valSize = static_cast<uint64_t>(matrix.values.size());
    file.write(reinterpret_cast<const char*>(&valSize), sizeof(valSize));
    if (valSize > 0) {
        file.write(reinterpret_cast<const char*>(matrix.values.data()),
                   static_cast<std::streamsize>(valSize * sizeof(Scalar)));
    }

    const uint64_t checksumRowPtr = static_cast<uint64_t>(
        matrix.rowPointers.empty() ? 0 : matrix.rowPointers.back());
    const uint64_t checksumNNZ = nnz;
    file.write(reinterpret_cast<const char*>(&checksumRowPtr), sizeof(checksumRowPtr));
    file.write(reinterpret_cast<const char*>(&checksumNNZ), sizeof(checksumNNZ));

    file.flush();

    if (!file.good()) {
        std::cerr << "SparseIO: I/O error occurred during write" << std::endl;
        return false;
    }

    file.close();
    return true;
}

sparse::CSRMatrix SparseIO::readCSRBinary(const std::string& filename) const {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("SparseIO: Cannot open file for reading: " + filename);
    }

    uint32_t magic = 0;
    uint32_t version = 0;

    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!file.good() || magic != MAGIC_NUMBER) {
        throw std::runtime_error("SparseIO: Invalid magic number in sparse matrix file: " + filename);
    }

    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file.good()) {
        throw std::runtime_error("SparseIO: Failed to read version from: " + filename);
    }

    sparse::CSRMatrix matrix;

    if (version == VERSION_LEGACY) {
        uint32_t rowsU32 = 0, colsU32 = 0, nnzU32 = 0, isDoubleU32 = 0;
        file.read(reinterpret_cast<char*>(&rowsU32), sizeof(rowsU32));
        file.read(reinterpret_cast<char*>(&colsU32), sizeof(colsU32));
        file.read(reinterpret_cast<char*>(&nnzU32), sizeof(nnzU32));
        file.read(reinterpret_cast<char*>(&isDoubleU32), sizeof(isDoubleU32));
        if (!file.good()) {
            throw std::runtime_error("SparseIO: Failed to read legacy v1 header");
        }

        matrix.rows = static_cast<Index>(rowsU32);
        matrix.cols = static_cast<Index>(colsU32);
        matrix.nnz  = static_cast<Index>(nnzU32);

        uint64_t rowPtrSize = 0;
        file.read(reinterpret_cast<char*>(&rowPtrSize), sizeof(rowPtrSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read rowPtrSize (v1)");
        matrix.rowPointers.resize(static_cast<size_t>(rowPtrSize));
        if (rowPtrSize > 0) {
            file.read(reinterpret_cast<char*>(matrix.rowPointers.data()),
                      static_cast<std::streamsize>(rowPtrSize * sizeof(Index)));
        }

        uint64_t colSize = 0;
        file.read(reinterpret_cast<char*>(&colSize), sizeof(colSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read colSize (v1)");
        matrix.colIndices.resize(static_cast<size_t>(colSize));
        if (colSize > 0) {
            file.read(reinterpret_cast<char*>(matrix.colIndices.data()),
                      static_cast<std::streamsize>(colSize * sizeof(Index)));
        }

        uint64_t valSize = 0;
        file.read(reinterpret_cast<char*>(&valSize), sizeof(valSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read valSize (v1)");
        matrix.values.resize(static_cast<size_t>(valSize));
        if (valSize > 0) {
            file.read(reinterpret_cast<char*>(matrix.values.data()),
                      static_cast<std::streamsize>(valSize * sizeof(Scalar)));
        }

        if (!file.good()) {
            throw std::runtime_error("SparseIO: I/O error during v1 data read");
        }
    } else if (version == VERSION) {
        uint64_t rowsU64 = 0, colsU64 = 0, nnzU64 = 0;
        uint8_t  isDoubleU8 = 0, indexByteSize = 0, reserved1 = 0, reserved2 = 0;

        file.read(reinterpret_cast<char*>(&rowsU64), sizeof(rowsU64));
        file.read(reinterpret_cast<char*>(&colsU64), sizeof(colsU64));
        file.read(reinterpret_cast<char*>(&nnzU64),  sizeof(nnzU64));
        file.read(reinterpret_cast<char*>(&isDoubleU8),  sizeof(isDoubleU8));
        file.read(reinterpret_cast<char*>(&indexByteSize), sizeof(indexByteSize));
        file.read(reinterpret_cast<char*>(&reserved1), sizeof(reserved1));
        file.read(reinterpret_cast<char*>(&reserved2), sizeof(reserved2));
        if (!file.good()) {
            throw std::runtime_error("SparseIO: Failed to read v2 header fields");
        }

        if (indexByteSize != 4 && indexByteSize != 8) {
            throw std::runtime_error("SparseIO: Unsupported indexByteSize=" +
                                      std::to_string(static_cast<int>(indexByteSize)));
        }
        if (isDoubleU8 != 1) {
            throw std::runtime_error("SparseIO: Unsupported Scalar type (not double)");
        }

        matrix.rows = static_cast<Index>(rowsU64);
        matrix.cols = static_cast<Index>(colsU64);
        matrix.nnz  = static_cast<Index>(nnzU64);

        if (matrix.rows < 0 || matrix.cols < 0 || matrix.nnz < 0) {
            throw std::runtime_error("SparseIO: Corrupted file - negative dimensions after read");
        }

        uint64_t rowPtrSize = 0;
        file.read(reinterpret_cast<char*>(&rowPtrSize), sizeof(rowPtrSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read rowPtrSize (v2)");

        if (indexByteSize == sizeof(Index)) {
            matrix.rowPointers.resize(static_cast<size_t>(rowPtrSize));
            if (rowPtrSize > 0) {
                file.read(reinterpret_cast<char*>(matrix.rowPointers.data()),
                          static_cast<std::streamsize>(rowPtrSize * sizeof(Index)));
            }
        } else {
            std::vector<uint8_t> rawBuffer(
                static_cast<size_t>(rowPtrSize) * static_cast<size_t>(indexByteSize));
            if (rowPtrSize > 0) {
                file.read(reinterpret_cast<char*>(rawBuffer.data()),
                          static_cast<std::streamsize>(rawBuffer.size()));
            }
            matrix.rowPointers.resize(static_cast<size_t>(rowPtrSize));
            for (uint64_t i = 0; i < rowPtrSize; ++i) {
                uint64_t value = 0;
                if (indexByteSize == 4) {
                    uint32_t v32 = 0;
                    std::memcpy(&v32, rawBuffer.data() + i * 4, 4);
                    value = static_cast<uint64_t>(v32);
                } else {
                    std::memcpy(&value, rawBuffer.data() + i * 8, 8);
                }
                matrix.rowPointers[static_cast<size_t>(i)] = static_cast<Index>(value);
            }
        }
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read rowPointers (v2)");

        uint64_t colSize = 0;
        file.read(reinterpret_cast<char*>(&colSize), sizeof(colSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read colSize (v2)");

        if (indexByteSize == sizeof(Index)) {
            matrix.colIndices.resize(static_cast<size_t>(colSize));
            if (colSize > 0) {
                file.read(reinterpret_cast<char*>(matrix.colIndices.data()),
                          static_cast<std::streamsize>(colSize * sizeof(Index)));
            }
        } else {
            std::vector<uint8_t> rawBuffer(
                static_cast<size_t>(colSize) * static_cast<size_t>(indexByteSize));
            if (colSize > 0) {
                file.read(reinterpret_cast<char*>(rawBuffer.data()),
                          static_cast<std::streamsize>(rawBuffer.size()));
            }
            matrix.colIndices.resize(static_cast<size_t>(colSize));
            for (uint64_t i = 0; i < colSize; ++i) {
                uint64_t value = 0;
                if (indexByteSize == 4) {
                    uint32_t v32 = 0;
                    std::memcpy(&v32, rawBuffer.data() + i * 4, 4);
                    value = static_cast<uint64_t>(v32);
                } else {
                    std::memcpy(&value, rawBuffer.data() + i * 8, 8);
                }
                matrix.colIndices[static_cast<size_t>(i)] = static_cast<Index>(value);
            }
        }
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read colIndices (v2)");

        uint64_t valSize = 0;
        file.read(reinterpret_cast<char*>(&valSize), sizeof(valSize));
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read valSize (v2)");
        matrix.values.resize(static_cast<size_t>(valSize));
        if (valSize > 0) {
            file.read(reinterpret_cast<char*>(matrix.values.data()),
                      static_cast<std::streamsize>(valSize * sizeof(Scalar)));
        }
        if (!file.good()) throw std::runtime_error("SparseIO: Failed to read values (v2)");

        uint64_t checksumRowPtrOnFile = 0;
        uint64_t checksumNNZOnFile = 0;
        file.read(reinterpret_cast<char*>(&checksumRowPtrOnFile), sizeof(checksumRowPtrOnFile));
        file.read(reinterpret_cast<char*>(&checksumNNZOnFile),   sizeof(checksumNNZOnFile));
        if (!file.good()) {
            throw std::runtime_error("SparseIO: Failed to read trailing checksums (v2)");
        }

        const uint64_t actualRowPtrBack = static_cast<uint64_t>(
            matrix.rowPointers.empty() ? 0 : matrix.rowPointers.back());
        const uint64_t actualNNZ = static_cast<uint64_t>(matrix.nnz);

        if (checksumRowPtrOnFile != actualRowPtrBack) {
            throw std::runtime_error(
                "SparseIO: CHECKSUM FAIL - rowPointers.back() mismatch! "
                "file=" + std::to_string(checksumRowPtrOnFile) +
                " actual=" + std::to_string(actualRowPtrBack) +
                " [Partial non-zero elements LOST during original write]");
        }
        if (checksumNNZOnFile != actualNNZ) {
            throw std::runtime_error(
                "SparseIO: CHECKSUM FAIL - nnz mismatch! "
                "file=" + std::to_string(checksumNNZOnFile) +
                " actual=" + std::to_string(actualNNZ) +
                " [Non-zero count truncated/corrupted]");
        }
    } else {
        throw std::runtime_error("SparseIO: Unsupported file version=" +
                                  std::to_string(version) + " in " + filename);
    }

    if (matrix.rows < 0 || matrix.cols < 0 || matrix.nnz < 0) {
        throw std::runtime_error("SparseIO: Corrupted file - negative dimensions after read");
    }
    if (matrix.nnz > 0 && !matrix.rowPointers.empty()) {
        if (matrix.rowPointers.size() != static_cast<size_t>(matrix.rows) + 1) {
            throw std::runtime_error(
                "SparseIO: Post-read check FAIL - rowPointers size=" +
                std::to_string(matrix.rowPointers.size()) +
                " expected rows+1=" + std::to_string(matrix.rows + 1));
        }
        if (matrix.rowPointers[0] != 0) {
            throw std::runtime_error(
                "SparseIO: Post-read check FAIL - rowPointers[0]=" +
                std::to_string(matrix.rowPointers[0]) + " (expected 0)");
        }
        if (matrix.rowPointers[matrix.rows] != matrix.nnz) {
            throw std::runtime_error(
                "SparseIO: Post-read check FAIL - rowPointers[rows]=" +
                std::to_string(matrix.rowPointers[matrix.rows]) +
                " != nnz=" + std::to_string(matrix.nnz) +
                " [NON-ZERO ELEMENTS LIKELY LOST!]");
        }
        for (Index i = 0; i < matrix.rows; ++i) {
            if (matrix.rowPointers[i + 1] < matrix.rowPointers[i]) {
                throw std::runtime_error(
                    "SparseIO: Post-read check FAIL - rowPointers non-monotonic at row=" +
                    std::to_string(i));
            }
        }
    }
    if (matrix.colIndices.size() != static_cast<size_t>(matrix.nnz)) {
        throw std::runtime_error(
            "SparseIO: Post-read check FAIL - colIndices.size()=" +
            std::to_string(matrix.colIndices.size()) +
            " != nnz=" + std::to_string(matrix.nnz));
    }
    if (matrix.values.size() != static_cast<size_t>(matrix.nnz)) {
        throw std::runtime_error(
            "SparseIO: Post-read check FAIL - values.size()=" +
            std::to_string(matrix.values.size()) +
            " != nnz=" + std::to_string(matrix.nnz));
    }

    file.peek();
    if (file.bad()) {
        throw std::runtime_error("SparseIO: I/O error (badbit) after read completion");
    }

    file.close();
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
