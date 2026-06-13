#ifndef FEM_IO_SPARSE_IO_H
#define FEM_IO_SPARSE_IO_H

#include "common/Types.h"
#include "sparse/SparseMatrix.h"
#include <string>

namespace fem {
namespace io {

class SparseIO {
public:
    SparseIO();
    ~SparseIO();

    bool writeCSRBinary(const sparse::CSRMatrix& matrix,
                        const std::string& filename) const;
    sparse::CSRMatrix readCSRBinary(const std::string& filename) const;

    bool writeCOOASCII(const sparse::COOMatrix& matrix,
                       const std::string& filename) const;
    sparse::COOMatrix readCOOASCII(const std::string& filename) const;

    bool writeMatrixMarket(const SparseMatrix& matrix,
                           const std::string& filename) const;

    bool writeDenseMatrixASCII(const DenseMatrix& matrix,
                               const std::string& filename) const;
    DenseMatrix readDenseMatrixASCII(const std::string& filename) const;

private:
    static constexpr uint32_t MAGIC_NUMBER = 0x46454D53;
    static constexpr uint32_t VERSION = 1;
};

} // namespace io
} // namespace fem

#endif // FEM_IO_SPARSE_IO_H
