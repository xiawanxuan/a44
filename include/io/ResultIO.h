#ifndef FEM_IO_RESULT_IO_H
#define FEM_IO_RESULT_IO_H

#include "common/Types.h"
#include <string>

namespace fem {
namespace io {

class ResultIO {
public:
    ResultIO();
    ~ResultIO();

    bool writeDisplacements(const DenseVector& displacements,
                            const Mesh& mesh,
                            const std::string& filename) const;

    bool writeStresses(const DenseVector& stresses,
                       const Mesh& mesh,
                       const std::string& filename) const;

    bool writeSimulationSummary(const SimulationResult& result,
                                const std::string& filename) const;

    bool writeNodeDisplacementsVTK(const Mesh& mesh,
                                   const DenseVector& displacements,
                                   const std::string& filename) const;

    bool exportAllResults(const Mesh& mesh,
                          const SimulationResult& result,
                          const std::string& outputDir) const;

private:
    bool ensureDirectory(const std::string& path) const;
};

} // namespace io
} // namespace fem

#endif // FEM_IO_RESULT_IO_H
