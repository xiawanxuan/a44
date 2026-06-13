#ifndef FEM_IO_MESH_IO_H
#define FEM_IO_MESH_IO_H

#include "common/Types.h"
#include <string>

namespace fem {
namespace io {

class MeshIO {
public:
    MeshIO();
    ~MeshIO();

    bool writeMesh(const Mesh& mesh, const std::string& filename) const;
    Mesh readMesh(const std::string& filename) const;

    bool writeMeshVTK(const Mesh& mesh,
                      const DenseVector& displacements,
                      const std::string& filename) const;

    bool writeNodeCoords(const Mesh& mesh,
                         const std::string& filename) const;
    bool writeElementConnectivity(const Mesh& mesh,
                                  const std::string& filename) const;

private:
    bool parseMeshFile(const std::string& filename, Mesh& mesh) const;
    std::string trim(const std::string& s) const;
};

} // namespace io
} // namespace fem

#endif // FEM_IO_MESH_IO_H
