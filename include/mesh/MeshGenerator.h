#ifndef FEM_MESH_MESH_GENERATOR_H
#define FEM_MESH_MESH_GENERATOR_H

#include "common/Types.h"
#include <vector>

namespace fem {
namespace mesh {

class MeshGenerator {
public:
    MeshGenerator();
    ~MeshGenerator();

    Mesh generateStructuredGrid2D(Scalar width, Scalar height,
                                  Index numElementsX, Index numElementsY,
                                  Scalar youngsModulus, Scalar poissonRatio,
                                  Scalar thickness);

    Mesh generateRectangularMesh(Scalar length, Scalar width,
                                 Index nx, Index ny,
                                 Scalar youngsModulus, Scalar poissonRatio,
                                 Scalar thickness);

    static Index getNodeIndex(Index i, Index j, Index numNodesX);
    static Index getElementIndex(Index i, Index j, Index numElementsX);

private:
    void generateNodes(Mesh& mesh, Scalar width, Scalar height,
                       Index numNodesX, Index numNodesY);
    void generateQuadElements(Mesh& mesh,
                              Index numElementsX, Index numElementsY,
                              Scalar youngsModulus, Scalar poissonRatio,
                              Scalar thickness);
    void computeTotalDofs(Mesh& mesh);
};

} // namespace mesh
} // namespace fem

#endif // FEM_MESH_MESH_GENERATOR_H
