#include "mesh/MeshGenerator.h"
#include <cmath>
#include <stdexcept>

namespace fem {
namespace mesh {

MeshGenerator::MeshGenerator() = default;
MeshGenerator::~MeshGenerator() = default;

Mesh MeshGenerator::generateStructuredGrid2D(Scalar width, Scalar height,
                                             Index numElementsX, Index numElementsY,
                                             Scalar youngsModulus, Scalar poissonRatio,
                                             Scalar thickness) {
    if (numElementsX <= 0 || numElementsY <= 0) {
        throw std::invalid_argument("Number of elements must be positive");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    Mesh mesh;
    mesh.dofsPerNode = 2;

    Index numNodesX = numElementsX + 1;
    Index numNodesY = numElementsY + 1;

    generateNodes(mesh, width, height, numNodesX, numNodesY);
    generateQuadElements(mesh, numElementsX, numElementsY,
                         youngsModulus, poissonRatio, thickness);
    computeTotalDofs(mesh);

    return mesh;
}

Mesh MeshGenerator::generateRectangularMesh(Scalar length, Scalar width,
                                            Index nx, Index ny,
                                            Scalar youngsModulus, Scalar poissonRatio,
                                            Scalar thickness) {
    return generateStructuredGrid2D(length, width, nx, ny,
                                    youngsModulus, poissonRatio, thickness);
}

Index MeshGenerator::getNodeIndex(Index i, Index j, Index numNodesX) {
    return j * numNodesX + i;
}

Index MeshGenerator::getElementIndex(Index i, Index j, Index numElementsX) {
    return j * numElementsX + i;
}

void MeshGenerator::generateNodes(Mesh& mesh, Scalar width, Scalar height,
                                  Index numNodesX, Index numNodesY) {
    mesh.nodes.reserve(numNodesX * numNodesY);

    Scalar dx = width / static_cast<Scalar>(numNodesX - 1);
    Scalar dy = height / static_cast<Scalar>(numNodesY - 1);

    for (Index j = 0; j < numNodesY; ++j) {
        for (Index i = 0; i < numNodesX; ++i) {
            Index id = getNodeIndex(i, j, numNodesX);
            Scalar x = static_cast<Scalar>(i) * dx;
            Scalar y = static_cast<Scalar>(j) * dy;
            mesh.nodes.emplace_back(id, x, y, 0.0);
        }
    }
}

void MeshGenerator::generateQuadElements(Mesh& mesh,
                                         Index numElementsX, Index numElementsY,
                                         Scalar youngsModulus, Scalar poissonRatio,
                                         Scalar thickness) {
    Index numNodesX = numElementsX + 1;
    mesh.elements.reserve(numElementsX * numElementsY);

    for (Index j = 0; j < numElementsY; ++j) {
        for (Index i = 0; i < numElementsX; ++i) {
            Element elem;
            elem.id = getElementIndex(i, j, numElementsX);
            elem.youngsModulus = youngsModulus;
            elem.poissonRatio = poissonRatio;
            elem.thickness = thickness;

            elem.nodeIds.resize(4);
            elem.nodeIds[0] = getNodeIndex(i, j, numNodesX);
            elem.nodeIds[1] = getNodeIndex(i + 1, j, numNodesX);
            elem.nodeIds[2] = getNodeIndex(i + 1, j + 1, numNodesX);
            elem.nodeIds[3] = getNodeIndex(i, j + 1, numNodesX);

            mesh.elements.push_back(elem);
        }
    }
}

void MeshGenerator::computeTotalDofs(Mesh& mesh) {
    mesh.totalDofs = static_cast<Index>(mesh.nodes.size()) * mesh.dofsPerNode;
}

} // namespace mesh
} // namespace fem
