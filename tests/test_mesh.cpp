#include "mesh/MeshGenerator.h"
#include <stdexcept>

using namespace fem;
using namespace fem::mesh;

TEST_CASE(mesh_generator_creates_correct_number_of_nodes) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 3,
                                                    2.1e11, 0.3, 0.1);

    Index expectedNodes = (2 + 1) * (3 + 1);
    ASSERT_EQ(mesh.nodes.size(), expectedNodes);
}

TEST_CASE(mesh_generator_creates_correct_number_of_elements) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(2.0, 3.0, 4, 5,
                                                    2.1e11, 0.3, 0.1);

    Index expectedElements = 4 * 5;
    ASSERT_EQ(mesh.elements.size(), expectedElements);
}

TEST_CASE(mesh_generator_sets_correct_total_dofs) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 3, 3,
                                                    2.1e11, 0.3, 0.1);

    Index expectedNodes = (3 + 1) * (3 + 1);
    Index expectedDofs = expectedNodes * 2;
    ASSERT_EQ(mesh.totalDofs, expectedDofs);
}

TEST_CASE(mesh_generator_node_coordinates_are_correct) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(2.0, 3.0, 2, 3,
                                                    2.1e11, 0.3, 0.1);

    Index numNodesX = 3;

    Index idx0 = MeshGenerator::getNodeIndex(0, 0, numNodesX);
    ASSERT_NEAR(mesh.nodes[idx0].x, 0.0, 1e-10);
    ASSERT_NEAR(mesh.nodes[idx0].y, 0.0, 1e-10);

    Index idx22 = MeshGenerator::getNodeIndex(2, 3, numNodesX);
    ASSERT_NEAR(mesh.nodes[idx22].x, 2.0, 1e-10);
    ASSERT_NEAR(mesh.nodes[idx22].y, 3.0, 1e-10);

    Index idx11 = MeshGenerator::getNodeIndex(1, 1, numNodesX);
    ASSERT_NEAR(mesh.nodes[idx11].x, 1.0, 1e-10);
    ASSERT_NEAR(mesh.nodes[idx11].y, 1.0, 1e-10);
}

TEST_CASE(mesh_generator_element_has_four_nodes) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2,
                                                    2.1e11, 0.3, 0.1);

    for (const auto& elem : mesh.elements) {
        ASSERT_EQ(elem.nodeIds.size(), 4);
    }
}

TEST_CASE(mesh_generator_element_material_properties) {
    MeshGenerator generator;
    Scalar E = 2.1e11;
    Scalar nu = 0.3;
    Scalar t = 0.1;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2, E, nu, t);

    for (const auto& elem : mesh.elements) {
        ASSERT_NEAR(elem.youngsModulus, E, 1e-6);
        ASSERT_NEAR(elem.poissonRatio, nu, 1e-10);
        ASSERT_NEAR(elem.thickness, t, 1e-10);
    }
}

TEST_CASE(mesh_generator_invalid_params_throws) {
    MeshGenerator generator;
    bool threw = false;
    try {
        generator.generateStructuredGrid2D(1.0, 1.0, 0, 2,
                                            2.1e11, 0.3, 0.1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

TEST_CASE(mesh_generator_node_indexing_consistency) {
    Index nx = 5;
    Index ny = 4;

    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            Index idx = MeshGenerator::getNodeIndex(i, j, nx);
            ASSERT_GT(idx, -1);
        }
    }

    Index total = MeshGenerator::getNodeIndex(nx - 1, ny - 1, nx) + 1;
    ASSERT_EQ(total, nx * ny);
}
