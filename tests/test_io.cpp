#include "io/MeshIO.h"
#include "io/SparseIO.h"
#include "io/ResultIO.h"
#include "mesh/MeshGenerator.h"
#include "sparse/SparseMatrix.h"
#include "solver/CGSolver.h"
#include "assembly/StiffnessAssembler.h"
#include <fstream>
#include <cstdio>
#include <cmath>

using namespace fem;
using namespace fem::io;
using namespace fem::mesh;
using namespace fem::sparse;
using namespace fem::solver;
using namespace fem::assembly;

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static void removeFile(const std::string& path) {
    std::remove(path.c_str());
}

TEST_CASE(mesh_io_write_and_read_roundtrip) {
    MeshGenerator generator;
    Mesh original = generator.generateStructuredGrid2D(2.0, 3.0, 2, 3,
                                                       2.1e11, 0.3, 0.1);

    std::string filename = "test_mesh_io_roundtrip.txt";
    MeshIO meshIO;

    ASSERT_TRUE(meshIO.writeMesh(original, filename));
    ASSERT_TRUE(fileExists(filename));

    Mesh loaded = meshIO.readMesh(filename);

    ASSERT_EQ(loaded.nodes.size(), original.nodes.size());
    ASSERT_EQ(loaded.elements.size(), original.elements.size());
    ASSERT_EQ(loaded.totalDofs, original.totalDofs);
    ASSERT_EQ(loaded.dofsPerNode, original.dofsPerNode);

    for (size_t i = 0; i < original.nodes.size(); ++i) {
        ASSERT_NEAR(loaded.nodes[i].x, original.nodes[i].x, 1e-10);
        ASSERT_NEAR(loaded.nodes[i].y, original.nodes[i].y, 1e-10);
        ASSERT_EQ(loaded.nodes[i].id, original.nodes[i].id);
    }

    for (size_t i = 0; i < original.elements.size(); ++i) {
        ASSERT_EQ(loaded.elements[i].nodeIds.size(), original.elements[i].nodeIds.size());
        ASSERT_NEAR(loaded.elements[i].youngsModulus, original.elements[i].youngsModulus, 1e-6);
    }

    removeFile(filename);
}

TEST_CASE(mesh_io_write_vtk_creates_file) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2,
                                                   2.1e11, 0.3, 0.1);

    DenseVector displacements = DenseVector::Zero(mesh.totalDofs);
    for (Index i = 0; i < mesh.totalDofs; i += 2) {
        displacements(i) = 0.001 * static_cast<Scalar>(i);
    }

    std::string filename = "test_mesh.vtk";
    MeshIO meshIO;

    ASSERT_TRUE(meshIO.writeMeshVTK(mesh, displacements, filename));
    ASSERT_TRUE(fileExists(filename));

    std::ifstream f(filename);
    std::string line;
    bool hasHeader = false;
    while (std::getline(f, line)) {
        if (line.find("vtk DataFile") != std::string::npos) {
            hasHeader = true;
            break;
        }
    }
    ASSERT_TRUE(hasHeader);

    removeFile(filename);
}

TEST_CASE(sparse_io_csr_binary_roundtrip) {
    DenseMatrix dense = DenseMatrix::Zero(10, 10);
    for (int i = 0; i < 10; ++i) {
        dense(i, i) = static_cast<Scalar>(i + 1);
        if (i > 0) dense(i, i - 1) = -0.5;
        if (i < 9) dense(i, i + 1) = -0.3;
    }
    dense(0, 5) = 99.0;
    dense(5, 0) = 99.0;

    SparseConverter converter;
    CSRMatrix original = converter.denseToCSR(dense);

    std::string filename = "test_csr_binary.csr";
    SparseIO sparseIO;

    ASSERT_TRUE(sparseIO.writeCSRBinary(original, filename));
    ASSERT_TRUE(fileExists(filename));

    CSRMatrix loaded = sparseIO.readCSRBinary(filename);

    ASSERT_EQ(loaded.rows, original.rows);
    ASSERT_EQ(loaded.cols, original.cols);
    ASSERT_EQ(loaded.nnz, original.nnz);
    ASSERT_EQ(loaded.rowPointers.size(), original.rowPointers.size());
    ASSERT_EQ(loaded.colIndices.size(), original.colIndices.size());
    ASSERT_EQ(loaded.values.size(), original.values.size());

    DenseMatrix matOriginal = converter.csrToDense(original);
    DenseMatrix matLoaded = converter.csrToDense(loaded);

    for (Index i = 0; i < 10; ++i) {
        for (Index j = 0; j < 10; ++j) {
            ASSERT_NEAR(matOriginal(i, j), matLoaded(i, j), 1e-12);
        }
    }

    removeFile(filename);
}

TEST_CASE(sparse_io_coo_ascii_roundtrip) {
    DenseMatrix dense = DenseMatrix::Zero(6, 6);
    dense(0, 0) = 1.5;
    dense(1, 3) = 2.5;
    dense(3, 1) = -1.0;
    dense(5, 5) = 7.0;
    dense(2, 4) = 3.3;

    SparseConverter converter;
    COOMatrix original = converter.denseToCOO(dense);

    std::string filename = "test_coo_ascii.txt";
    SparseIO sparseIO;

    ASSERT_TRUE(sparseIO.writeCOOASCII(original, filename));
    ASSERT_TRUE(fileExists(filename));

    COOMatrix loaded = sparseIO.readCOOASCII(filename);

    DenseMatrix matOriginal = converter.cooToDense(original);
    DenseMatrix matLoaded = converter.cooToDense(loaded);

    for (Index i = 0; i < 6; ++i) {
        for (Index j = 0; j < 6; ++j) {
            ASSERT_NEAR(matOriginal(i, j), matLoaded(i, j), 1e-12);
        }
    }

    removeFile(filename);
}

TEST_CASE(sparse_io_dense_matrix_ascii_roundtrip) {
    DenseMatrix original(4, 5);
    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 5; ++j) {
            original(i, j) = static_cast<Scalar>(i * 10 + j) * 0.1;
        }
    }

    std::string filename = "test_dense_ascii.txt";
    SparseIO sparseIO;

    ASSERT_TRUE(sparseIO.writeDenseMatrixASCII(original, filename));
    ASSERT_TRUE(fileExists(filename));

    DenseMatrix loaded = sparseIO.readDenseMatrixASCII(filename);

    ASSERT_EQ(loaded.rows(), original.rows());
    ASSERT_EQ(loaded.cols(), original.cols());
    for (Index i = 0; i < original.rows(); ++i) {
        for (Index j = 0; j < original.cols(); ++j) {
            ASSERT_NEAR(loaded(i, j), original(i, j), 1e-10);
        }
    }

    removeFile(filename);
}

TEST_CASE(result_io_write_displacements) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2,
                                                   2.1e11, 0.3, 0.1);

    DenseVector displacements(mesh.totalDofs);
    for (Index i = 0; i < mesh.totalDofs; ++i) {
        displacements(i) = static_cast<Scalar>(i) * 0.0001;
    }

    std::string filename = "test_displacements.txt";
    ResultIO resultIO;

    ASSERT_TRUE(resultIO.writeDisplacements(displacements, mesh, filename));
    ASSERT_TRUE(fileExists(filename));

    removeFile(filename);
}

TEST_CASE(result_io_write_simulation_summary) {
    SimulationResult result;
    result.converged = true;
    result.iterations = 150;
    result.residual = 1.5e-10;
    result.solveTimeSeconds = 0.1234;
    result.displacements = DenseVector::Constant(20, 0.005);
    result.stresses = DenseVector::Constant(9, 1.0e6);

    std::string filename = "test_summary.txt";
    ResultIO resultIO;

    ASSERT_TRUE(resultIO.writeSimulationSummary(result, filename));
    ASSERT_TRUE(fileExists(filename));

    removeFile(filename);
}

TEST_CASE(result_io_write_node_displacements_vtk) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 3, 2,
                                                   2.1e11, 0.3, 0.1);

    DenseVector displacements = DenseVector::Zero(mesh.totalDofs);
    for (Index i = 0; i < mesh.nodes.size(); ++i) {
        displacements(2 * i) = 0.01 * mesh.nodes[i].x;
        displacements(2 * i + 1) = 0.005 * mesh.nodes[i].y;
    }

    std::string filename = "test_result_vtk.vtk";
    ResultIO resultIO;

    ASSERT_TRUE(resultIO.writeNodeDisplacementsVTK(mesh, displacements, filename));
    ASSERT_TRUE(fileExists(filename));

    removeFile(filename);
}

TEST_CASE(sparse_io_matrix_market_format) {
    Index n = 5;
    SparseMatrix mat(n, n);
    std::vector<Triplet> triplets;
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(1, 2, 2.0);
    triplets.emplace_back(3, 1, 3.0);
    triplets.emplace_back(4, 4, 4.0);
    mat.setFromTriplets(triplets.begin(), triplets.end());

    std::string filename = "test_matrix_market.mtx";
    SparseIO sparseIO;

    ASSERT_TRUE(sparseIO.writeMatrixMarket(mat, filename));
    ASSERT_TRUE(fileExists(filename));

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line);
    ASSERT_TRUE(line.find("MatrixMarket") != std::string::npos);

    removeFile(filename);
}

TEST_CASE(mesh_io_write_node_coords_and_connectivity) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 2.0, 2, 2,
                                                   2.1e11, 0.3, 0.1);

    std::string coordsFile = "test_nodes.txt";
    std::string connFile = "test_elements.txt";
    MeshIO meshIO;

    ASSERT_TRUE(meshIO.writeNodeCoords(mesh, coordsFile));
    ASSERT_TRUE(meshIO.writeElementConnectivity(mesh, connFile));
    ASSERT_TRUE(fileExists(coordsFile));
    ASSERT_TRUE(fileExists(connFile));

    removeFile(coordsFile);
    removeFile(connFile);
}
