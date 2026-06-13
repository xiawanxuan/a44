#include "assembly/StiffnessAssembler.h"
#include "mesh/MeshGenerator.h"
#include <cmath>

using namespace fem;
using namespace fem::assembly;
using namespace fem::mesh;

TEST_CASE(assembler_element_stiffness_is_symmetric) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 1, 1,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    DenseMatrix Ke = assembler.computeElementStiffness(
        mesh.elements[0], mesh.nodes);

    ASSERT_EQ(Ke.rows(), Ke.cols());
    for (int i = 0; i < Ke.rows(); ++i) {
        for (int j = 0; j < Ke.cols(); ++j) {
            ASSERT_NEAR(Ke(i, j), Ke(j, i), 1e-8);
        }
    }
}

TEST_CASE(assembler_dense_matrix_is_symmetric) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 2.0, 2, 3,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    DenseMatrix K = assembler.assembleDenseMatrix(mesh);

    ASSERT_EQ(K.rows(), mesh.totalDofs);
    ASSERT_EQ(K.cols(), mesh.totalDofs);

    for (Index i = 0; i < K.rows(); ++i) {
        for (Index j = 0; j < K.cols(); ++j) {
            ASSERT_NEAR(K(i, j), K(j, i), 1e-8);
        }
    }
}

TEST_CASE(assembler_sparse_matrix_dimensions) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(2.0, 1.0, 3, 2,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    SparseMatrix K = assembler.assembleSparseMatrix(mesh);

    ASSERT_EQ(K.rows(), mesh.totalDofs);
    ASSERT_EQ(K.cols(), mesh.totalDofs);
    ASSERT_GT(K.nonZeros(), 0);
}

TEST_CASE(assembler_dense_sparse_equivalent) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    DenseMatrix Kdense = assembler.assembleDenseMatrix(mesh);
    SparseMatrix Ksparse = assembler.assembleSparseMatrix(mesh);

    DenseMatrix KfromSparse = Ksparse;

    ASSERT_EQ(Kdense.rows(), KfromSparse.rows());
    ASSERT_EQ(Kdense.cols(), KfromSparse.cols());
    for (Index i = 0; i < Kdense.rows(); ++i) {
        for (Index j = 0; j < Kdense.cols(); ++j) {
            ASSERT_NEAR(Kdense(i, j), KfromSparse(i, j), 1e-8);
        }
    }
}

TEST_CASE(assembler_constitutive_matrix_2d) {
    StiffnessAssembler assembler;
    Scalar E = 2.1e11;
    Scalar nu = 0.3;

    DenseMatrix D = assembler.computeConstitutiveMatrix(E, nu);

    ASSERT_EQ(D.rows(), 3);
    ASSERT_EQ(D.cols(), 3);

    Scalar factor = E / (1.0 - nu * nu);
    ASSERT_NEAR(D(0, 0), factor * 1.0, 1e-6);
    ASSERT_NEAR(D(0, 1), factor * nu, 1e-6);
    ASSERT_NEAR(D(1, 0), factor * nu, 1e-6);
    ASSERT_NEAR(D(1, 1), factor * 1.0, 1e-6);
    ASSERT_NEAR(D(2, 2), factor * 0.5 * (1.0 - nu), 1e-6);
}

TEST_CASE(assembler_load_vector_assembly) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 2, 2,
                                                    2.1e11, 0.3, 0.1);

    std::vector<Load> loads = {
        {0, 0, 100.0},
        {0, 1, 200.0},
        {4, 1, -500.0}
    };

    StiffnessAssembler assembler;
    DenseVector F = assembler.assembleLoadVector(mesh, loads);

    ASSERT_EQ(F.size(), mesh.totalDofs);
    ASSERT_NEAR(F(0), 100.0, 1e-15);
    ASSERT_NEAR(F(1), 200.0, 1e-15);
    ASSERT_NEAR(F(4 * 2 + 1), -500.0, 1e-15);
    ASSERT_NEAR(F(2), 0.0, 1e-15);
}

TEST_CASE(assembler_boundary_conditions_dense) {
    Index n = 6;
    DenseMatrix K = DenseMatrix::Identity(n, n);
    DenseVector F = DenseVector::Ones(n);

    std::vector<BoundaryCondition> bcs = {
        {0, 0, 0.0},
        {1, 0, 0.0}
    };

    StiffnessAssembler assembler;
    Mesh dummyMesh;
    dummyMesh.dofsPerNode = 2;
    dummyMesh.nodes.resize(n / 2);
    dummyMesh.totalDofs = n;

    assembler.applyBoundaryConditions(K, F, bcs);

    ASSERT_NEAR(K(0, 0), 1.0, 1e-15);
    ASSERT_NEAR(K(2, 2), 1.0, 1e-15);
    ASSERT_NEAR(F(0), 0.0, 1e-15);
    ASSERT_NEAR(F(2), 0.0, 1e-15);

    for (Index j = 0; j < n; ++j) {
        if (j != 0) ASSERT_NEAR(K(0, j), 0.0, 1e-15);
        if (j != 2) ASSERT_NEAR(K(2, j), 0.0, 1e-15);
    }
}

TEST_CASE(assembler_sparse_matrix_symmetry) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 3, 2,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    SparseMatrix K = assembler.assembleSparseMatrix(mesh);
    DenseMatrix Kdense = K;

    for (Index i = 0; i < Kdense.rows(); ++i) {
        for (Index j = 0; j < Kdense.cols(); ++j) {
            ASSERT_NEAR(Kdense(i, j), Kdense(j, i), 1e-8);
        }
    }
}
