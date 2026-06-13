#include "solver/CGSolver.h"
#include "sparse/SparseMatrix.h"
#include <cmath>

using namespace fem;
using namespace fem::solver;
using namespace fem::sparse;

TEST_CASE(cg_solver_simple_diagonal_system) {
    Index n = 5;
    DenseMatrix A = DenseMatrix::Identity(n, n);
    DenseVector b(n);
    b << 1.0, 2.0, 3.0, 4.0, 5.0;

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 1000;
    CGSolver solver(config);

    DenseVector x = solver.solve(A, b);

    ASSERT_EQ(x.size(), n);
    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x(i), b(i), 1e-10);
    }
    ASSERT_TRUE(solver.getLastConverged());
}

TEST_CASE(cg_solver_tridiagonal_system) {
    Index n = 10;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }

    DenseVector b = DenseVector::Ones(n);

    SolverConfig config;
    config.tolerance = 1e-10;
    config.maxIterations = 10000;
    CGSolver solver(config);

    DenseVector x = solver.solve(A, b);
    DenseVector residual = A * x - b;

    ASSERT_LT(residual.norm(), 1e-8);
    ASSERT_TRUE(solver.getLastConverged());
}

TEST_CASE(cg_solver_sparse_csr_format) {
    Index n = 8;
    DenseMatrix dense = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        dense(i, i) = 3.0;
        if (i > 0) dense(i, i - 1) = -0.5;
        if (i < n - 1) dense(i, i + 1) = -0.5;
    }

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    DenseVector b(n);
    for (Index i = 0; i < n; ++i) b(i) = static_cast<Scalar>(i + 1);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 5000;
    CGSolver solver(config);

    DenseVector x = solver.solve(csr, b);
    DenseVector residual = dense * x - b;

    ASSERT_LT(residual.norm(), 1e-8);
}

TEST_CASE(cg_solver_eigen_sparse_format) {
    Index n = 12;
    SparseMatrix A(n, n);
    std::vector<Triplet> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 5.0);
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
        if (i > 1) triplets.emplace_back(i, i - 2, -0.5);
        if (i < n - 2) triplets.emplace_back(i, i + 2, -0.5);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    DenseVector b = DenseVector::LinSpaced(n, 1.0, static_cast<Scalar>(n));

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 10000;
    CGSolver solver(config);
    solver.setPreconditioner(PreconditionerType::Jacobi);

    SimulationResult result = solver.solveWithStats(A, b);

    ASSERT_TRUE(result.converged);
    ASSERT_GT(result.iterations, 0);
    ASSERT_GT(result.displacements.size(), 0);
}

TEST_CASE(cg_solver_iteration_count_tracking) {
    Index n = 20;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 2.0 + static_cast<Scalar>(i) / 10.0;
        if (i > 0) A(i, i - 1) = -0.3;
        if (i < n - 1) A(i, i + 1) = -0.3;
    }
    DenseVector b = DenseVector::Constant(n, 1.5);

    SolverConfig config;
    config.tolerance = 1e-10;
    config.maxIterations = 1000;
    CGSolver solver(config);

    DenseVector x = solver.solve(A, b);

    ASSERT_GT(solver.getLastIterations(), 0);
    ASSERT_LT(solver.getLastIterations(), 1000);
    ASSERT_LT(solver.getLastResidual(), 1e-6);
}

TEST_CASE(cg_solver_jacobi_preconditioner) {
    Index n = 50;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 100.0 + static_cast<Scalar>(i);
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }
    DenseVector b = DenseVector::Random(n);

    SolverConfig config;
    config.tolerance = 1e-10;
    config.maxIterations = 5000;

    CGSolver solverNoPrec(config);
    solverNoPrec.setPreconditioner(PreconditionerType::None);
    DenseVector x1 = solverNoPrec.solve(A, b);
    Index iterNoPrec = solverNoPrec.getLastIterations();

    CGSolver solverJacobi(config);
    solverJacobi.setPreconditioner(PreconditionerType::Jacobi);
    DenseVector x2 = solverJacobi.solve(A, b);
    Index iterJacobi = solverJacobi.getLastIterations();

    ASSERT_TRUE(solverNoPrec.getLastConverged());
    ASSERT_TRUE(solverJacobi.getLastConverged());
    ASSERT_EQ(x1.size(), x2.size());
    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x1(i), x2(i), 1e-8);
    }
}

TEST_CASE(cg_solver_zero_rhs) {
    Index n = 10;
    DenseMatrix A = DenseMatrix::Identity(n, n);
    DenseVector b = DenseVector::Zero(n);

    SolverConfig config;
    config.tolerance = 1e-12;
    CGSolver solver(config);

    DenseVector x = solver.solve(A, b);

    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x(i), 0.0, 1e-15);
    }
}

TEST_CASE(cg_solver_solve_with_stats_measures_time) {
    Index n = 30;
    SparseMatrix A(n, n);
    std::vector<Triplet> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 6.0);
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    DenseVector b = DenseVector::Ones(n);

    SolverConfig config;
    config.tolerance = 1e-10;
    config.maxIterations = 10000;
    CGSolver solver(config);

    SimulationResult result = solver.solveWithStats(A, b);

    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.displacements.size(), n);
    ASSERT_GT(result.solveTimeSeconds, 0.0);
    ASSERT_GE(result.iterations, 1);
}
