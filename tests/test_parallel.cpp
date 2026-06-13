#include "solver/CGSolver.h"
#include "solver/ParallelCGSolver.h"
#include "sparse/SparseMatrix.h"
#include "mesh/MeshGenerator.h"
#include "assembly/StiffnessAssembler.h"
#include <cmath>

using namespace fem;
using namespace fem::solver;
using namespace fem::sparse;
using namespace fem::mesh;
using namespace fem::assembly;

TEST_CASE(parallel_solver_diagonal_system_1thread) {
    Index n = 10;
    DenseMatrix A = DenseMatrix::Identity(n, n);
    DenseVector b(n);
    for (Index i = 0; i < n; ++i) b(i) = static_cast<Scalar>(i + 1);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 1000;
    config.useParallel = true;
    config.numThreads = 1;

    CGSolver solver(config);
    solver.setUseParallel(true);
    solver.setNumThreads(1);

    DenseVector x = solver.solve(A, b);

    ASSERT_EQ(x.size(), n);
    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x(i), b(i), 1e-10);
    }
    ASSERT_TRUE(solver.getLastConverged());
    ASSERT_EQ(solver.getLastUsedThreads(), 1);
    ASSERT_TRUE(solver.isParallelEnabled());
}

TEST_CASE(parallel_solver_tridiagonal_parallel) {
    Index n = 50;
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
    config.useParallel = true;

    CGSolver solver(config);
    DenseVector x = solver.solve(A, b);

    DenseVector residual = A * x - b;
    ASSERT_LT(residual.norm(), 1e-8);
    ASSERT_TRUE(solver.getLastConverged());
    ASSERT_GT(solver.getLastUsedThreads(), 0);
}

TEST_CASE(parallel_vs_serial_result_equivalence) {
    Index n = 30;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 5.0 + 0.1 * static_cast<Scalar>(i);
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
        if (i > 2) A(i, i - 3) = -0.2;
        if (i < n - 3) A(i, i + 3) = -0.2;
    }
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (i != j) {
                A(j, i) = A(i, j);
            }
        }
    }

    DenseVector b = DenseVector::Random(n);

    SolverConfig cfgSerial;
    cfgSerial.tolerance = 1e-12;
    cfgSerial.maxIterations = 10000;
    cfgSerial.useParallel = false;

    SolverConfig cfgParallel;
    cfgParallel.tolerance = 1e-12;
    cfgParallel.maxIterations = 10000;
    cfgParallel.useParallel = true;
    cfgParallel.numThreads = 2;

    CGSolver solverSerial(cfgSerial);
    solverSerial.setUseParallel(false);
    solverSerial.setPreconditioner(PreconditionerType::Jacobi);
    DenseVector xSerial = solverSerial.solve(A, b);

    CGSolver solverParallel(cfgParallel);
    solverParallel.setUseParallel(true);
    solverParallel.setPreconditioner(PreconditionerType::Jacobi);
    DenseVector xParallel = solverParallel.solve(A, b);

    ASSERT_EQ(xSerial.size(), xParallel.size());
    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(xSerial(i), xParallel(i), 1e-8);
    }

    ASSERT_TRUE(solverSerial.getLastConverged());
    ASSERT_TRUE(solverParallel.getLastConverged());
    ASSERT_EQ(solverSerial.getLastUsedThreads(), 1);
    ASSERT_GT(solverParallel.getLastUsedThreads(), 0);
}

TEST_CASE(parallel_solver_csr_format) {
    Index n = 40;
    DenseMatrix dense = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        dense(i, i) = 3.0;
        if (i > 0) dense(i, i - 1) = -0.5;
        if (i < n - 1) dense(i, i + 1) = -0.5;
        if (i > 5) dense(i, i - 6) = -0.1;
        if (i < n - 6) dense(i, i + 6) = -0.1;
    }
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            if (i != j) dense(j, i) = dense(i, j);
        }
    }

    SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(dense);

    DenseVector b = DenseVector::LinSpaced(n, 1.0, static_cast<Scalar>(n));

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 10000;
    config.useParallel = true;

    CGSolver solver(config);
    solver.setPreconditioner(PreconditionerType::Jacobi);
    DenseVector x = solver.solve(csr, b);

    DenseVector residual = dense * x - b;
    ASSERT_LT(residual.norm(), 1e-8);
    ASSERT_TRUE(solver.getLastConverged());
}

TEST_CASE(parallel_solver_eigen_sparse_format) {
    Index n = 60;
    SparseMatrix A(n, n);
    std::vector<Triplet> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 8.0);
        if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
        if (i < n - 1) triplets.emplace_back(i, i + 1, -1.0);
        if (i > 3) triplets.emplace_back(i, i - 4, -0.5);
        if (i < n - 3) triplets.emplace_back(i, i + 4, -0.5);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    DenseVector b = DenseVector::Random(n);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 50000;
    config.useParallel = true;
    CGSolver solver(config);
    solver.setPreconditioner(PreconditionerType::Jacobi);

    SimulationResult result = solver.solveWithStats(A, b);

    ASSERT_TRUE(result.converged);
    ASSERT_GT(result.iterations, 0);
    ASSERT_GT(result.displacements.size(), 0);
    ASSERT_GT(solver.getLastUsedThreads(), 0);
}

TEST_CASE(parallel_solver_solve_explicit_parallel) {
    Index n = 25;
    DenseMatrix A = DenseMatrix::Identity(n, n);
    DenseVector b = DenseVector::Ones(n);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.useParallel = false;
    CGSolver solver(config);
    solver.setUseParallel(false);

    DenseVector x1 = solver.solve(A, b);
    ASSERT_EQ(solver.getLastUsedThreads(), 1);

    DenseVector x2 = solver.solveParallel(A, b);
    ASSERT_GT(solver.getLastUsedThreads(), 0);

    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x1(i), x2(i), 1e-15);
    }
}

TEST_CASE(parallel_solver_thread_count_control) {
    Index n = 100;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 3.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }
    DenseVector b = DenseVector::Ones(n);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 10000;
    config.useParallel = true;

    Index maxThreads = CGSolver::getMaxThreads();
    ASSERT_GT(maxThreads, 0);

    CGSolver solver1(config);
    solver1.setNumThreads(1);
    solver1.solve(A, b);
    ASSERT_EQ(solver1.getLastUsedThreads(), 1);

    CGSolver solver2(config);
    solver2.setNumThreads(std::min(Index(2), maxThreads));
    solver2.solve(A, b);
    ASSERT_EQ(solver2.getLastUsedThreads(), std::min(Index(2), maxThreads));
}

TEST_CASE(parallel_solver_full_mesh_simulation) {
    MeshGenerator generator;
    Mesh mesh = generator.generateStructuredGrid2D(1.0, 1.0, 8, 8,
                                                    2.1e11, 0.3, 0.1);

    StiffnessAssembler assembler;
    SparseMatrix K_sparse = assembler.assembleSparseMatrix(mesh);

    std::vector<BoundaryCondition> bcs;
    Index numNodesX = 9;
    for (Index j = 0; j <= 8; ++j) {
        Index nodeId = j * numNodesX;
        bcs.push_back({nodeId, 0, 0.0});
        bcs.push_back({nodeId, 1, 0.0});
    }
    std::vector<Load> loads;
    Index topRightNode = 8 * numNodesX + 8;
    loads.push_back({topRightNode, 1, -1000.0});
    DenseVector F = assembler.assembleLoadVector(mesh, loads);

    SparseMatrix K_bc = K_sparse;
    DenseVector F_bc = F;
    assembler.applyBoundaryConditions(K_bc, F_bc, bcs);

    SolverConfig config;
    config.tolerance = 1e-10;
    config.maxIterations = 100000;
    config.useParallel = true;

    CGSolver solver(config);
    solver.setPreconditioner(PreconditionerType::Jacobi);

    SimulationResult result = solver.solveWithStats(K_bc, F_bc);

    ASSERT_TRUE(result.converged);
    ASSERT_GT(result.iterations, 0);
    ASSERT_GT(solver.getLastUsedThreads(), 0);
    ASSERT_GT(result.displacements.size(), 0);

    if (result.displacements.size() > 0) {
        Scalar maxDisp = result.displacements.cwiseAbs().maxCoeff();
        ASSERT_GT(maxDisp, 0.0);
    }
}

TEST_CASE(parallel_cgsolver_submodule_direct_use) {
    Index n = 15;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 6.0;
        if (i > 0) A(i, i - 1) = -2.0;
        if (i < n - 1) A(i, i + 1) = -2.0;
    }
    DenseVector b = DenseVector::Constant(n, 3.0);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 1000;
    config.useParallel = true;

    ParallelCGSolver parallelSolver(config);
    parallelSolver.setPreconditioner(PreconditionerType::Jacobi);

    Index maxThreads = ParallelCGSolver::getMaxThreads();
    ASSERT_GT(maxThreads, 0);

    DenseVector x = parallelSolver.solve(A, b);
    DenseVector residual = A * x - b;

    ASSERT_LT(residual.norm(), 1e-8);
    ASSERT_TRUE(parallelSolver.getLastConverged());
    ASSERT_GT(parallelSolver.getLastUsedThreads(), 0);
    ASSERT_GT(parallelSolver.getLastSolveTime(), 0.0);
}

TEST_CASE(parallel_solver_no_parallel_equals_serial) {
    Index n = 20;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (Index i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }
    DenseVector b = DenseVector::Random(n);

    SolverConfig config;
    config.tolerance = 1e-12;
    config.maxIterations = 10000;
    config.useParallel = false;

    CGSolver solver(config);
    solver.setPreconditioner(PreconditionerType::Jacobi);

    DenseVector x1 = solver.solve(A, b);
    Index threads1 = solver.getLastUsedThreads();

    solver.setUseParallel(false);
    DenseVector x2 = solver.solve(A, b);
    Index threads2 = solver.getLastUsedThreads();

    ASSERT_EQ(threads1, 1);
    ASSERT_EQ(threads2, 1);
    for (Index i = 0; i < n; ++i) {
        ASSERT_NEAR(x1(i), x2(i), 1e-15);
    }
}
