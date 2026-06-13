#include "solver/ParallelCGSolver.h"
#include "sparse/SparseMatrix.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace fem {
namespace solver {

ParallelCGSolver::ParallelCGSolver()
    : preconditionerType_(PreconditionerType::None),
      preconditionerOmega_(1.0),
      numThreads_(0),
      schedule_(ParallelSchedule::Auto),
      chunkSize_(0),
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false),
      lastSolveTime_(0),
      lastUsedThreads_(1) {}

ParallelCGSolver::ParallelCGSolver(const SolverConfig& config)
    : config_(config),
      preconditionerType_(PreconditionerType::None),
      preconditionerOmega_(1.0),
      numThreads_(config.numThreads > 0 ? config.numThreads : 0),
      schedule_(ParallelSchedule::Auto),
      chunkSize_(0),
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false),
      lastSolveTime_(0),
      lastUsedThreads_(1) {}

ParallelCGSolver::~ParallelCGSolver() = default;

void ParallelCGSolver::setConfig(const SolverConfig& config) {
    config_ = config;
    if (config.numThreads > 0) {
        numThreads_ = config.numThreads;
    }
    jacobiComputed_ = false;
}

void ParallelCGSolver::setPreconditioner(PreconditionerType type, Scalar omega) {
    preconditionerType_ = type;
    preconditionerOmega_ = omega;
    jacobiComputed_ = false;
}

void ParallelCGSolver::setNumThreads(Index numThreads) {
    numThreads_ = numThreads;
}

void ParallelCGSolver::setSchedule(ParallelSchedule schedule, Index chunkSize) {
    schedule_ = schedule;
    chunkSize_ = chunkSize;
}

Index ParallelCGSolver::getMaxThreads() {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

Index ParallelCGSolver::getCurrentThreads() {
#ifdef USE_OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

DenseVector ParallelCGSolver::solve(const SparseMatrix& A,
                                    const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.eigenSparseToCSR(A);
    return solveCGParallel(csr, b);
}

DenseVector ParallelCGSolver::solve(const sparse::CSRMatrix& A,
                                    const DenseVector& b) const {
    return solveCGParallel(A, b);
}

DenseVector ParallelCGSolver::solve(const DenseMatrix& A,
                                    const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.denseToCSR(A);
    return solveCGParallel(csr, b);
}

SimulationResult ParallelCGSolver::solveWithStats(
    const SparseMatrix& A, const DenseVector& b) const {

    auto startTime = std::chrono::high_resolution_clock::now();

    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.eigenSparseToCSR(A);
    DenseVector x = solveCGParallel(csr, b);

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    SimulationResult result;
    result.displacements = x;
    result.iterations = lastIterations_;
    result.residual = lastResidual_;
    result.converged = lastConverged_;
    result.solveTimeSeconds = elapsed;
    lastSolveTime_ = elapsed;

    return result;
}

DenseVector ParallelCGSolver::solveCGParallel(
    const sparse::CSRMatrix& A, const DenseVector& b) const {

    if (A.rows != A.cols) {
        throw std::invalid_argument("ParallelCGSolver: Matrix must be square");
    }
    if (A.rows != b.size()) {
        throw std::invalid_argument("ParallelCGSolver: Matrix/vector size mismatch");
    }

    Index effectiveThreads = numThreads_;
#ifdef USE_OPENMP
    if (effectiveThreads <= 0) {
        effectiveThreads = omp_get_max_threads();
    }
    omp_set_num_threads(static_cast<int>(effectiveThreads));
    if (config_.verbose) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                std::cout << "Parallel CG: Using "
                          << omp_get_num_threads() << " threads" << std::endl;
                lastUsedThreads_ = omp_get_num_threads();
            }
        }
    } else {
        lastUsedThreads_ = effectiveThreads;
    }
#else
    lastUsedThreads_ = 1;
#endif

    const Index n = A.rows;
    DenseVector x = DenseVector::Zero(n);
    DenseVector r = b;
    DenseVector p(n);
    DenseVector Ap(n);
    DenseVector z(n);

    if (preconditionerType_ != PreconditionerType::None) {
        computeJacobiPreconditioner(A);
        applyPreconditioner(A, r, z);
        p = z;
    } else {
        p = r;
    }

    Scalar rsOld = dotProductParallel(r, r);
    const Scalar initialResidual = std::sqrt(rsOld);
    const Scalar targetResidual = config_.tolerance * initialResidual;

    if (initialResidual < 1e-30) {
        lastIterations_ = 0;
        lastResidual_ = 0.0;
        lastConverged_ = true;
        return x;
    }

    Index k;
    bool converged = false;

    for (k = 0; k < config_.maxIterations; ++k) {
        sparseMultiplyParallel(A, p, Ap);

        const Scalar pAp = dotProductParallel(p, Ap);
        if (std::abs(pAp) < 1e-30) {
            if (config_.verbose) {
                std::cout << "Parallel CG: Breakdown at iteration " << k
                          << " (pAp ~ 0)" << std::endl;
            }
            break;
        }

        const Scalar alpha = rsOld / pAp;

        axpyParallel(alpha, p, x);
        axpyParallel(-alpha, Ap, r);

        const Scalar rsNew = dotProductParallel(r, r);
        const Scalar currentResidual = std::sqrt(rsNew);

        if (config_.verbose && k % 100 == 0) {
            std::cout << "Parallel Iter " << k
                      << ", residual: " << currentResidual
                      << ", rel: " << currentResidual / initialResidual
                      << std::endl;
        }

        if (currentResidual < targetResidual) {
            converged = true;
            break;
        }

        if (preconditionerType_ != PreconditionerType::None) {
            applyPreconditioner(A, r, z);
            const Scalar rz = dotProductParallel(r, z);
            const Scalar beta = rz / rsOld;
            axpyParallel(beta, p, z);
            p = z;
            rsOld = rz;
        } else {
            const Scalar beta = rsNew / rsOld;
            DenseVector r_copy = r;
            axpyParallel(beta, p, r_copy);
            p = r_copy;
            rsOld = rsNew;
        }
    }

    lastIterations_ = k + 1;
    lastResidual_ = std::sqrt(dotProductParallel(r, r));
    lastConverged_ = converged;

    if (config_.verbose) {
        std::cout << "Parallel CG finished: " << lastIterations_
                  << " iterations, residual: " << lastResidual_
                  << (converged ? " (converged)" : " (not converged)")
                  << ", threads: " << lastUsedThreads_
                  << std::endl;
    }

    return x;
}

void ParallelCGSolver::applyPreconditioner(
    const sparse::CSRMatrix& A, const DenseVector& r, DenseVector& z) const {

    switch (preconditionerType_) {
        case PreconditionerType::None:
            z = r;
            break;
        case PreconditionerType::Jacobi:
            if (!jacobiComputed_) {
                computeJacobiPreconditioner(A);
            }
            #ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (Index i = 0; i < A.rows; ++i) {
                if (std::abs(jacobiDiag_[i]) > 1e-30) {
                    z(i) = r(i) / jacobiDiag_[i];
                } else {
                    z(i) = r(i);
                }
            }
            break;
        case PreconditionerType::SSOR:
            z = r;
            break;
    }
}

void ParallelCGSolver::computeJacobiPreconditioner(
    const sparse::CSRMatrix& A) const {

    jacobiDiag_.resize(A.rows);

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (Index i = 0; i < A.rows; ++i) {
        Scalar diag = 0.0;
        for (Index j = A.rowPointers[i]; j < A.rowPointers[i + 1]; ++j) {
            if (A.colIndices[j] == i) {
                diag = A.values[j];
                break;
            }
        }
        jacobiDiag_[i] = diag;
    }

    jacobiComputed_ = true;
}

Scalar ParallelCGSolver::dotProductParallel(
    const DenseVector& a, const DenseVector& b) {

    Scalar result = 0.0;
    const Index n = a.size();

    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:result) schedule(static)
    #endif
    for (Index i = 0; i < n; ++i) {
        result += a(i) * b(i);
    }

    return result;
}

void ParallelCGSolver::sparseMultiplyParallel(
    const sparse::CSRMatrix& A, const DenseVector& x, DenseVector& result) {

    const Index n = A.rows;
    result.setZero(n);

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 128)
    #endif
    for (Index i = 0; i < n; ++i) {
        Scalar sum = 0.0;
        for (Index j = A.rowPointers[i]; j < A.rowPointers[i + 1]; ++j) {
            sum += A.values[j] * x(A.colIndices[j]);
        }
        result(i) = sum;
    }
}

void ParallelCGSolver::axpyParallel(
    Scalar alpha, const DenseVector& x, DenseVector& y) {

    const Index n = x.size();

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index i = 0; i < n; ++i) {
        y(i) += alpha * x(i);
    }
}

void ParallelCGSolver::scaleParallel(DenseVector& x, Scalar alpha) {
    const Index n = x.size();

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index i = 0; i < n; ++i) {
        x(i) *= alpha;
    }
}

Scalar ParallelCGSolver::norm2Parallel(const DenseVector& x) {
    return std::sqrt(dotProductParallel(x, x));
}

} // namespace solver
} // namespace fem
