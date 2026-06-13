#include "solver/CGSolver.h"
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

CGSolver::CGSolver()
    : preconditionerType_(PreconditionerType::None),
      preconditionerOmega_(1.0),
      numThreads_(0),
      useParallel_(true),
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false),
      lastUsedThreads_(1) {}

CGSolver::CGSolver(const SolverConfig& config)
    : config_(config),
      preconditionerType_(PreconditionerType::None),
      preconditionerOmega_(1.0),
      numThreads_(config.numThreads > 0 ? config.numThreads : 0),
      useParallel_(config.useParallel),
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false),
      lastUsedThreads_(1) {}

CGSolver::~CGSolver() = default;

void CGSolver::setConfig(const SolverConfig& config) {
    config_ = config;
    if (config.numThreads > 0) {
        numThreads_ = config.numThreads;
    }
    useParallel_ = config.useParallel;
    jacobiComputed_ = false;
}

void CGSolver::setPreconditioner(PreconditionerType type, Scalar omega) {
    preconditionerType_ = type;
    preconditionerOmega_ = omega;
    jacobiComputed_ = false;
}

void CGSolver::setNumThreads(Index numThreads) {
    numThreads_ = numThreads;
    jacobiComputed_ = false;
}

void CGSolver::setUseParallel(bool useParallel) {
    useParallel_ = useParallel;
    jacobiComputed_ = false;
}

Index CGSolver::getMaxThreads() {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

DenseVector CGSolver::solve(const SparseMatrix& A, const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.eigenSparseToCSR(A);
    if (useParallel_) {
        return solveCGParallel(csr, b);
    } else {
        return solveCG(csr, b);
    }
}

DenseVector CGSolver::solve(const sparse::CSRMatrix& A,
                            const DenseVector& b) const {
    if (useParallel_) {
        return solveCGParallel(A, b);
    } else {
        return solveCG(A, b);
    }
}

DenseVector CGSolver::solve(const DenseMatrix& A, const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.denseToCSR(A);
    if (useParallel_) {
        return solveCGParallel(csr, b);
    } else {
        return solveCG(csr, b);
    }
}

DenseVector CGSolver::solveParallel(const SparseMatrix& A,
                                    const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.eigenSparseToCSR(A);
    return solveCGParallel(csr, b);
}

DenseVector CGSolver::solveParallel(const sparse::CSRMatrix& A,
                                    const DenseVector& b) const {
    return solveCGParallel(A, b);
}

DenseVector CGSolver::solveParallel(const DenseMatrix& A,
                                    const DenseVector& b) const {
    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.denseToCSR(A);
    return solveCGParallel(csr, b);
}

SimulationResult CGSolver::solveWithStats(const SparseMatrix& A,
                                          const DenseVector& b) const {
    auto startTime = std::chrono::high_resolution_clock::now();

    sparse::SparseConverter converter;
    sparse::CSRMatrix csr = converter.eigenSparseToCSR(A);
    DenseVector x = useParallel_ ? solveCGParallel(csr, b) : solveCG(csr, b);

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    SimulationResult result;
    result.displacements = x;
    result.iterations = lastIterations_;
    result.residual = lastResidual_;
    result.converged = lastConverged_;
    result.solveTimeSeconds = elapsed;

    return result;
}

SimulationResult CGSolver::solveWithStatsParallel(const SparseMatrix& A,
                                                   const DenseVector& b) const {
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

    return result;
}

DenseVector CGSolver::solveCG(const sparse::CSRMatrix& A,
                              const DenseVector& b) const {
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix must be square");
    }
    if (A.rows != b.size()) {
        throw std::invalid_argument("Matrix and vector size mismatch");
    }

    lastUsedThreads_ = 1;

    Index n = A.rows;
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

    Scalar rsOld = dotProduct(r, r);
    Scalar initialResidual = std::sqrt(rsOld);
    Scalar targetResidual = config_.tolerance * initialResidual;

    if (initialResidual < 1e-30) {
        lastIterations_ = 0;
        lastResidual_ = 0.0;
        lastConverged_ = true;
        return x;
    }

    Index k;
    bool converged = false;

    for (k = 0; k < config_.maxIterations; ++k) {
        sparseMultiply(A, p, Ap);

        Scalar pAp = dotProduct(p, Ap);
        if (std::abs(pAp) < 1e-30) {
            break;
        }

        Scalar alpha = rsOld / pAp;
        x += alpha * p;
        r -= alpha * Ap;

        Scalar rsNew = dotProduct(r, r);
        Scalar currentResidual = std::sqrt(rsNew);

        if (config_.verbose && k % 100 == 0) {
            std::cout << "Iteration " << k
                      << ", residual: " << currentResidual
                      << ", relative: " << currentResidual / initialResidual
                      << std::endl;
        }

        if (currentResidual < targetResidual) {
            converged = true;
            break;
        }

        if (preconditionerType_ != PreconditionerType::None) {
            applyPreconditioner(A, r, z);
            Scalar rz = dotProduct(r, z);
            Scalar beta = rz / rsOld;
            p = z + beta * p;
            rsOld = rz;
        } else {
            Scalar beta = rsNew / rsOld;
            p = r + beta * p;
            rsOld = rsNew;
        }
    }

    lastIterations_ = k + 1;
    lastResidual_ = std::sqrt(dotProduct(r, r));
    lastConverged_ = converged;

    if (config_.verbose) {
        std::cout << "CG (serial) finished: " << lastIterations_
                  << " iterations, residual: " << lastResidual_
                  << (converged ? " (converged)" : " (not converged)")
                  << std::endl;
    }

    return x;
}

DenseVector CGSolver::solveCGParallel(const sparse::CSRMatrix& A,
                                      const DenseVector& b) const {
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
        computeJacobiPreconditionerParallel(A);
        applyPreconditionerParallel(A, r, z);
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

        x += alpha * p;
        r -= alpha * Ap;

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
            applyPreconditionerParallel(A, r, z);
            const Scalar rz = dotProductParallel(r, z);
            const Scalar beta = rz / rsOld;

            DenseVector beta_p = beta * p;
            axpyParallel(1.0, beta_p, z);
            p = z;
            rsOld = rz;
        } else {
            const Scalar beta = rsNew / rsOld;
            DenseVector beta_p = beta * p;
            axpyParallel(1.0, beta_p, r);
            p = r;
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

void CGSolver::applyPreconditioner(const sparse::CSRMatrix& A,
                                   const DenseVector& r,
                                   DenseVector& z) const {
    switch (preconditionerType_) {
        case PreconditionerType::None:
            z = r;
            break;
        case PreconditionerType::Jacobi:
            if (!jacobiComputed_) {
                computeJacobiPreconditioner(A);
            }
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

void CGSolver::applyPreconditionerParallel(const sparse::CSRMatrix& A,
                                           const DenseVector& r,
                                           DenseVector& z) const {
    switch (preconditionerType_) {
        case PreconditionerType::None:
            z = r;
            break;
        case PreconditionerType::Jacobi:
            if (!jacobiComputed_) {
                computeJacobiPreconditionerParallel(A);
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

void CGSolver::computeJacobiPreconditioner(const sparse::CSRMatrix& A) const {
    jacobiDiag_.resize(A.rows);

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

void CGSolver::computeJacobiPreconditionerParallel(
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

Scalar CGSolver::dotProduct(const DenseVector& a, const DenseVector& b) {
    Scalar result = 0.0;
    const Index n = a.size();
    for (Index i = 0; i < n; ++i) {
        result += a(i) * b(i);
    }
    return result;
}

Scalar CGSolver::dotProductParallel(const DenseVector& a,
                                    const DenseVector& b) {
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

void CGSolver::sparseMultiply(const sparse::CSRMatrix& A,
                              const DenseVector& x,
                              DenseVector& result) {
    const Index n = A.rows;
    result.setZero(n);
    for (Index i = 0; i < n; ++i) {
        Scalar sum = 0.0;
        for (Index j = A.rowPointers[i]; j < A.rowPointers[i + 1]; ++j) {
            sum += A.values[j] * x(A.colIndices[j]);
        }
        result(i) = sum;
    }
}

void CGSolver::sparseMultiplyParallel(const sparse::CSRMatrix& A,
                                      const DenseVector& x,
                                      DenseVector& result) {
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

void CGSolver::axpyParallel(Scalar alpha, const DenseVector& x,
                            DenseVector& y) {
    const Index n = x.size();

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (Index i = 0; i < n; ++i) {
        y(i) += alpha * x(i);
    }
}

} // namespace solver
} // namespace fem
