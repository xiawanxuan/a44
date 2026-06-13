#include "solver/CGSolver.h"
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
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false) {}

CGSolver::CGSolver(const SolverConfig& config)
    : config_(config),
      preconditionerType_(PreconditionerType::None),
      preconditionerOmega_(1.0),
      jacobiComputed_(false),
      lastIterations_(0),
      lastResidual_(0),
      lastConverged_(false) {}

CGSolver::~CGSolver() = default;

void CGSolver::setConfig(const SolverConfig& config) {
    config_ = config;
}

void CGSolver::setPreconditioner(PreconditionerType type, Scalar omega) {
    preconditionerType_ = type;
    preconditionerOmega_ = omega;
    jacobiComputed_ = false;
}

DenseVector CGSolver::solve(const SparseMatrix& A, const DenseVector& b) const {
    sparse::SparseConverter converter;
    CSRMatrix csr = converter.eigenSparseToCSR(A);
    return solveCG(csr, b);
}

DenseVector CGSolver::solve(const CSRMatrix& A, const DenseVector& b) const {
    return solveCG(A, b);
}

DenseVector CGSolver::solve(const DenseMatrix& A, const DenseVector& b) const {
    sparse::SparseConverter converter;
    CSRMatrix csr = converter.denseToCSR(A);
    return solveCG(csr, b);
}

SimulationResult CGSolver::solveWithStats(const SparseMatrix& A,
                                          const DenseVector& b) const {
    auto startTime = std::chrono::high_resolution_clock::now();

    sparse::SparseConverter converter;
    CSRMatrix csr = converter.eigenSparseToCSR(A);
    DenseVector x = solveCG(csr, b);

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

DenseVector CGSolver::solveCG(const CSRMatrix& A, const DenseVector& b) const {
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix must be square");
    }
    if (A.rows != b.size()) {
        throw std::invalid_argument("Matrix and vector size mismatch");
    }

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
        std::cout << "CG finished: " << lastIterations_ << " iterations, "
                  << "residual: " << lastResidual_
                  << (converged ? " (converged)" : " (not converged)")
                  << std::endl;
    }

    return x;
}

DenseVector CGSolver::solveCGParallel(const CSRMatrix& A,
                                      const DenseVector& b) const {
    return solveCG(A, b);
}

void CGSolver::applyPreconditioner(const CSRMatrix& A, const DenseVector& r,
                                   DenseVector& z) const {
    switch (preconditionerType_) {
        case PreconditionerType::None:
            z = r;
            break;
        case PreconditionerType::Jacobi:
            if (!jacobiComputed_) {
                computeJacobiPreconditioner(A);
            }
            #ifdef USE_OPENMP
            #pragma omp parallel for
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

void CGSolver::computeJacobiPreconditioner(const CSRMatrix& A) const {
    jacobiDiag_.resize(A.rows);

    #ifdef USE_OPENMP
    #pragma omp parallel for
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
    Index n = a.size();

    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:result)
    #endif
    for (Index i = 0; i < n; ++i) {
        result += a(i) * b(i);
    }

    return result;
}

void CGSolver::sparseMultiply(const CSRMatrix& A, const DenseVector& x,
                              DenseVector& result) {
    Index n = A.rows;
    result.setZero(n);

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (Index i = 0; i < n; ++i) {
        Scalar sum = 0.0;
        for (Index j = A.rowPointers[i]; j < A.rowPointers[i + 1]; ++j) {
            sum += A.values[j] * x(A.colIndices[j]);
        }
        result(i) = sum;
    }
}

} // namespace solver
} // namespace fem
