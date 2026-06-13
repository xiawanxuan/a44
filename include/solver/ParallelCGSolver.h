#ifndef FEM_SOLVER_PARALLEL_CG_SOLVER_H
#define FEM_SOLVER_PARALLEL_CG_SOLVER_H

#include "common/Types.h"
#include "sparse/SparseMatrix.h"

namespace fem {
namespace solver {

enum class ParallelSchedule {
    Static,
    Dynamic,
    Guided,
    Auto
};

class ParallelCGSolver {
public:
    ParallelCGSolver();
    explicit ParallelCGSolver(const SolverConfig& config);
    ~ParallelCGSolver();

    void setConfig(const SolverConfig& config);
    void setPreconditioner(PreconditionerType type, Scalar omega = 1.0);
    void setNumThreads(Index numThreads);
    void setSchedule(ParallelSchedule schedule, Index chunkSize = 0);

    static Index getMaxThreads();
    static Index getCurrentThreads();
    Index getNumThreads() const { return numThreads_; }

    DenseVector solve(const SparseMatrix& A, const DenseVector& b) const;
    DenseVector solve(const sparse::CSRMatrix& A, const DenseVector& b) const;
    DenseVector solve(const DenseMatrix& A, const DenseVector& b) const;

    SimulationResult solveWithStats(const SparseMatrix& A,
                                    const DenseVector& b) const;

    Index getLastIterations() const { return lastIterations_; }
    Scalar getLastResidual() const { return lastResidual_; }
    bool getLastConverged() const { return lastConverged_; }
    double getLastSolveTime() const { return lastSolveTime_; }
    Index getLastUsedThreads() const { return lastUsedThreads_; }

private:
    DenseVector solveCGParallel(const sparse::CSRMatrix& A,
                                const DenseVector& b) const;

    void applyPreconditioner(const sparse::CSRMatrix& A, const DenseVector& r,
                             DenseVector& z) const;

    void computeJacobiPreconditioner(const sparse::CSRMatrix& A) const;

    static Scalar dotProductParallel(const DenseVector& a, const DenseVector& b);
    static void sparseMultiplyParallel(const sparse::CSRMatrix& A,
                                       const DenseVector& x,
                                       DenseVector& result);
    static void axpyParallel(Scalar alpha, const DenseVector& x,
                             DenseVector& y);
    static void scaleParallel(DenseVector& x, Scalar alpha);
    static Scalar norm2Parallel(const DenseVector& x);

    SolverConfig config_;
    PreconditionerType preconditionerType_;
    Scalar preconditionerOmega_;
    Index numThreads_;
    ParallelSchedule schedule_;
    Index chunkSize_;

    mutable std::vector<Scalar> jacobiDiag_;
    mutable bool jacobiComputed_;

    mutable Index lastIterations_;
    mutable Scalar lastResidual_;
    mutable bool lastConverged_;
    mutable double lastSolveTime_;
    mutable Index lastUsedThreads_;
};

} // namespace solver
} // namespace fem

#endif // FEM_SOLVER_PARALLEL_CG_SOLVER_H
