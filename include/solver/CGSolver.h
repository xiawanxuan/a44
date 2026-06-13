#ifndef FEM_SOLVER_CG_SOLVER_H
#define FEM_SOLVER_CG_SOLVER_H

#include "common/Types.h"
#include "sparse/SparseMatrix.h"

namespace fem {
namespace solver {

enum class PreconditionerType {
    None,
    Jacobi,
    SSOR
};

class CGSolver {
public:
    CGSolver();
    explicit CGSolver(const SolverConfig& config);
    ~CGSolver();

    void setConfig(const SolverConfig& config);
    void setPreconditioner(PreconditionerType type, Scalar omega = 1.0);

    DenseVector solve(const SparseMatrix& A, const DenseVector& b) const;
    DenseVector solve(const CSRMatrix& A, const DenseVector& b) const;
    DenseVector solve(const DenseMatrix& A, const DenseVector& b) const;

    SimulationResult solveWithStats(const SparseMatrix& A,
                                    const DenseVector& b) const;

    Index getLastIterations() const { return lastIterations_; }
    Scalar getLastResidual() const { return lastResidual_; }
    bool getLastConverged() const { return lastConverged_; }

private:
    DenseVector solveCG(const CSRMatrix& A, const DenseVector& b) const;
    DenseVector solveCGParallel(const CSRMatrix& A, const DenseVector& b) const;

    void applyPreconditioner(const CSRMatrix& A, const DenseVector& r,
                             DenseVector& z) const;

    void computeJacobiPreconditioner(const CSRMatrix& A) const;

    static Scalar dotProduct(const DenseVector& a, const DenseVector& b);
    static void sparseMultiply(const CSRMatrix& A, const DenseVector& x,
                               DenseVector& result);

    SolverConfig config_;
    PreconditionerType preconditionerType_;
    Scalar preconditionerOmega_;

    mutable std::vector<Scalar> jacobiDiag_;
    mutable bool jacobiComputed_;

    mutable Index lastIterations_;
    mutable Scalar lastResidual_;
    mutable bool lastConverged_;
};

} // namespace solver
} // namespace fem

#endif // FEM_SOLVER_CG_SOLVER_H
