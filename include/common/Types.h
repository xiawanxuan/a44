#ifndef FEM_COMMON_TYPES_H
#define FEM_COMMON_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <cstdint>

namespace fem {

using Scalar = double;
using Index = Eigen::Index;
using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<Scalar>;

struct Node {
    Index id;
    Scalar x;
    Scalar y;
    Scalar z;

    Node() : id(0), x(0), y(0), z(0) {}
    Node(Index id_, Scalar x_, Scalar y_, Scalar z_ = 0.0)
        : id(id_), x(x_), y(y_), z(z_) {}
};

struct Element {
    Index id;
    std::vector<Index> nodeIds;
    Scalar youngsModulus;
    Scalar poissonRatio;
    Scalar thickness;

    Element() : id(0), youngsModulus(0), poissonRatio(0), thickness(0) {}
};

struct Mesh {
    std::vector<Node> nodes;
    std::vector<Element> elements;
    Index totalDofs;
    Index dofsPerNode;

    Mesh() : totalDofs(0), dofsPerNode(2) {}
};

struct BoundaryCondition {
    Index nodeId;
    Index dof;
    Scalar value;
};

struct Load {
    Index nodeId;
    Index dof;
    Scalar value;
};

struct SolverConfig {
    Scalar tolerance;
    Index maxIterations;
    bool verbose;
    Index numThreads;
    bool useParallel;

    SolverConfig()
        : tolerance(1e-8), maxIterations(10000), verbose(false),
          numThreads(0), useParallel(true) {}
};

struct SimulationConfig {
    std::string meshFile;
    std::string outputFile;
    std::string matrixFile;
    Scalar youngsModulus;
    Scalar poissonRatio;
    Scalar thickness;
    SolverConfig solver;
    bool exportSparseMatrix;

    SimulationConfig()
        : youngsModulus(2.1e11), poissonRatio(0.3), thickness(0.1),
          exportSparseMatrix(false) {}
};

struct SimulationResult {
    DenseVector displacements;
    DenseVector stresses;
    DenseVector reactions;
    Index iterations;
    Scalar residual;
    bool converged;
    double solveTimeSeconds;

    SimulationResult()
        : iterations(0), residual(0), converged(false), solveTimeSeconds(0) {}
};

} // namespace fem

#endif // FEM_COMMON_TYPES_H
