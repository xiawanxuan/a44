#include "common/Types.h"
#include "mesh/MeshGenerator.h"
#include "assembly/StiffnessAssembler.h"
#include "sparse/SparseMatrix.h"
#include "solver/CGSolver.h"
#include "io/MeshIO.h"
#include "io/SparseIO.h"
#include "io/ResultIO.h"
#include "utils/CmdLineParser.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace fem;
using namespace fem::mesh;
using namespace fem::assembly;
using namespace fem::sparse;
using namespace fem::solver;
using namespace fem::io;
using namespace fem::utils;

int main(int argc, char* argv[]) {
    CmdLineParser parser;
    CommandLineOptions options = parser.parse(argc, argv);

    if (options.showHelp) {
        parser.printHelp();
        return 0;
    }

    if (options.verbose) {
        parser.printOptions(options);
    }

    std::cout << "=== FEM 稀疏矩阵求解器 ===" << std::endl;
    std::cout << std::endl;

    auto totalStart = std::chrono::high_resolution_clock::now();

    Mesh mesh;
    if (options.generateMesh) {
        std::cout << "[1/6] 生成结构化网格..." << std::endl;
        MeshGenerator generator;
        mesh = generator.generateStructuredGrid2D(
            options.meshWidth,
            options.meshHeight,
            options.numElementsX,
            options.numElementsY,
            options.youngsModulus,
            options.poissonRatio,
            options.thickness
        );
    } else {
        std::cout << "[1/6] 从文件导入网格: " << options.meshFile << std::endl;
        MeshIO meshIO;
        mesh = meshIO.readMesh(options.meshFile);
    }

    std::cout << "      节点数: " << mesh.nodes.size() << std::endl;
    std::cout << "      单元数: " << mesh.elements.size() << std::endl;
    std::cout << "      总自由度: " << mesh.totalDofs << std::endl;
    std::cout << std::endl;

    std::cout << "[2/6] 组装刚度矩阵..." << std::endl;
    auto assemblyStart = std::chrono::high_resolution_clock::now();

    StiffnessAssembler assembler;
    SparseMatrix K_sparse = assembler.assembleSparseMatrix(mesh);

    auto assemblyEnd = std::chrono::high_resolution_clock::now();
    double assemblyTime = std::chrono::duration<double>(assemblyEnd - assemblyStart).count();

    std::cout << "      矩阵维度: " << K_sparse.rows() << " x " << K_sparse.cols() << std::endl;
    std::cout << "      非零元素: " << K_sparse.nonZeros() << std::endl;
    double sparseRatio = 1.0 - static_cast<double>(K_sparse.nonZeros()) /
                              (static_cast<double>(K_sparse.rows()) * K_sparse.cols());
    std::cout << "      稀疏度: " << (sparseRatio * 100.0) << "%" << std::endl;
    std::cout << "      组装耗时: " << assemblyTime << " 秒" << std::endl;
    std::cout << std::endl;

    std::cout << "[3/6] 转换为 CSR 稀疏格式..." << std::endl;
    SparseConverter converter;
    CSRMatrix csrMatrix = converter.eigenSparseToCSR(K_sparse);

    std::cout << "      CSR 内存占用: " << csrMatrix.memoryUsageMB() << " MB" << std::endl;
    double denseMemoryMB = static_cast<double>(csrMatrix.rows) * csrMatrix.cols *
                           sizeof(Scalar) / (1024.0 * 1024.0);
    std::cout << "      稠密矩阵预估: " << denseMemoryMB << " MB" << std::endl;
    std::cout << "      压缩比: " << (denseMemoryMB / csrMatrix.memoryUsageMB()) << "x" << std::endl;
    std::cout << std::endl;

    std::cout << "[4/6] 施加边界条件和荷载..." << std::endl;

    std::vector<BoundaryCondition> bcs;
    Index numNodesX = options.numElementsX + 1;
    for (Index j = 0; j <= options.numElementsY; ++j) {
        Index nodeId = j * numNodesX;
        bcs.push_back({nodeId, 0, 0.0});
        bcs.push_back({nodeId, 1, 0.0});
    }

    std::vector<Load> loads;
    Index topRightNode = (options.numElementsY) * numNodesX + options.numElementsX;
    loads.push_back({topRightNode, 1, -1000.0});

    DenseVector F = assembler.assembleLoadVector(mesh, loads);

    SparseMatrix K_bc = K_sparse;
    DenseVector F_bc = F;
    assembler.applyBoundaryConditions(K_bc, F_bc, bcs);

    std::cout << "      约束数: " << bcs.size() << std::endl;
    std::cout << "      荷载数: " << loads.size() << std::endl;
    std::cout << std::endl;

    std::cout << "[5/6] 共轭梯度求解..." << std::endl;

    SolverConfig solverConfig;
    solverConfig.tolerance = options.solverTolerance;
    solverConfig.maxIterations = options.maxIterations;
    solverConfig.verbose = options.verbose;

    CGSolver solver(solverConfig);
    solver.setPreconditioner(PreconditionerType::Jacobi);

    auto solveStart = std::chrono::high_resolution_clock::now();
    SimulationResult result = solver.solveWithStats(K_bc, F_bc);
    auto solveEnd = std::chrono::high_resolution_clock::now();
    result.solveTimeSeconds = std::chrono::duration<double>(solveEnd - solveStart).count();

    std::cout << "      迭代次数: " << result.iterations << std::endl;
    std::cout << "      最终残差: " << result.residual << std::endl;
    std::cout << "      收敛状态: "
              << (result.converged ? "收敛" : "未收敛") << std::endl;
    std::cout << "      求解耗时: " << result.solveTimeSeconds << " 秒" << std::endl;

    if (result.displacements.size() > 0) {
        Scalar maxDisp = result.displacements.cwiseAbs().maxCoeff();
        std::cout << "      最大位移: " << maxDisp << " m" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "[6/6] 导出结果..." << std::endl;

    ResultIO resultIO;
    if (!resultIO.exportAllResults(mesh, result, options.outputDir)) {
        std::cerr << "      警告: 部分结果导出失败" << std::endl;
    }

    if (options.exportMatrix) {
        SparseIO sparseIO;
        std::string matrixFile = options.matrixFile.empty() ?
            options.outputDir + "/stiffness_matrix.csr" : options.matrixFile;

        if (sparseIO.writeCSRBinary(csrMatrix, matrixFile)) {
            std::cout << "      稀疏矩阵已导出: " << matrixFile << std::endl;
        } else {
            std::cerr << "      错误: 矩阵导出失败" << std::endl;
        }
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cout << "      结果目录: " << options.outputDir << std::endl;
    std::cout << std::endl;

    std::cout << "=== 计算完成 ===" << std::endl;
    std::cout << "总耗时: " << totalTime << " 秒" << std::endl;

    return result.converged ? 0 : 1;
}
