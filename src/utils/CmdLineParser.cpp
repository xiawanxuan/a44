#include "utils/CmdLineParser.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace fem {
namespace utils {

CommandLineOptions::CommandLineOptions()
    : youngsModulus(2.1e11),
      poissonRatio(0.3),
      thickness(0.1),
      numElementsX(10),
      numElementsY(10),
      meshWidth(1.0),
      meshHeight(1.0),
      solverTolerance(1e-8),
      maxIterations(10000),
      verbose(false),
      exportMatrix(false),
      generateMesh(true),
      showHelp(false) {}

CmdLineParser::CmdLineParser() = default;
CmdLineParser::~CmdLineParser() = default;

CommandLineOptions CmdLineParser::parse(int argc, char* argv[]) const {
    CommandLineOptions options;
    std::map<std::string, std::string> optionMap;
    std::vector<std::string> positionalArgs;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            options.showHelp = true;
            return options;
        } else if (arg == "-v" || arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--export-matrix") {
            options.exportMatrix = true;
        } else if (arg.substr(0, 2) == "--") {
            size_t eqPos = arg.find('=');
            if (eqPos != std::string::npos) {
                std::string key = arg.substr(2, eqPos - 2);
                std::string value = arg.substr(eqPos + 1);
                optionMap[key] = value;
            } else if (i + 1 < argc) {
                std::string key = arg.substr(2);
                optionMap[key] = argv[++i];
            }
        } else if (arg[0] == '-' && arg.size() > 1) {
            if (i + 1 < argc) {
                optionMap[arg.substr(1)] = argv[++i];
            }
        } else {
            positionalArgs.push_back(arg);
        }
    }

    if (hasOption(optionMap, "config") || hasOption(optionMap, "c")) {
        options.configFile = getOptionValue(optionMap, "config",
                                            getOptionValue(optionMap, "c", ""));
    }

    if (hasOption(optionMap, "mesh") || hasOption(optionMap, "m")) {
        options.meshFile = getOptionValue(optionMap, "mesh",
                                          getOptionValue(optionMap, "m", ""));
        options.generateMesh = options.meshFile.empty();
    }

    if (hasOption(optionMap, "output") || hasOption(optionMap, "o")) {
        options.outputDir = getOptionValue(optionMap, "output",
                                           getOptionValue(optionMap, "o", "results"));
    }

    if (hasOption(optionMap, "matrix-file")) {
        options.matrixFile = getOptionValue(optionMap, "matrix-file", "");
    }

    options.youngsModulus = toScalar(
        getOptionValue(optionMap, "E", ""), options.youngsModulus);
    options.poissonRatio = toScalar(
        getOptionValue(optionMap, "nu", ""), options.poissonRatio);
    options.thickness = toScalar(
        getOptionValue(optionMap, "t", ""), options.thickness);

    options.numElementsX = toIndex(
        getOptionValue(optionMap, "nx", ""), options.numElementsX);
    options.numElementsY = toIndex(
        getOptionValue(optionMap, "ny", ""), options.numElementsY);
    options.meshWidth = toScalar(
        getOptionValue(optionMap, "width", ""), options.meshWidth);
    options.meshHeight = toScalar(
        getOptionValue(optionMap, "height", ""), options.meshHeight);

    options.solverTolerance = toScalar(
        getOptionValue(optionMap, "tol", ""), options.solverTolerance);
    options.maxIterations = toIndex(
        getOptionValue(optionMap, "max-iter", ""), options.maxIterations);

    options.verbose = toBool(
        getOptionValue(optionMap, "verbose", ""), options.verbose);

    return options;
}

SimulationConfig CmdLineParser::toSimulationConfig(
    const CommandLineOptions& options) const {

    SimulationConfig config;
    config.meshFile = options.meshFile;
    config.outputFile = options.outputDir;
    config.matrixFile = options.matrixFile;
    config.youngsModulus = options.youngsModulus;
    config.poissonRatio = options.poissonRatio;
    config.thickness = options.thickness;
    config.exportSparseMatrix = options.exportMatrix;
    config.solver.tolerance = options.solverTolerance;
    config.solver.maxIterations = options.maxIterations;
    config.solver.verbose = options.verbose;
    return config;
}

void CmdLineParser::printHelp() const {
    std::cout << "\n";
    std::cout << "FEM Sparse Solver - 有限元稀疏矩阵求解器\n";
    std::cout << "=========================================\n\n";
    std::cout << "用法:\n";
    std::cout << "  fem_solver [选项]\n\n";
    std::cout << "选项:\n";
    std::cout << "  -h, --help              显示帮助信息\n";
    std::cout << "  -v, --verbose           启用详细输出\n";
    std::cout << "  -c, --config <file>     指定配置文件\n";
    std::cout << "  -m, --mesh <file>       从文件导入网格\n";
    std::cout << "  -o, --output <dir>      输出目录 (默认: results)\n";
    std::cout << "  --export-matrix         导出稀疏矩阵二进制文件\n";
    std::cout << "  --matrix-file <file>    矩阵输出文件路径\n\n";
    std::cout << "网格生成参数:\n";
    std::cout << "  --nx <N>                X方向单元数 (默认: 10)\n";
    std::cout << "  --ny <N>                Y方向单元数 (默认: 10)\n";
    std::cout << "  --width <val>           网格宽度 (默认: 1.0)\n";
    std::cout << "  --height <val>          网格高度 (默认: 1.0)\n";
    std::cout << "  -E <val>                弹性模量 (默认: 2.1e11)\n";
    std::cout << "  --nu <val>              泊松比 (默认: 0.3)\n";
    std::cout << "  -t <val>                厚度 (默认: 0.1)\n\n";
    std::cout << "求解器参数:\n";
    std::cout << "  --tol <val>             收敛容差 (默认: 1e-8)\n";
    std::cout << "  --max-iter <N>          最大迭代次数 (默认: 10000)\n\n";
    std::cout << "示例:\n";
    std::cout << "  fem_solver --nx 20 --ny 20 -E 2.1e11 -v\n";
    std::cout << "  fem_solver -m mesh.txt -o results --export-matrix\n\n";
}

void CmdLineParser::printOptions(const CommandLineOptions& options) const {
    std::cout << "\n配置参数:\n";
    std::cout << "  网格模式:    "
              << (options.generateMesh ? "自动生成" : "从文件导入")
              << "\n";
    if (options.generateMesh) {
        std::cout << "  单元数 (nx*ny): " << options.numElementsX
                  << " x " << options.numElementsY << "\n";
        std::cout << "  尺寸 (宽x高):   " << options.meshWidth
                  << " x " << options.meshHeight << "\n";
    } else {
        std::cout << "  网格文件:    " << options.meshFile << "\n";
    }
    std::cout << "  弹性模量:    " << options.youngsModulus << "\n";
    std::cout << "  泊松比:      " << options.poissonRatio << "\n";
    std::cout << "  厚度:        " << options.thickness << "\n";
    std::cout << "  求解容差:    " << options.solverTolerance << "\n";
    std::cout << "  最大迭代:    " << options.maxIterations << "\n";
    std::cout << "  输出目录:    " << options.outputDir << "\n";
    std::cout << "  导出矩阵:    "
              << (options.exportMatrix ? "是" : "否") << "\n\n";
}

std::string CmdLineParser::getOptionValue(
    const std::map<std::string, std::string>& options,
    const std::string& key,
    const std::string& defaultValue) const {

    auto it = options.find(key);
    if (it != options.end()) {
        return it->second;
    }
    return defaultValue;
}

bool CmdLineParser::hasOption(const std::map<std::string, std::string>& options,
                              const std::string& key) const {
    return options.find(key) != options.end();
}

Scalar CmdLineParser::toScalar(const std::string& value,
                               Scalar defaultValue) const {
    if (value.empty()) return defaultValue;
    try {
        return std::stod(value);
    } catch (...) {
        return defaultValue;
    }
}

Index CmdLineParser::toIndex(const std::string& value,
                             Index defaultValue) const {
    if (value.empty()) return defaultValue;
    try {
        return static_cast<Index>(std::stoi(value));
    } catch (...) {
        return defaultValue;
    }
}

bool CmdLineParser::toBool(const std::string& value,
                           bool defaultValue) const {
    if (value.empty()) return defaultValue;
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
}

} // namespace utils
} // namespace fem
