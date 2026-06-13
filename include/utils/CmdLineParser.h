#ifndef FEM_UTILS_CMDLINE_PARSER_H
#define FEM_UTILS_CMDLINE_PARSER_H

#include "common/Types.h"
#include <string>
#include <map>
#include <vector>

namespace fem {
namespace utils {

struct CommandLineOptions {
    std::string configFile;
    std::string meshFile;
    std::string outputDir;
    std::string matrixFile;

    Scalar youngsModulus;
    Scalar poissonRatio;
    Scalar thickness;

    Index numElementsX;
    Index numElementsY;
    Scalar meshWidth;
    Scalar meshHeight;

    Scalar solverTolerance;
    Index maxIterations;
    bool verbose;

    bool exportMatrix;
    bool generateMesh;
    bool showHelp;

    CommandLineOptions();
};

class CmdLineParser {
public:
    CmdLineParser();
    ~CmdLineParser();

    CommandLineOptions parse(int argc, char* argv[]) const;
    SimulationConfig toSimulationConfig(const CommandLineOptions& options) const;

    void printHelp() const;
    void printOptions(const CommandLineOptions& options) const;

private:
    std::string getOptionValue(const std::map<std::string, std::string>& options,
                               const std::string& key,
                               const std::string& defaultValue) const;

    bool hasOption(const std::map<std::string, std::string>& options,
                   const std::string& key) const;

    Scalar toScalar(const std::string& value, Scalar defaultValue) const;
    Index toIndex(const std::string& value, Index defaultValue) const;
    bool toBool(const std::string& value, bool defaultValue) const;
};

} // namespace utils
} // namespace fem

#endif // FEM_UTILS_CMDLINE_PARSER_H
