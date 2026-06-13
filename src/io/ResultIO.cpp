#include "io/ResultIO.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <cerrno>

#ifdef _WIN32
#include <direct.h>
#endif

namespace fem {
namespace io {

ResultIO::ResultIO() = default;
ResultIO::~ResultIO() = default;

bool ResultIO::writeDisplacements(const DenseVector& displacements,
                                  const Mesh& mesh,
                                  const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# Node Displacements\n";
    file << "# NodeID   Ux          Uy\n";

    Index dofIdx = 0;
    for (const auto& node : mesh.nodes) {
        Scalar ux = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        Scalar uy = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        file << node.id << " " << ux << " " << uy << "\n";
    }

    return true;
}

bool ResultIO::writeStresses(const DenseVector& stresses,
                             const Mesh& mesh,
                             const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# Element Stresses\n";
    file << "# ElementID   Sigma_x    Sigma_y    Tau_xy\n";

    Index stressPerElem = 3;
    for (size_t i = 0; i < mesh.elements.size(); ++i) {
        Index idx = static_cast<Index>(i) * stressPerElem;
        if (idx + 2 < stresses.size()) {
            file << mesh.elements[i].id << " "
                 << stresses[idx] << " "
                 << stresses[idx + 1] << " "
                 << stresses[idx + 2] << "\n";
        }
    }

    return true;
}

bool ResultIO::writeSimulationSummary(const SimulationResult& result,
                                      const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# Simulation Summary\n";
    file << "# ===================\n\n";

    file << "Solver Status:        "
         << (result.converged ? "Converged" : "Not Converged") << "\n";
    file << "Iterations:           " << result.iterations << "\n";
    file << "Final Residual:       " << result.residual << "\n";
    file << "Solve Time:           " << result.solveTimeSeconds << " seconds\n\n";

    file << "Displacement DOFs:    " << result.displacements.size() << "\n";

    if (result.displacements.size() > 0) {
        Scalar maxDisp = result.displacements.cwiseAbs().maxCoeff();
        file << "Max Displacement:     " << maxDisp << "\n";
    }

    if (result.stresses.size() > 0) {
        Scalar maxStress = result.stresses.cwiseAbs().maxCoeff();
        file << "Max Stress:           " << maxStress << "\n";
    }

    return true;
}

bool ResultIO::writeNodeDisplacementsVTK(const Mesh& mesh,
                                         const DenseVector& displacements,
                                         const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "FEM Displacement Results\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n\n";

    file << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto& node : mesh.nodes) {
        file << node.x << " " << node.y << " " << node.z << "\n";
    }

    file << "\nCELLS " << mesh.elements.size() << " ";
    Index totalSize = 0;
    for (const auto& elem : mesh.elements) {
        totalSize += static_cast<Index>(elem.nodeIds.size()) + 1;
    }
    file << totalSize << "\n";
    for (const auto& elem : mesh.elements) {
        file << elem.nodeIds.size();
        for (Index nodeId : elem.nodeIds) {
            file << " " << nodeId;
        }
        file << "\n";
    }

    file << "\nCELL_TYPES " << mesh.elements.size() << "\n";
    for (size_t i = 0; i < mesh.elements.size(); ++i) {
        file << "9\n";
    }

    file << "\nPOINT_DATA " << mesh.nodes.size() << "\n";
    file << "VECTORS displacements double\n";

    Index dofIdx = 0;
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Scalar ux = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        Scalar uy = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        file << ux << " " << uy << " 0.0\n";
    }

    file << "\nSCALARS displacement_magnitude double 1\n";
    file << "LOOKUP_TABLE default\n";
    dofIdx = 0;
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Scalar ux = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        Scalar uy = (dofIdx < displacements.size()) ? displacements[dofIdx++] : 0.0;
        Scalar mag = std::sqrt(ux * ux + uy * uy);
        file << mag << "\n";
    }

    return true;
}

bool ResultIO::exportAllResults(const Mesh& mesh,
                                const SimulationResult& result,
                                const std::string& outputDir) const {
    if (!ensureDirectory(outputDir)) {
        std::cerr << "Cannot create output directory: " << outputDir << std::endl;
        return false;
    }

    bool ok = true;

    ok &= writeDisplacements(result.displacements, mesh,
                             outputDir + "/displacements.txt");

    if (result.stresses.size() > 0) {
        ok &= writeStresses(result.stresses, mesh,
                            outputDir + "/stresses.txt");
    }

    ok &= writeSimulationSummary(result,
                                 outputDir + "/summary.txt");

    ok &= writeNodeDisplacementsVTK(mesh, result.displacements,
                                    outputDir + "/results.vtk");

    return ok;
}

bool ResultIO::ensureDirectory(const std::string& path) const {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path.c_str(), 0755) == 0;
#endif
}

} // namespace io
} // namespace fem
