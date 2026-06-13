#include "io/MeshIO.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace fem {
namespace io {

MeshIO::MeshIO() = default;
MeshIO::~MeshIO() = default;

bool MeshIO::writeMesh(const Mesh& mesh, const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# FEM Mesh File\n";
    file << "# Nodes: " << mesh.nodes.size() << "\n";
    file << "# Elements: " << mesh.elements.size() << "\n";
    file << "# DOFs per node: " << mesh.dofsPerNode << "\n";
    file << "# Total DOFs: " << mesh.totalDofs << "\n\n";

    file << "*NODES\n";
    for (const auto& node : mesh.nodes) {
        file << node.id << " " << node.x << " " << node.y << " " << node.z << "\n";
    }

    file << "\n*ELEMENTS\n";
    for (const auto& elem : mesh.elements) {
        file << elem.id << " " << elem.youngsModulus << " "
             << elem.poissonRatio << " " << elem.thickness;
        for (Index nodeId : elem.nodeIds) {
            file << " " << nodeId;
        }
        file << "\n";
    }

    return true;
}

Mesh MeshIO::readMesh(const std::string& filename) const {
    Mesh mesh;
    if (!parseMeshFile(filename, mesh)) {
        throw std::runtime_error("Failed to read mesh file: " + filename);
    }
    return mesh;
}

bool MeshIO::parseMeshFile(const std::string& filename, Mesh& mesh) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    enum class Section { None, Nodes, Elements };
    Section currentSection = Section::None;

    mesh.dofsPerNode = 2;

    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        if (trimmed == "*NODES") {
            currentSection = Section::Nodes;
            continue;
        } else if (trimmed == "*ELEMENTS") {
            currentSection = Section::Elements;
            continue;
        }

        std::istringstream iss(trimmed);

        if (currentSection == Section::Nodes) {
            Index id;
            Scalar x, y, z = 0.0;
            if (iss >> id >> x >> y) {
                iss >> z;
                mesh.nodes.emplace_back(id, x, y, z);
            }
        } else if (currentSection == Section::Elements) {
            Element elem;
            if (iss >> elem.id >> elem.youngsModulus >> elem.poissonRatio >> elem.thickness) {
                Index nodeId;
                while (iss >> nodeId) {
                    elem.nodeIds.push_back(nodeId);
                }
                mesh.elements.push_back(elem);
            }
        }
    }

    mesh.totalDofs = static_cast<Index>(mesh.nodes.size()) * mesh.dofsPerNode;

    return true;
}

bool MeshIO::writeMeshVTK(const Mesh& mesh,
                          const DenseVector& displacements,
                          const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "FEM Simulation Results\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    file << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto& node : mesh.nodes) {
        file << node.x << " " << node.y << " " << node.z << "\n";
    }

    file << "\nCELLS " << mesh.elements.size() << " ";
    Index totalSize = 0;
    for (const auto& elem : mesh.elements) {
        totalSize += elem.nodeIds.size() + 1;
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

    if (displacements.size() > 0) {
        file << "\nPOINT_DATA " << mesh.nodes.size() << "\n";
        file << "VECTORS displacements double\n";
        Index idx = 0;
        for (size_t i = 0; i < mesh.nodes.size(); ++i) {
            Scalar dx = (idx < displacements.size()) ? displacements[idx++] : 0.0;
            Scalar dy = (idx < displacements.size()) ? displacements[idx++] : 0.0;
            file << dx << " " << dy << " 0.0\n";
        }
    }

    return true;
}

bool MeshIO::writeNodeCoords(const Mesh& mesh,
                             const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& node : mesh.nodes) {
        file << node.id << " " << node.x << " " << node.y << " " << node.z << "\n";
    }

    return true;
}

bool MeshIO::writeElementConnectivity(const Mesh& mesh,
                                      const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& elem : mesh.elements) {
        file << elem.id;
        for (Index nodeId : elem.nodeIds) {
            file << " " << nodeId;
        }
        file << "\n";
    }

    return true;
}

std::string MeshIO::trim(const std::string& s) const {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }

    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

} // namespace io
} // namespace fem
