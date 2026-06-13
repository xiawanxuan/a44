#include "assembly/StiffnessAssembler.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace fem {
namespace assembly {

StiffnessAssembler::StiffnessAssembler() = default;
StiffnessAssembler::~StiffnessAssembler() = default;

DenseMatrix StiffnessAssembler::assembleDenseMatrix(const Mesh& mesh) {
    Index nDofs = mesh.totalDofs;
    DenseMatrix K = DenseMatrix::Zero(nDofs, nDofs);

    for (const auto& element : mesh.elements) {
        DenseMatrix Ke = computeElementStiffness(element, mesh.nodes);
        assembleElementIntoDense(K, Ke, element.nodeIds, mesh.dofsPerNode);
    }

    return K;
}

SparseMatrix StiffnessAssembler::assembleSparseMatrix(const Mesh& mesh) {
    Index nDofs = mesh.totalDofs;
    std::vector<Triplet> triplets;
    triplets.reserve(mesh.elements.size() * ELEMENT_DOFS * ELEMENT_DOFS);

    for (const auto& element : mesh.elements) {
        DenseMatrix Ke = computeElementStiffness(element, mesh.nodes);
        assembleElementIntoTriplets(triplets, Ke, element.nodeIds, mesh.dofsPerNode);
    }

    SparseMatrix K(nDofs, nDofs);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();

    return K;
}

DenseMatrix StiffnessAssembler::computeElementStiffness(
    const Element& element,
    const std::vector<Node>& nodes) const {
    return computeQuad4Stiffness(element, nodes);
}

DenseMatrix StiffnessAssembler::computeQuad4Stiffness(
    const Element& element,
    const std::vector<Node>& nodes) const {

    DenseMatrix Ke = DenseMatrix::Zero(ELEMENT_DOFS, ELEMENT_DOFS);
    DenseMatrix D = computeConstitutiveMatrix(element.youngsModulus,
                                              element.poissonRatio);

    std::vector<Node> elemNodes(NODES_PER_ELEMENT);
    for (int i = 0; i < NODES_PER_ELEMENT; ++i) {
        elemNodes[i] = nodes[element.nodeIds[i]];
    }

    std::vector<Scalar> xiPoints = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    std::vector<Scalar> etaPoints = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    Scalar weight = 1.0;

    for (int ip = 0; ip < 2; ++ip) {
        for (int jp = 0; jp < 2; ++jp) {
            Scalar xi = xiPoints[ip];
            Scalar eta = etaPoints[jp];

            DenseMatrix B = computeStrainDisplacementMatrix(xi, eta, elemNodes);

            DenseMatrix J = DenseMatrix::Zero(2, 2);
            Scalar dNdxi[4], dNdeta[4];
            dNdxi[0] = -0.25 * (1 - eta);
            dNdxi[1] =  0.25 * (1 - eta);
            dNdxi[2] =  0.25 * (1 + eta);
            dNdxi[3] = -0.25 * (1 + eta);
            dNdeta[0] = -0.25 * (1 - xi);
            dNdeta[1] = -0.25 * (1 + xi);
            dNdeta[2] =  0.25 * (1 + xi);
            dNdeta[3] =  0.25 * (1 - xi);

            for (int k = 0; k < 4; ++k) {
                J(0, 0) += dNdxi[k] * elemNodes[k].x;
                J(0, 1) += dNdxi[k] * elemNodes[k].y;
                J(1, 0) += dNdeta[k] * elemNodes[k].x;
                J(1, 1) += dNdeta[k] * elemNodes[k].y;
            }

            Scalar detJ = J.determinant();
            if (detJ <= 0) {
                throw std::runtime_error("Negative or zero Jacobian determinant");
            }

            Ke += B.transpose() * D * B * detJ * weight * weight * element.thickness;
        }
    }

    return Ke;
}

DenseMatrix StiffnessAssembler::computeStrainDisplacementMatrix(
    Scalar xi, Scalar eta,
    const std::vector<Node>& elementNodes) const {

    DenseMatrix B = DenseMatrix::Zero(3, ELEMENT_DOFS);

    Scalar dNdxi[4], dNdeta[4];
    dNdxi[0] = -0.25 * (1 - eta);
    dNdxi[1] =  0.25 * (1 - eta);
    dNdxi[2] =  0.25 * (1 + eta);
    dNdxi[3] = -0.25 * (1 + eta);
    dNdeta[0] = -0.25 * (1 - xi);
    dNdeta[1] = -0.25 * (1 + xi);
    dNdeta[2] =  0.25 * (1 + xi);
    dNdeta[3] =  0.25 * (1 - xi);

    DenseMatrix J = DenseMatrix::Zero(2, 2);
    for (int k = 0; k < 4; ++k) {
        J(0, 0) += dNdxi[k] * elementNodes[k].x;
        J(0, 1) += dNdxi[k] * elementNodes[k].y;
        J(1, 0) += dNdeta[k] * elementNodes[k].x;
        J(1, 1) += dNdeta[k] * elementNodes[k].y;
    }

    DenseMatrix invJ = J.inverse();

    for (int i = 0; i < 4; ++i) {
        Scalar dNdx = invJ(0, 0) * dNdxi[i] + invJ(0, 1) * dNdeta[i];
        Scalar dNdy = invJ(1, 0) * dNdxi[i] + invJ(1, 1) * dNdeta[i];

        B(0, 2 * i)     = dNdx;
        B(1, 2 * i + 1) = dNdy;
        B(2, 2 * i)     = dNdy;
        B(2, 2 * i + 1) = dNdx;
    }

    return B;
}

DenseMatrix StiffnessAssembler::computeConstitutiveMatrix(
    Scalar youngsModulus, Scalar poissonRatio) const {

    DenseMatrix D = DenseMatrix::Zero(3, 3);
    Scalar factor = youngsModulus / (1.0 - poissonRatio * poissonRatio);

    D(0, 0) = factor * 1.0;
    D(0, 1) = factor * poissonRatio;
    D(1, 0) = factor * poissonRatio;
    D(1, 1) = factor * 1.0;
    D(2, 2) = factor * 0.5 * (1.0 - poissonRatio);

    return D;
}

void StiffnessAssembler::assembleElementIntoDense(
    DenseMatrix& globalK,
    const DenseMatrix& elementK,
    const std::vector<Index>& nodeIds,
    Index dofsPerNode) const {

    for (int i = 0; i < ELEMENT_DOFS; ++i) {
        Index globalRow = nodeIds[i / dofsPerNode] * dofsPerNode + (i % dofsPerNode);
        for (int j = 0; j < ELEMENT_DOFS; ++j) {
            Index globalCol = nodeIds[j / dofsPerNode] * dofsPerNode + (j % dofsPerNode);
            globalK(globalRow, globalCol) += elementK(i, j);
        }
    }
}

void StiffnessAssembler::assembleElementIntoTriplets(
    std::vector<Triplet>& triplets,
    const DenseMatrix& elementK,
    const std::vector<Index>& nodeIds,
    Index dofsPerNode) const {

    for (int i = 0; i < ELEMENT_DOFS; ++i) {
        Index globalRow = nodeIds[i / dofsPerNode] * dofsPerNode + (i % dofsPerNode);
        for (int j = 0; j < ELEMENT_DOFS; ++j) {
            Index globalCol = nodeIds[j / dofsPerNode] * dofsPerNode + (j % dofsPerNode);
            triplets.emplace_back(globalRow, globalCol, elementK(i, j));
        }
    }
}

void StiffnessAssembler::applyBoundaryConditions(
    DenseMatrix& K, DenseVector& F,
    const std::vector<BoundaryCondition>& bcs) const {

    for (const auto& bc : bcs) {
        Index dofIndex = bc.nodeId * 2 + bc.dof;
        if (dofIndex >= K.rows()) {
            continue;
        }

        F -= K.col(dofIndex) * bc.value;

        K.row(dofIndex).setZero();
        K.col(dofIndex).setZero();
        K(dofIndex, dofIndex) = 1.0;
        F(dofIndex) = bc.value;
    }
}

void StiffnessAssembler::applyBoundaryConditions(
    SparseMatrix& K, DenseVector& F,
    const std::vector<BoundaryCondition>& bcs) const {

    for (const auto& bc : bcs) {
        Index dofIndex = bc.nodeId * 2 + bc.dof;
        if (dofIndex >= K.rows()) {
            continue;
        }

        for (SparseMatrix::InnerIterator it(K, dofIndex); it; ++it) {
            if (it.row() != dofIndex) {
                F(it.row()) -= it.value() * bc.value;
            }
        }

        for (int k = 0; k < K.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
                if (it.row() == dofIndex || it.col() == dofIndex) {
                    if (it.row() == dofIndex && it.col() == dofIndex) {
                        const_cast<SparseMatrix::Scalar&>(it.valueRef()) = 1.0;
                    } else {
                        const_cast<SparseMatrix::Scalar&>(it.valueRef()) = 0.0;
                    }
                }
            }
        }

        F(dofIndex) = bc.value;
    }

    K.prune([](Index, Index, Scalar value) { return std::abs(value) > 1e-15; });
}

DenseVector StiffnessAssembler::assembleLoadVector(
    const Mesh& mesh,
    const std::vector<Load>& loads) const {

    DenseVector F = DenseVector::Zero(mesh.totalDofs);

    for (const auto& load : loads) {
        Index dofIndex = load.nodeId * mesh.dofsPerNode + load.dof;
        if (dofIndex < mesh.totalDofs) {
            F(dofIndex) += load.value;
        }
    }

    return F;
}

} // namespace assembly
} // namespace fem
