#ifndef FEM_ASSEMBLY_STIFFNESS_ASSEMBLER_H
#define FEM_ASSEMBLY_STIFFNESS_ASSEMBLER_H

#include "common/Types.h"
#include <vector>

namespace fem {
namespace assembly {

class StiffnessAssembler {
public:
    StiffnessAssembler();
    ~StiffnessAssembler();

    DenseMatrix assembleDenseMatrix(const Mesh& mesh);
    SparseMatrix assembleSparseMatrix(const Mesh& mesh);

    DenseMatrix computeElementStiffness(const Element& element,
                                        const std::vector<Node>& nodes) const;

    void applyBoundaryConditions(DenseMatrix& K, DenseVector& F,
                                 const std::vector<BoundaryCondition>& bcs) const;

    void applyBoundaryConditions(SparseMatrix& K, DenseVector& F,
                                 const std::vector<BoundaryCondition>& bcs) const;

    DenseVector assembleLoadVector(const Mesh& mesh,
                                   const std::vector<Load>& loads) const;

private:
    DenseMatrix computeQuad4Stiffness(const Element& element,
                                      const std::vector<Node>& nodes) const;

    DenseMatrix computeStrainDisplacementMatrix(
        Scalar xi, Scalar eta,
        const std::vector<Node>& elementNodes) const;

    DenseMatrix computeConstitutiveMatrix(Scalar youngsModulus,
                                          Scalar poissonRatio) const;

    void assembleElementIntoDense(DenseMatrix& globalK,
                                  const DenseMatrix& elementK,
                                  const std::vector<Index>& nodeIds,
                                  Index dofsPerNode) const;

    void assembleElementIntoTriplets(std::vector<Triplet>& triplets,
                                     const DenseMatrix& elementK,
                                     const std::vector<Index>& nodeIds,
                                     Index dofsPerNode) const;

    static constexpr int NODES_PER_ELEMENT = 4;
    static constexpr int DOFS_PER_NODE = 2;
    static constexpr int ELEMENT_DOFS = NODES_PER_ELEMENT * DOFS_PER_NODE;
};

} // namespace assembly
} // namespace fem

#endif // FEM_ASSEMBLY_STIFFNESS_ASSEMBLER_H
