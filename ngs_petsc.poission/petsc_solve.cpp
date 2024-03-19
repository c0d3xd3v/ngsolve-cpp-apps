#undef NETGEN_PYTHON

#include <fem.hpp>
#include <comp.hpp>
#include <meshaccess.hpp>

#include <petsc.h>
#include <petsc_interface.hpp>
namespace pci = ngs_petsc_interface;

void petscSolve(std::shared_ptr<ngcomp::H1HighOrderFESpace> fes,
           std::shared_ptr<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>> a,
           std::shared_ptr<ngcomp::T_LinearForm<ngcore::Complex>> f,
           std::shared_ptr<ngcomp::GridFunction> gfu,
           netgen::NgMPI_Comm &comm)
{
    int argc = 0;
    char **argv;
    PetscInitialize(&argc, &argv, NULL, NULL);

    std::shared_ptr<pci::PETScMatrix> mata =
            std::make_shared<pci::PETScMatrix>(
                a->GetMatrixPtr(),
                fes->GetFreeDofs(),
                fes->GetFreeDofs(),
                (comm.Size() > 1) ? pci::PETScMatrix::MAT_TYPE::IS_AIJ : pci::PETScMatrix::MAT_TYPE::AIJ );

    if (comm.Size() > 1) {
        Mat petsc_mat = mata->GetPETScMat();
        MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
    }

    /** map row-vectors between PETSc and NGSovle**/
    auto vec_map = mata->GetRowMap();

    /** Create comaptible PETSc vectors to hold the right hand side and solution vectors **/
    Vec pc_rhs = vec_map->CreatePETScVector();
    Vec pc_sol = vec_map->CreatePETScVector();

    ngcomp::BaseVector & ng_rhs = f->GetVector();
    ngcomp::BaseVector & vecu = gfu->GetVector();

    /** Convert rhs and solve equation **/
    vec_map->NGs2PETSc(ng_rhs, pc_rhs);

    /** Create itterative solver **/
    ngcore::Array<std::string> solver_flags({ "ksp_type cg", "pc_type gamg", "ksp_monitor " });
    std::shared_ptr<pci::PETScKSP> ksp = std::make_shared<pci::PETScKSP>(mata, solver_flags, "my_poisson_ksp");

    /** We can directly access the wrapped PETSc KSP object. **/
    KSPSetTolerances(ksp->GetKSP(), 1e-10, 0, 1e10, 50);

    /** Calls KSPSetUp. **/
    ksp->Finalize();

    KSPSolve(ksp->GetKSP(), pc_rhs, pc_sol);

    /** convert solution back **/
    vec_map->PETSc2NGs(vecu, pc_sol);

    PetscFinalize();
}
