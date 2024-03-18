#undef NETGEN_PYTHON

#include <fem.hpp>
#include <comp.hpp>
#include <meshaccess.hpp>
#include <vtkoutput.hpp>


int main(int argc, char** argv)
{
    std::string inverse_direct_solver = "pardiso";
    if(argc > 2) {
        inverse_direct_solver = argv[2];
    }

    ngcomp::MyMPI mpi_init(argc, argv);
    netgen::NgMPI_Comm comm(MPI_COMM_WORLD);

    ngcore::LocalHeap lh(100000);

    netgen::Ngx_Mesh mesh(argv[1], comm);
    std::shared_ptr<netgen::Mesh> ngmesh = mesh.GetMesh();
    std::shared_ptr<ngcomp::MeshAccess> ma = std::make_shared<ngcomp::MeshAccess>(ngmesh);

    ngcomp::Flags flags_fes;
    flags_fes.SetFlag("order", 2);
    flags_fes.SetFlag("complex", true);
    flags_fes.SetFlag("dirichlet", ".*");
    std::shared_ptr<ngcomp::H1HighOrderFESpace> fes = std::make_shared<ngcomp::H1HighOrderFESpace>(ma, flags_fes);
    fes->Update();
    fes->FinalizeUpdate();

    const ngcomp::ProxyNode &u = fes->GetTrialFunction();
    const ngcomp::ProxyNode &v = fes->GetTestFunction();

    std::shared_ptr<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>> a = std::make_shared<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>>(fes, "a", fes->GetFlags());
    a->AddIntegrator(std::make_shared<ngfem::SymbolicBilinearFormIntegrator>(u->Deriv() * v->Deriv(), ngfem::VOL, ngfem::VOL));

    std::shared_ptr<ngcomp::ConstantCoefficientFunction> c = std::make_shared<ngcomp::ConstantCoefficientFunction>(1.0);
    std::shared_ptr<ngcomp::T_LinearForm<ngcore::Complex>> f = std::make_shared<ngcomp::T_LinearForm<ngcore::Complex>>(fes, "f", flags_fes);
    f->AddIntegrator(std::make_shared<ngfem::SymbolicLinearFormIntegrator>(v*c, ngfem::VOL, ngfem::VOL));

    a->Assemble(lh);
    f->Assemble(lh);

    a->GetMatrix().SetInverseType(inverse_direct_solver);
    ngcomp::BaseMatrix & am = a->GetMatrix();
    std::shared_ptr<ngcomp::BaseMatrix> inv = am.InverseMatrix(fes->GetFreeDofs());
    ngcomp::VVecExpr result = *inv * *f->GetVectorPtr();
    std::shared_ptr<ngcomp::BaseVector> x = am.CreateColVector();
    result.AssignTo<ngcore::Complex>(1.0, *x);

    ngcomp::Flags flags_gfu;
    flags_fes.SetFlag("complex", true);
    std::shared_ptr<ngcomp::GridFunction> gfu = ngcomp::CreateGridFunction(fes, "gfu", flags_gfu);
    gfu->Update();
    gfu->GetVector().Set(1.0, *x);

    ngcore::Array<std::shared_ptr<ngcomp::CoefficientFunction>> cfs;
    cfs.Append(ngcomp::Real(gfu));
    ngcore::Flags vtk_flags;
    vtk_flags.SetFlag("legacy", true);
    vtk_flags.SetFlag("subdivision", 0);
    ngcomp::VTKOutput<3> vtkoutput(cfs, vtk_flags, ma);
    ngcore::LocalHeap vtk_lh(100000);
    vtkoutput.Do(vtk_lh);

    return 0;
}
