#undef NETGEN_PYTHON

#include <fem.hpp>
#include <comp.hpp>
#include <meshaccess.hpp>
#include <vtkoutput.hpp>

int main(int argc, char** argv)
{
    ngcomp::MyMPI mpi_init(argc, argv);
    netgen::NgMPI_Comm comm(MPI_COMM_WORLD);
    /*
    ngcore::TaskManager taskmanager;
    taskmanager.SetNumThreads(4);
    taskmanager.StartWorkers();
    */
    ngcore::LocalHeap lh(100000);

    netgen::Ngx_Mesh mesh(argv[1], comm);
    std::shared_ptr<netgen::Mesh> ngmesh = mesh.GetMesh();
    std::shared_ptr<ngcomp::MeshAccess> ma = std::make_shared<ngcomp::MeshAccess>(ngmesh);

    netgen::Mesh* ngmsh_ptr = ngmesh.get();
    //ngmesh->SendRecvMesh();
    //ngmsh_ptr->ReceiveParallelMesh();

    ngcomp::Flags flags_fes;
    flags_fes.SetFlag("order", 2);
    flags_fes.SetFlag("complex", true);
    //flags_fes.SetFlag("dirichlet", dirbnd);

    std::shared_ptr<ngcomp::VectorH1FESpace> fes = std::make_shared<ngcomp::VectorH1FESpace>(ma, flags_fes);
    fes->Update();
    fes->FinalizeUpdate();

    const ngcomp::ProxyNode & u = fes->GetTrialFunction();
    const ngcomp::ProxyNode & v = fes->GetTestFunction();
    std::shared_ptr<ngcomp::CoefficientFunction> divu = u->Operator("div");
    std::shared_ptr<ngcomp::CoefficientFunction> divv = v->Operator("div");

    float mu = 8.59375e+07;
    float lam = 1.09375e+08;
    float rho = 7.7e-06;
    ngcore::Flags flags_bfa;
    std::shared_ptr<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>> bfa = std::make_shared<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>>(fes, "a", fes->GetFlags());
    bfa->AddIntegrator(std::make_shared<ngfem::SymbolicBilinearFormIntegrator>(
                           2.0 * mu *
                           ngcomp::InnerProduct(0.5 * (u->Deriv() + ngcomp::TransposeCF(u->Deriv())),
                                                0.5 * (v->Deriv() + ngcomp::TransposeCF(v->Deriv())))
                           + lam * ngcomp::InnerProduct(divu, divv)
                           , ngfem::VOL, ngfem::VOL));

    std::shared_ptr<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>> bfm = std::make_shared<ngcomp::T_BilinearFormSymmetric<ngcore::Complex>> (fes, "m", fes->GetFlags());
    bfm->AddIntegrator(std::make_shared<ngfem::SymbolicBilinearFormIntegrator>(
                           rho * ngcomp::InnerProduct(u, v), ngfem::VOL, ngfem::VOL));

    ngcore::Flags flags;
    //flags.SetFlag("type", "amgh1");
    // ngcomp::GetPreconditionerClasses().Print(std::cout);
    // multigrid, direct, local, bddc, bddcc, bddcrc, h1amg
    auto creator = ngcomp::GetPreconditionerClasses().GetPreconditioner("local");
    std::shared_ptr<ngcomp::Preconditioner> pre = creator->creatorbf(bfa, flags, "local");

    bfa->Assemble(lh);
    bfm->Assemble(lh);

    double precision = 1e-5;
    ngcomp::EigenSystem es(*(bfa->GetMatrixPtr()));
    es.SetPrecision(precision);
    es.SetPrecond(pre->GetMatrix());
    es.Calc();
    //es.PrintEigenValues(std::cout);

    int num = 15;
    ngla::Complex shift(0, 4000);
    ngcore::Array<ngcore::Complex> lams(num);
    ngcore::Array<std::shared_ptr<ngla::BaseVector>> evecs(num);
    ngla::Arnoldi<ngla::Complex> arnoldi(bfa->GetMatrixPtr(), bfm->GetMatrixPtr(), fes->GetFreeDofs());

    arnoldi.SetShift(shift);
    // allowed is: 'sparsecholesky', 'pardiso', 'pardisospd', 'mumps', 'masterinverse', 'umfpack'
    arnoldi.SetInverseType("pardiso");
    arnoldi.Calc(2*num+1, lams, num, evecs);

    for(int i = 0; i < num; i++)
    {
        std::cout << lams[i] << std::endl;
    }

    num = evecs.Size();
    ngcore::Flags gfu_flags;
    gfu_flags.SetFlag("multidim", num);
    std::shared_ptr<ngcomp::GridFunction> gfu = ngcomp::CreateGridFunction(fes, "gfu", gfu_flags);
    gfu->Update();
    for(int i = 0; i < num; i++)
        gfu->GetVector(i).Set(1.0, (*evecs[i]));
    gfu->GetMeshAccess()->SelectMesh();


    ma = gfu->GetMeshAccess();
    ngcore::Array<std::shared_ptr<ngcomp::CoefficientFunction>> cfs;
    ngcore::Array<std::string> names;

    for(int i = 0; i < num; i++)
    {
        std::stringstream ss;
        names.Append(ss.str());
        std::shared_ptr<ngcomp::GridFunctionCoefficientFunction> cf = std::make_shared<ngcomp::GridFunctionCoefficientFunction>(gfu, i);
        auto CF = Real(cf) + Imag(cf);
        auto norm = ngcomp::NormCF(cf);
        cfs.Append(Real(cf));
    }

    ngcore::Flags vtk_flags;
    vtk_flags.SetFlag("legacy", true);
    vtk_flags.SetFlag("subdivision", 0);
    ngcomp::VTKOutput<3> vtkoutput(cfs, vtk_flags, ma);
    ngcore::LocalHeap vtk_lh(100000);
    vtkoutput.Do(vtk_lh);

    return 0;
}
