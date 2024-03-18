#undef NETGEN_PYTHON

#include <cmath>

#include <fem.hpp>
#include <comp.hpp>
#include <meshaccess.hpp>
#include <vtkoutput.hpp>

class xCF : public ngfem::CoefficientFunction
{
    public:
        xCF() {}
        double Evaluate(const ngfem::BaseMappedIntegrationPoint &ip) const
        {
            double x = ip.GetPoint().Data()[0];
            //double y = ip.GetPoint().Data()[1];
            //double z = ip.GetPoint().Data()[2];
            return x; // f(x, y, z)
        }
};


int main(int argc, char** argv)
{
    ngcomp::MyMPI mpi_init(argc, argv);
    netgen::NgMPI_Comm comm(MPI_COMM_WORLD);

    ngcore::LocalHeap lh(100000);

    netgen::Ngx_Mesh mesh(argv[1], comm);
    std::shared_ptr<netgen::Mesh> ngmesh = mesh.GetMesh();
    std::shared_ptr<ngcomp::MeshAccess> ma = std::make_shared<ngcomp::MeshAccess>(ngmesh);

    std::shared_ptr<xCF> mycf = std::make_shared<xCF>();

    ngcore::Array<std::shared_ptr<ngcomp::CoefficientFunction>> cfs;
    cfs.Append(ngcomp::Real(mycf));
    ngcore::Flags vtk_flags;
    vtk_flags.SetFlag("legacy", true);
    vtk_flags.SetFlag("subdivision", 0);
    ngcomp::VTKOutput<3> vtkoutput(cfs, vtk_flags, ma);
    ngcore::LocalHeap vtk_lh(100000);
    vtkoutput.Do(vtk_lh);

    return 0;
}
