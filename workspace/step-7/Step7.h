#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

namespace Step7 {
	using namespace dealii;

	template <int dim>
	class SolutionBase {
	protected:
		static const unsigned int n_source_centers = 3;
		static const Point<dim> source_centers[n_source_centers];
		static const double width;
	};

	template <int dim>
	class Solution : public Function<dim>, protected SolutionBase<dim> {
	public:
		Solution() : Function<dim>();
		
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
		virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;
	};

	template <int dim>
	class RightHandSide : public Function<dim>, protected SolutionBase<dim> {
	public:
		RightHandSide() : Function<dim>();

		virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
	};

	template <int dim>
	class HelmholtzProblem {
	public:
		enum RefinementMode{ global_refinement, adaptive_refinement };
		HelmholtzProblem(const FiniteElement<dim> &fe, const RefinementMode refinement_mode);
		~HelmholtzProblem();
		void run();

	private:
		void setup_system();
		void assemble_system();
		void solve();
		void refine_grid();
		void process_solution(const unsigned int cycle);

		Triangulation<dim> triangulation;
		DoFHandler<dim> dof_handler;

		SmartPointer<const FiniteElement<dim>> fe;

		ConstraintMatrix hanging_node_constraints;

		SparsityPattern sparsity_pattern;
		SparseMatrix<double> system_matrix;

		Vector<double> solution;
		Vector<double> system_rhs;

		const RefinementMode refinement_mode;

		ConvergenceTable convergence_table;
	};
}