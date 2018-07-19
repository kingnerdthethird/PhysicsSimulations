#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>


#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;

template <int dim>
class Step5 {
public:
	Step5();
	void run();

private:
	void setup_system();
	void assemble_system();
	void solve();
	void output_results(const unsigned int cycle) const;

	Triangulation<dim> triangulation;
	FE_Q<dim> fe;
	DoFHandler<dim> dof_handler;

	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;

	Vector<double> solution;
	Vector<double> system_rhs;
};

template <int dim>
double coefficient (const Point<dim> &p) {
	if (p.square() < 0.5*0.5) {
		return 20;
	}

	else {
		return 1;
	}
}


int main() {
  Step5<2> laplace_problem_2d;
  laplace_problem_2d.run ();
  return 0;
}

template <int dim>
Step5<dim>::Step5() : fe(1), dof_handler(triangulation) {}

template <int dim>
void Step5<dim>::run() {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input_file("circle-grid.inp");
	Assert(dim == 2, ExcInternalError());
	grid_in.read_ucd(input_file);

	const SphericalManifold<dim> boundary;
	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold(0, boundary);

	for (unsigned int cycle = 0; cycle < 6; ++cycle) {
		std::cout << "Cycle " << cycle << ':' << std::endl;

		if (cycle != 0) {
			triangulation.refine_global(1);
		}

		std::cout << "   Number of active cells: " << triangulation.n_active_cells() << std::endl;
		std::cout << "   Total number of cells: " << triangulation.n_cells() << std::endl;

		setup_system();
		assemble_system();
		solve();
		output_results(cycle);
	}
}

template <int dim>
void Step5<dim>::setup_system() {
	dof_handler.distribute_dofs(fe);

	std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void Step5<dim>::assemble_system() {
	QGauss<dim> quadrature_formula(2);

	FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators()) {
		cell_matrix = 0;
		cell_rhs = 0;

		fe_values.reinit(cell);

		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
			const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_index));
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					cell_matrix(i, j) += (current_coefficient * fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
				}

				cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			for (unsigned int j = 0; j < dofs_per_cell; ++j) {
				system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
			}

			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}


	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template <int dim>
void Step5<dim>::solve() {
	SolverControl solver_control(1000, 1e-12);
	SolverCG<> solver(solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);

	solver.solve(system_matrix, solution, system_rhs, preconditioner);

	std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim>
void Step5<dim>::output_results(const unsigned int cycle) const {
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches();

	DataOutBase::EpsFlags eps_flags;
	eps_flags.z_scaling = 4;
	eps_flags.azimut_angle = 40;
	eps_flags.turn_angle = 10;
	data_out.set_flags(eps_flags);
	std::ofstream output("solution-" + std::to_string(cycle) + ".eps");
	data_out.write_eps(output);
}
