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
#include "Step7.h"

int main() {
	const unsigned int dim = 2;

	try {
		using namespace Step7;
		using namespace dealii;

		{
			std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl;
			std::cout << "=============================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(1);
			HelmholtzProblem<dim> helmholtz_problem_2d(fe, HelmholtzProblem<dim>::adaptive_refinement);

			helmholtz_problem_2d.run();

			std::cout << std::endl;
		}

		{
			std::cout << "Solving with Q1 elements, global refinement" << std::endl;
			std::cout << "=============================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(1);
			HelmholtzProblem<dim> helmholtz_problem_2d(fe, HelmholtzProblem<dim>::global_refinement);

			helmholtz_problem_2d.run();

			std::cout << std::endl;
		}

		{
			std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl;
			std::cout << "=============================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(2);
			HelmholtzProblem<dim> helmholtz_problem_2d(fe, HelmholtzProblem<dim>::adaptive_refinement);

			helmholtz_problem_2d.run();

			std::cout << std::endl;
		}

		{
			std::cout << "Solving with Q2 elements, global refinement" << std::endl;
			std::cout << "=============================================" << std::endl;
			std::cout << std::endl;

			FE_Q<dim> fe(2);
			HelmholtzProblem<dim> helmholtz_problem_2d(fe, HelmholtzProblem<dim>::global_refinement);

			helmholtz_problem_2d.run();

			std::cout << std::endl;
		}
	}

	catch (std::exception &exc) {
		std::cerr << std::endl << std::endl << std::endl;

		std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl;
		std::cerr << "Aborting!" << std::endl << std::endl;

		return 1;
	}

	catch (...) {
		std::cerr << std::endl << std::endl << std::endl;

		std::cerr << "Unknown exception! " << std::endl << std::endl;
		std::cerr << "Aborting!" << std::endl << std::endl;

		return 2;
	}

	return 0;
}