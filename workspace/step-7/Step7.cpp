#include "Step7.h"

namespace Step7 {
	using namespace dealii;

	template <>
	const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] = {
		Point<1>(-1.0 / 3.0),
		Point<1>(0.0),
		Point<1>(1.0 / 3.0)
	};

	template <>
	const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] = {
		Point<2>(-0.5, +0.5),
		Point<2>(-0.5, -0.5),
		Point<2>(+0.5, -0.5)
	};

	template <>
	const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] = {
		Point<3>(-0.5, +0.5, +0.5),
		Point<3>(-0.5, -0.5, 0.0),
		Point<3>(+0.5, -0.5, -0.5)
	};

	template <int dim>
	const double SolutionBase<dim>::width = 1.0 / 8.0;

	template <int dim>
	Solution<dim>::Solution() : Function<dim>() {}

	template <int dim>
	double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
		double return_value = 0;
		for (unsigned int i = 0; i < this->n_source_centers; ++i) {
			const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
			return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
		}

		return return_value;
	}

	template <int dim>
	Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const {
		Tensor<1, dim> return_value;

		for (unsigned int i = 0; i < this->n_source_centers; ++i) {
			const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
			return_value += (-2 / (this->width * this->width) * std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
		}

		return return_value;
	}

	template <int dim>
	RightHandSide<dim>::RightHandSide() : Function<dim>() {}

	template <int dim>
	double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
		double return_value = 0;

		for (unsigned int i = 0; i < this->n_source_centers; ++i) {
			const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
			return_value += ((2 * dim - 4 * x_minus_xi.norm_square() / (this->width * this->width)) / (this->width * this->width) * std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
			return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
		}

		return value;
	}

	template <int dim>
	HelmholtzProblem<dim>::HelmholtzProblem(const FiniteElement<dim> &fe, const RefinementMode refinement_mode) : dof_handler(triangulation), fe(&fe), refinement_mode(refinement_mode) {}

	template <int dim>
	HelmholtzProblem<dim>::~HelmholtzProblem() {
		dof_handler.clear();
	}

	template <int dim>
	void HelmholtzProblem<dim>::run() {
		const unsigned int n_cycles = (refinement_mode == global_refinement) ? 5 : 9;
		for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
			if (cycle == 0) {
				GridGenerator::hyper_cube(triangulation, -1, 1);
				triangulation.refine_global(3);

				typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
				for (; cell != endc; ++cell) {
					for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
						if ((std::fabs(cell->face(face_number)->center()(0) - (-1)) < 1e-12) || (std::fabs(cell->face(face_number)->center(1) - (-1)) < 1e-12)) {
							cell->face(face_number)->set_boundary_id(1);
						}
					}
				}
			}

			else {
				refine_grid();
			}

			setup_system();
			assemble_system();
			solve();
			process_solution(cycle);
		}

		std::string vtk_filename;
		switch (refinement_mode) {
		case global_refinement: {
			vtk_filename = "solution-global";

			break;
		}

		case adaptive_refinement: {
			vtk_filename = "solution-adaptive";

			break;
		}

		default: {
			Assert(false, ExcNotImplemented());
		}
		}

		switch (fe->degree) {
		case 1: {
			vtk_filename += "-q1";

			break;
		}

		case 2: {
			vtk_filename += "-q2";

			break;
		}

		default: {
			Assert(false, ExcNotImplemented());
		}
		}

		vtk_filename += std::to_string(dim);
		vtk_filename += ".vtk";
		std::ofstream output(vtk_filename.c_str());

		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(solution, "solution");

		data_out.build_patches(fe->degree);
		data_out.write_vtk(output);

		convergence_table.set_precision("L2", 3);
		convergence_table.set_precision("H1", 3);
		convergence_table.set_precision("Linfty", 3);

		convergence_table.set_scientific("L2", true);
		convergence_table.set_scientific("H1", true);
		convergence_table.set_scientific("Linfty", true);

		convergence_table.set_tex_caption("cells", "\\# cells");
		convergence_table.set_tex_caption("dofs", "\\# dofs");
		convergence_table.set_tex_caption("L2", "L^2-error");
		convergence_table.set_tex_caption("H1", "H^1-error");
		convergence_table.set_tex_caption("Linfty", "L^\\infty-error");

		convergence_table.set_tex_format("cells", "r");
		convergence_table.set_tex_format("dofs", "r");

		std::cout << std::endl;
		convergence_table.write_text(std::cout);

		std::string error_filename = "error";
		switch (refinement_mode) {
		case global_refinement: {
			error_filename += "-global";
			break;
		}

		case adaptive_refinement: {
			error_filename += "-adaptive";
			break;
		}

		default: {
			Assert(false, ExcNotImplemented());
		}
		}

		error_filename += std::to_string(dim);
		error_filename += ".tex";
		std::ofstream error_table_file(error_filename.c_str());

		convergence_table.write_tex(error_table_file);

		if (refinement_mode == global_refinement) {
			convergence_table.add_column_to_supercolumn("cycle", "n cells");
			convergence_table.add_column_to_supercolumn("cycle", "n cells");

			std::vector<std::string> new_order;
			new_order.emplace_back("n cells");
			new_order.emplace_back("H1");
			new_order.emplace_back("L2");
			convergence_table.set_column_order(new_order);

			convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
			convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
			convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
			convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

			std::cout << std::endl;
			convergence_table.write_text(std::cout);

			std::string conv_filename = "convergence";
			switch (refinement_mode) {
			case global_refinement: {
				conv_filename += "-global";
				break;
			}

			case adaptive_refinement: {
				conv_filename += "-adaptive";
				break;
			}

			default: {
				Assert(false, ExcNotImplemented());
			}
			}

			conv_filename += std::to_string(dim);
			conv_filename += ".tex";

			std::ofstream table_file(conv_filename.c_str());
			convergence_table.write_tex(table_file);
		}
	}

	template <int dim>
	void HelmholtzProblem<dim>::setup_system() {
		dof_handler.distribute_dofs(*fe);
		DoFRenumbering::Cuthill_McKee(dof_handler);

		hanging_node_constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);

		hanging_node_constraints.close();

		DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp);
		hanging_node_constraints.condense(dsp);
		sparsity_pattern.copy_from(dsp);

		system_matrix.reinit(sparsity_pattern);

		solution.reinit(dof_handler.n_dofs());
		system_rhs.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void HelmholtzProblem<dim>::assemble_system() {
		QGauss<dim> quadrature_formula(3);
		QGauss<dim - 1> face_quadrature_formula(3);

		const unsigned int n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();

		const unsigned int dofs_per_cell = fe->dofs_per_cell;

		FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double> cell_rhs(dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		FEValues<dim> fe_values(*fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
		FEFaceValues<dim> fe_face_values(*fe, face_quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

		const RightHandSide<dim> right_hand_side;
		std::vector<double> rhs_values(n_q_points);

		const Solution<dim> exact_solution;

		typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();

		for (; cell != endc; ++cell) {
			cell_matrix = 0;
			cell_rhs = 0;

			fe_values.reinit(cell);

			right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
				for (unsigned int i = 0; i < dofs_per_cell; ++i) {
					for (unsigned int j = 0; j < dofs_per_cell; ++j) {
						cell_matrix(i, j) += ((fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) + fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point)) * fe_values.JxW(q_point));
					}

					cell_rhs(i) += (fe_values.shape_value(i, q_point) * rhs_values[q_point] * fe_values.JxW(q_point));
				}
			}

			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number) {
				if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == 1) {
					fe_face_values.reinit(cell, face_number);
					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {
						const double neumann_value = (exact_solution.gradient(fe_face_values.quadrature_point(q_point)) * fe_face_values.normal_vector(q_point));

						for (unsigned int i = 0; i < dofs_per_cell; ++i) {
							cell_rhs(i) += (neumann_value * fe_face_values.shape_value(i, q_point) * fe_face_values.JxW(q_point));
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
			}
		}

		hanging_node_constraints.condense(system_matrix);
		hanging_node_constraints.condense(system_rhs);

		std::map<types::global_dof_index, double> boundary_values;
		VectorTools::interpolate_boundary_values(dof_handler, 0, Solution<dim>(), boundary_values);
		MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
	}

	template <int dim>
	void HelmholtzProblem<dim>::solve() {
		SolverControl solver_control(1000, 1e-12);
		SolverCG<> cg(solver_control);

		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(system_matrix, 1.2);

		cg.solve(system_matrix, solution, system_rhs, preconditioner);

		hanging_node_constraints.distribute(solution);
	}

	template <int dim>
	void HelmholtzProblem<dim>::refine_grid() {
		switch (refinement_mode) {
		case global_refinement: {
			triangulation.refine_global(1);

			break;
		}

		case adaptive_refinement: {
			Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

			KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(3), typename FunctionMap<dim>::type(), solution, estimated_error_per_cell);

			GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

			triangulation.execute_coarsening_and_refinement();

			break;
		}

		default: {
			Assert(false, ExcNotImplemented());
		}
		}
	}

	template <int dim>
	void HelmholtzProblem<dim>::process_solution(const unsigned int cycle) {
		Vector<float> difference_per_cell(triangulation.n_active_cells());
		VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(3), VectorTools::L2_norm);
		const double L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

		VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, QGauss<dim>(3), VectorTools::H1_seminorm);
		const double H1_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_seminorm);

		const QTrapez<1> q_trapez;
		const QIterated<dim> q_iterated(q_trapez, 5);
		VectorTools::integrate_difference(dof_handler, solution, Solution<dim>(), difference_per_cell, q_iterated, VectorTools::Linfty_norm);
		const double Linfty_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::Linfty_norm);

		const unsigned int n_active_cells = triangulation.n_active_cells();
		const unsigned int n_dofs = dof_handler.n_dofs();

		std::cout << "Cycle " << cycle << ':' << std::endl;
		std::cout << "    Number of active cells: " << n_active_cells << std::endl;
		std::cout << "    Number of degrees of freedom: " << n_dofs << std::endl;

		convergence_table.add_value("cycle", cycle);
		convergence_table.add_value("cells", n_active_cells);
		convergence_table.add_value("dofs", n_dofs);
		convergence_table.add_value("L2", L2_error);
		convergence_table.add_value("H1", H1_error);
		convergence_table.add_value("cycle", Linfty_error);
	}
}