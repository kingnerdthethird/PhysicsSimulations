#include "Step7.h"
using namespace Step7;

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

	for (unsigned int i = 0; i < this->n_source_centers; ++1) {
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
HelmholtzProblem<dim>::HelmholtzProblem(const FiniteElement<dim> &fe, const RefinementMode refinement_mode) {

}

template <int dim>
HelmholtzProblem<dim>::~HelmholtzProblem() {

}

template <int dim>
void HelmholtzProblem<dim>::run() {

}

template <int dim>
void HelmholtzProblem<dim>::setup_system() {

}

template <int dim>
void HelmholtzProblem<dim>::assemble_system() {

}

template <int dim>
void HelmholtzProblem<dim>::solve() {

}

template <int dim>
void HelmholtzProblem<dim>::refine_grid() {

}

template <int dim>
void HelmholtzProblem<dim>::process_solution(const unsigned int cycle) {

}