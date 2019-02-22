#include "diffdp/builder/dependency.h"

#include <vector>
#include <limits>

#include "dytools/functions/rooted_arborescence_marginals.h"
#include "dytools/functions/masking.h"
#include "diffdp/dynet/eisner.h"

namespace diffdp
{

DependencyBuilder::DependencyBuilder(const DependencySettings& settings) :
    settings(settings)
{}

void DependencyBuilder::new_graph(dynet::ComputationGraph& cg, bool training)
{
    _cg = &cg;
    _training = training;
}

dynet::Expression DependencyBuilder::relaxed(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes, dynet::Expression* e_mask)
{
    if (settings.type == DependencyType::Head)
        return relaxed_head(arc_weights, e_mask);
    else if (settings.type == DependencyType::NonProjective)
        return relaxed_nonprojective(arc_weights, sizes);
    else if (settings.type == DependencyType::ProjectiveAlgDiff)
        return relaxed_projective_alg_diff(arc_weights, sizes);
    else
        return relaxed_projective_entropy_reg(arc_weights, sizes);
}

dynet::Expression DependencyBuilder::relaxed_head(const dynet::Expression& arc_weights, dynet::Expression* e_mask)
{
    if (e_mask != nullptr)
        if (e_mask->dim().rows() != 1 || e_mask->dim().cols() != arc_weights.dim().cols())
            throw std::runtime_error("Relaxed Head: mask has the wrong dimension");
    const auto p_arc_weights = perturb(arc_weights);

    // mask the diagonal
    const unsigned n_max_vertices = arc_weights.dim().rows();
    const auto e_inf_mask = dytools::main_diagonal_mask(*_cg, {n_max_vertices, n_max_vertices}, -std::numeric_limits<float>::infinity());

    auto heads = dynet::softmax(p_arc_weights + e_inf_mask);

    if (e_mask != nullptr)
        heads = dynet::cmult(heads, *e_mask);

    // first column should be empty (the root word has no head)
    std::vector<float> values(arc_weights.dim().cols(), 1.f);
    values[0] = 0.f;
    const auto mask = dynet::input(*_cg, {1, n_max_vertices}, values);
    heads = dynet::cmult(heads, mask);

    return heads;
}

dynet::Expression DependencyBuilder::relaxed_nonprojective(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes)
{
    const auto p_arc_weights = perturb(arc_weights);
    return dytools::rooted_arborescence_marginals(*_cg, p_arc_weights, sizes);
}

dynet::Expression DependencyBuilder::relaxed_projective_alg_diff(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes)
{
    const auto p_arc_weights = perturb(arc_weights);
    return dynet::algorithmic_differentiable_eisner(
            p_arc_weights,
            DiscreteMode::ForwardRegularized,
            DependencyGraphMode::Adjacency,
            DependencyGraphMode::Adjacency,
            true,
            sizes
    );
}

dynet::Expression DependencyBuilder::relaxed_projective_entropy_reg(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes)
{
    const auto p_arc_weights = perturb(arc_weights);
    return dynet::entropy_regularized_eisner(
            p_arc_weights,
            DiscreteMode::ForwardRegularized,
            DependencyGraphMode::Adjacency,
            DependencyGraphMode::Adjacency,
            true,
            sizes
    );
}

dynet::Expression DependencyBuilder::argmax(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes, dynet::Expression* e_mask)
{
    if (settings.type == DependencyType::Head)
        return argmax_head(arc_weights, e_mask);
    else if (settings.type == DependencyType::NonProjective)
        return argmax_nonprojective(arc_weights, sizes);
    else if (settings.type == DependencyType::ProjectiveAlgDiff)
        return argmax_projective_alg_diff(arc_weights, sizes);
    else
        return argmax_projective_entropy_reg(arc_weights, sizes);
}

dynet::Expression DependencyBuilder::argmax_head(const dynet::Expression&, dynet::Expression*)
{
    throw std::runtime_error("Not implemented yet.");
}

dynet::Expression DependencyBuilder::argmax_nonprojective(const dynet::Expression&, std::vector<unsigned>*)
{
    throw std::runtime_error("Not implemented yet.");
}

dynet::Expression DependencyBuilder::argmax_projective_alg_diff(const dynet::Expression&, std::vector<unsigned>*)
{
    throw std::runtime_error("Not implemented yet.");
}

dynet::Expression DependencyBuilder::argmax_projective_entropy_reg(const dynet::Expression&, std::vector<unsigned>*)
{
    throw std::runtime_error("Not implemented yet.");
}


dynet::Expression DependencyBuilder::perturb(const dynet::Expression& arc_weights)
{
    if (settings.perturb and _training)
        return arc_weights + dynet::random_gumbel(*_cg, arc_weights.dim());
    else
        return arc_weights;
}


}