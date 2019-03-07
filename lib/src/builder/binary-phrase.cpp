#include "diffdp/builder/binary-phrase.h"

#include "diffdp/dynet/binary_phrase.h"
#include "dytools/algorithms/span-parser.h"
#include "dytools/utils.h"

namespace diffdp
{

BinaryPhraseBuilder::BinaryPhraseBuilder(const BinaryPhraseSettings& settings) :
    settings(settings)
{}

void BinaryPhraseBuilder::new_graph(dynet::ComputationGraph& cg, bool training)
{
    _cg = &cg;
    _training = training;
}

dynet::Expression BinaryPhraseBuilder::relaxed(const dynet::Expression& weights)
{
    if (settings.type == BinaryPhraseType::AlgDiff)
        return relaxed_alg_diff(weights);
    else
        return relaxed_entropy_Reg(weights);
}

dynet::Expression BinaryPhraseBuilder::argmax(const dynet::Expression& weights)
{
    const auto size = weights.dim().rows();

    const auto p_weights = perturb(weights);
    const auto v_weights = as_vector(_cg->incremental_forward(p_weights));

    const auto tree = dytools::binary_span_parser(size, v_weights);

    std::vector<unsigned> indices;
    for (const auto& span : tree)
        indices.push_back(span.first + span.second * size);
    std::vector<float> values(indices.size(), 1.f);

    const auto output = dynet::input(*_cg, {size, size}, indices, values);
    return output;
}


dynet::Expression BinaryPhraseBuilder::relaxed_alg_diff(const dynet::Expression& weights)
{
    const auto p_weights = perturb(weights);
    return dytools::force_cpu(dynet::algorithmic_differentiable_binary_phrase_structure, p_weights, DiscreteMode::ForwardRegularized, nullptr);
}

dynet::Expression BinaryPhraseBuilder::relaxed_entropy_Reg(const dynet::Expression& weights)
{
    const auto p_weights = perturb(weights);
    return dytools::force_cpu(dynet::entropy_regularized_binary_phrase_structure, p_weights, DiscreteMode::ForwardRegularized, nullptr);
}


dynet::Expression BinaryPhraseBuilder::perturb(const dynet::Expression& arc_weights)
{
    if (settings.perturb and _training)
        return arc_weights + dynet::random_gumbel(*_cg, arc_weights.dim());
    else
        return arc_weights;
}


}