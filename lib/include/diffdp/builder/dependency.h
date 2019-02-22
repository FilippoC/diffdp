#pragma once

#include "dynet/expr.h"

namespace diffdp
{

enum struct DependencyType
{
    Head,
    NonProjective,
    ProjectiveAlgDiff,
    ProjectiveEntropyReg
};

struct DependencySettings
{
    DependencyType type = DependencyType::Head;
    bool perturb = false;
};

struct DependencyBuilder
{
    const DependencySettings settings;
    dynet::ComputationGraph* _cg;
    bool _training = true;

    DependencyBuilder(const DependencySettings& settings);

    void new_graph(dynet::ComputationGraph& cg, bool training);
    dynet::Expression relaxed(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr, dynet::Expression* e_mask = nullptr);

    dynet::Expression relaxed_head(const dynet::Expression& arc_weights, dynet::Expression* e_mask = nullptr);
    dynet::Expression relaxed_nonprojective(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);
    dynet::Expression relaxed_projective_alg_diff(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);
    dynet::Expression relaxed_projective_entropy_reg(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);

    dynet::Expression argmax(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr, dynet::Expression* e_mask = nullptr);
    dynet::Expression argmax_head(const dynet::Expression& arc_weights, dynet::Expression* e_mask = nullptr);
    dynet::Expression argmax_nonprojective(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);
    dynet::Expression argmax_projective_alg_diff(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);
    dynet::Expression argmax_projective_entropy_reg(const dynet::Expression& arc_weights, std::vector<unsigned>* sizes = nullptr);

protected:
    /**
     * Perturb arc if training mode and setting.perturb == true
     */
    dynet::Expression perturb(const dynet::Expression& arc_weights);
};

}