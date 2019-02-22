#pragma once

#include "dynet/expr.h"

namespace diffdp
{

enum struct BinaryPhraseType
{
    AlgDiff,
    EntropyReg
};

struct BinaryPhraseSettings
{
    BinaryPhraseType type = BinaryPhraseType::AlgDiff;
    bool perturb = false;
};

struct BinaryPhraseBuilder
{
    const BinaryPhraseSettings settings;
    dynet::ComputationGraph* _cg;
    bool _training = true;

    BinaryPhraseBuilder(const BinaryPhraseSettings& settings);

    void new_graph(dynet::ComputationGraph& cg, bool training);
    dynet::Expression relaxed(const dynet::Expression& weights);
    dynet::Expression argmax(const dynet::Expression& weights);

    dynet::Expression relaxed_alg_diff(const dynet::Expression& weights);
    dynet::Expression relaxed_entropy_Reg(const dynet::Expression& weights);
protected:
    /**
     * Perturb arc if training mode and setting.perturb == true
     */
    dynet::Expression perturb(const dynet::Expression& arc_weights);
};

}