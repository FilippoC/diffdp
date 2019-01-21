#pragma once

#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>

#include "dynet/expr.h"
#include "dynet/tensor-eigen.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/nodes-def-macros.h"

#include "diffdp/dynet/args.h"
#include "diffdp/algorithm/binary_phrase.h"

namespace dynet
{

Expression algorithmic_differentiable_binary_phrase_structure(
        const Expression &x,
        diffdp::DiscreteMode mode,
        std::vector<unsigned> *batch_sizes = nullptr
);

Expression entropy_regularized_binary_phrase_structure(
        const Expression &x,
        diffdp::DiscreteMode mode,
        std::vector<unsigned> *batch_sizes = nullptr
);

struct AlgorithmicDifferentiableBinaryPhraseStructure :
        public dynet::Node
{
    const diffdp::DiscreteMode mode;
    std::vector<unsigned>* batch_sizes = nullptr;

    std::vector<diffdp::AlgorithmicDifferentiableBinaryPhraseStructure*> _ce_ptr;

    explicit AlgorithmicDifferentiableBinaryPhraseStructure(
            const std::initializer_list<VariableIndex>& a,
            diffdp::DiscreteMode mode,
            std::vector<unsigned>* batch_sizes
    );

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override;
    size_t aux_storage_size() const override;

    virtual ~AlgorithmicDifferentiableBinaryPhraseStructure();
};

struct EntropyRegularizedBinaryPhraseStructure :
        public dynet::Node
{
    const diffdp::DiscreteMode mode;
    std::vector<unsigned>* batch_sizes = nullptr;

    std::vector<diffdp::EntropyRegularizedBinaryPhraseStructure*> _ce_ptr;

    explicit EntropyRegularizedBinaryPhraseStructure(
            const std::initializer_list<VariableIndex>& a,
            diffdp::DiscreteMode mode,
            std::vector<unsigned>* batch_sizes
    );

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override;
    size_t aux_storage_size() const override;

    virtual ~EntropyRegularizedBinaryPhraseStructure();
};

}
