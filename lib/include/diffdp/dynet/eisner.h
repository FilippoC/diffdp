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
#include "diffdp/algorithm/eisner.h"

namespace diffdp
{

enum struct DependencyGraphMode
{
    Adjacency, // adjacency matrix
    Compact
};

std::pair<unsigned, unsigned> from_adjacency(const std::pair<unsigned, unsigned> dep, const diffdp::DependencyGraphMode mode);
std::pair<unsigned, unsigned> from_compact(const std::pair<unsigned, unsigned> dep, const diffdp::DependencyGraphMode mode);

}

namespace dynet
{

Expression algorithmic_differentiable_eisner(
        const Expression &x,
        diffdp::DiscreteMode mode,
        diffdp::DependencyGraphMode input_graph = diffdp::DependencyGraphMode::Compact,
        diffdp::DependencyGraphMode output_graph = diffdp::DependencyGraphMode::Compact,
        bool with_root_arcs = true,
        std::vector<unsigned> *batch_sizes = nullptr
);

Expression entropy_regularized_eisner(
        const Expression &x,
        diffdp::DiscreteMode mode,
        diffdp::DependencyGraphMode input_graph = diffdp::DependencyGraphMode::Compact,
        diffdp::DependencyGraphMode output_graph = diffdp::DependencyGraphMode::Compact,
        bool with_root_arcs = true,
        std::vector<unsigned> *batch_sizes = nullptr
);

struct AlgorithmicDifferentiableEisner :
        public dynet::Node
{
    const diffdp::DiscreteMode mode;
    const diffdp::DependencyGraphMode input_graph;
    const diffdp::DependencyGraphMode output_graph;
    bool with_root_arcs;
    std::vector<unsigned>* batch_sizes = nullptr;

    std::vector<diffdp::AlgorithmicDifferentiableEisner*> _ce_ptr;

    explicit AlgorithmicDifferentiableEisner(
            const std::initializer_list<VariableIndex>& a,
            diffdp::DiscreteMode mode,
            diffdp::DependencyGraphMode input_graph,
            diffdp::DependencyGraphMode output_graph,
            bool with_root_arcs,
            std::vector<unsigned>* batch_sizes
    );

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override;
    size_t aux_storage_size() const override;

    virtual ~AlgorithmicDifferentiableEisner();
};

struct EntropyRegularizedEisner :
        public dynet::Node
{
    const diffdp::DiscreteMode mode;
    const diffdp::DependencyGraphMode input_graph;
    const diffdp::DependencyGraphMode output_graph;
    bool with_root_arcs;
    std::vector<unsigned>* batch_sizes = nullptr;

    std::vector<diffdp::EntropyRegularizedEisner*> _ce_ptr;

    explicit EntropyRegularizedEisner(
            const std::initializer_list<VariableIndex>& a,
            diffdp::DiscreteMode mode,
            diffdp::DependencyGraphMode input_graph,
            diffdp::DependencyGraphMode output_graph,
            bool with_root_arcs,
            std::vector<unsigned>* batch_sizes
    );

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override;
    size_t aux_storage_size() const override;

    virtual ~EntropyRegularizedEisner();
};

}
