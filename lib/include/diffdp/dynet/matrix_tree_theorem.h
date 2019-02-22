#pragma once

#include <utility>
#include "dynet/expr.h"
#include "dynet/nodes-def-macros.h"

namespace dynet
{

Expression matrix_tree_theorem(const Expression &weights);


struct MatrixTreeTheorem :
        public dynet::Node
{
    explicit MatrixTreeTheorem(
            const std::initializer_list<VariableIndex>& a
    );

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override;
    size_t aux_storage_size() const override;
};


}