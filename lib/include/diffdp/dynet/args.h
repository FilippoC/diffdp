#pragma once

namespace diffdp
{

enum struct DiscreteMode
{
    Null, // do not backpropagate
    StraightThrough, // discrete output, copy input gradient
    ForwardRegularized, // differentiable surrogate
    BackwardRegularized // forward: discrete, backward: differentiable surrogate
};

}