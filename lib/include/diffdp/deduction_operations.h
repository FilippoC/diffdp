#pragma once

/**
 * Header-only implementation of operations on deduction rules
 *
 * Inputs/outputs are iterator-like objects that must be:
 * - copy-constructible
 * - dereferenceable
 * - incrementable
 *
 * Author: Caio Corro
 */

#include <iostream>
#include "diffdp/math.h"

namespace diffdp
{

template<class T, class U, class V, class W>
float forward_algorithmic_softmax(
        T left_antecedent, U right_antecedent,
        V split_weights,
        W backptr,
        unsigned size
)
{
    cwise_add(split_weights, left_antecedent, right_antecedent, size);
    softmax(backptr, split_weights, size);
    return dot(split_weights, backptr, size);
}

template<class T, class U, class V>
void forward_backtracking(
        T contrib_left_antecedent, U contrib_right_antecedent,
        const float contrib_consequent,
        V backptr,
        const unsigned size
)
{
    add_cwise_mult(contrib_left_antecedent, backptr, contrib_consequent, size);
    add_cwise_mult(contrib_right_antecedent, backptr, contrib_consequent, size);
}

template<class T, class U, class V, class A, class B, class C>
void backward_backtracking(
        T contrib_left_antecedent, U contrib_right_antecedent,
        const float contrib_consequent,
        V backptr,

        A gradient_contrib_left_antecedent, B gradient_contrib_right_antecedent,
        float *gradient_contrib_consequent,
        C gradient_backptr,

        const unsigned size
)
{
    *gradient_contrib_consequent += dot(backptr, gradient_contrib_left_antecedent, size);
    *gradient_contrib_consequent += dot(backptr, gradient_contrib_right_antecedent, size);
    add_cwise_mult(gradient_backptr, gradient_contrib_left_antecedent, contrib_consequent, size);
    add_cwise_mult(gradient_backptr, gradient_contrib_right_antecedent, contrib_consequent, size);
}


template<class T, class U, class V, class W, class A, class B, class C, class D>
void backward_algorithmic_softmax(
        T left_antecedent, U right_antecedent,
        V split_weights,
        W backptr,

        A gradient_left_antecedent, B gradient_right_antecedent,
        const float gradient_consequent,
        C gradient_split_weights,
        D gradient_backptr,

        unsigned size
)
{
    add_cwise_mult(gradient_backptr, split_weights, gradient_consequent, size);
    add_cwise_mult(gradient_split_weights, backptr, gradient_consequent, size);

    backprop_softmax(gradient_split_weights, gradient_backptr, split_weights, backptr, size);

    add(gradient_left_antecedent, gradient_split_weights, size);
    add(gradient_right_antecedent, gradient_split_weights, size);
}


template<class T, class U, class V, class W>
float forward_entropy_reg(
        T left_antecedent, U right_antecedent,
        V split_weights,
        W backptr,
        unsigned size
)
{
    cwise_add(split_weights, left_antecedent, right_antecedent, size);
    softmax(backptr, split_weights, size);
    float m = max(split_weights, size);
    float s = 0;
    for (unsigned i = 0 ; i < size ; ++i, ++split_weights)
        s += std::exp(*split_weights-m);
    return m + std::log(s);
}


template<class T, class U, class V, class W, class A, class B, class C, class D>
void backward_entropy_reg(
        T left_antecedent, U right_antecedent,
        V split_weights,
        W backptr,

        A gradient_left_antecedent, B gradient_right_antecedent,
        const float gradient_consequent,
        C gradient_split_weights,
        D gradient_backptr,

        unsigned size
)
{
    add_cwise_mult(gradient_split_weights, backptr, gradient_consequent, size);

    backprop_softmax(gradient_split_weights, gradient_backptr, split_weights, backptr, size);

    add(gradient_left_antecedent, gradient_split_weights, size);
    add(gradient_right_antecedent, gradient_split_weights, size);
}

}