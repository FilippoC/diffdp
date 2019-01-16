#pragma once

/**
 * Very small header-only math library used to simplify
 * the implementation of continuous relaxation of dynamic programming algorithms.
 *
 * Inputs/outputs are iterator-like objects that must be:
 * - copy-constructible
 * - dereferenceable
 * - incrementable
 *
 * Author: Caio Corro
 */

#include <cmath>
#include <limits>

namespace diffdp
{

/**
 * Performs an element-wise sum of vectors input1 and output2.
 * The result is stored in output.
 *
 * @param output Vector where the result will be stored
 * @param input1 First input vector
 * @param input2 Second intput vector
 * @param size Size of the input vectors
 */
template<class T, class U, class V>
void cwise_add(T output, U input1, V input2, const unsigned size)
{
    for (unsigned i = 0u; i < size; ++i, ++input1, ++input2, ++output)
        *output = *input1 + *input2;
}

/**
 * Return the maximum element stored in a vector.
 *
 * @param input Input vector
 * @param size Size of the input vector
 * @return The maximum element stored in the input vector
 */
template<class T>
float max(T input, const unsigned size)
{
    float value = -std::numeric_limits<float>::infinity();
    for (unsigned i = 0u; i < size; ++i, ++input)
        value = std::max(value, *input);
    return value;
}

/**
 * Divide each element of a vector by a given value.
 * The results is stored inplace.
 *
 * @param input Input/output vector
 * @param v Value to divide by
 * @param size Size of the input/output vector
 */
template<class T>
void inplace_cwise_div(T input, const float v, const unsigned size)
{
    for (unsigned i = 0u; i < size; ++i, ++input)
        *input = *input / v;
}

/**
 * Component wise addition between two vectors.
 * The results is stored in the first argument.
 *
 * @param input Input vector
 * @param output Output vector
 * @param size Size of the input vector
 */
template<class T, class U>
void add(T output, U input, const unsigned size)
{
    for (unsigned i = 0u; i < size; ++i, ++input, ++output)
        *output += *input;
}

/**
 * Perform a component wise multiplication of the input vector with a scalar
 * and store the results as a component-wise addition with the output.
 *
 * @param output Output vector
 * @param input Input vector
 * @param v Scalar use for the multiplication
 * @param size Size of the input vector
 */
template<class T, class U>
void add_cwise_mult(T output, U input, const float v, const unsigned size)
{
    for (unsigned i = 0u; i < size; ++i, ++input, ++output)
        *output += (*input) * v;
}

/**
 * Return the dot product between the two input vectors.
 *
 * @param input1 First input vector
 * @param input2 Second input vector
 * @param size Size of the input vectors
 * @return The dot product between the two input vectors
 */
template<class T, class U>
float dot(T input1, U input2, const unsigned size)
{
    float ret = 0.f;
    for (unsigned i = 0u; i < size; ++i, ++input1, ++input2)
        ret += (*input1) * (*input2);
    return ret;
}

/**
 * Exponentiate each element of a vector by first substracting a scalar.
 *
 * @param output Output vector
 * @param input Input vector
 * @param m Scalar to substract
 * @param size Size of the input
 * @return Return the sum of the elements of the output vector (i.e. the partition)
 */
template<class T, class U>
float exp_minus_cst(T output, U input, const float m, const unsigned size)
{
    float ret = 0.f;
    for (unsigned i = 0u; i < size; ++i, ++input, ++output)
    {
        const float v = std::exp(*input - m);
        *output = v;
        ret += v;
    }
    return ret;
}

/**
 * Compute the softmax of the input.
 *
 * @param output Output vector
 * @param input Input vector
 * @param size Size of the input vector
 */
template<class T, class U>
void softmax(T output, U input, unsigned size) noexcept
{

    float m = max(input, size);
    float z = exp_minus_cst(output, input, m, size);
    inplace_cwise_div(output, z, size);
}

/**
 * Backpropagate through a softmax function.
 *
 * @param gradient_input Gradient of the softmax
 * @param gradient_output Gradient incoming to the softmax
 * @param input Input of the softmax
 * @param output Output of the softmax (i.e. it should be computed beforehand)
 * @param size Size of the input
 */
template<class T, class U, class A, class B>
void backprop_softmax(A gradient_input, B gradient_output, T input, U output, const unsigned size)
{
    const float s = dot(gradient_output, output, size);
    for (unsigned i = 0; i < size; ++i, ++gradient_input, ++gradient_output, ++output)
        *gradient_input += (*output) * ((*gradient_output) - s);
}

}