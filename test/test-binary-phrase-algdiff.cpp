#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "AlgorithmicDifferentiableEisner"

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include "diffdp/algorithm/binary_phrase.h"

// using boost test with intolerance fails (too precise),
// so let's just use the same test as in Dynet.
bool check_grad(float g, float g_act)
{
    float f = std::fabs(g - g_act);
    float m = std::max(std::fabs(g), std::fabs(g_act));
    if (f > 0.01 && m > 0.f)
        f /= m;

    if (f > 0.01 || std::isnan(f))
        return false;
    else
        return true;
}

BOOST_AUTO_TEST_CASE(gradient)
        {
                const unsigned size = 10;
                const float sensitivity = 1e-2;

                diffdp::AlgorithmicDifferentiableBinaryPhraseStructure alg_diff(size);

                // check gradient
                std::vector<float> weights(size * size);
                for (unsigned output_left = 0 ; output_left < size ; ++output_left)
                {
                    for (unsigned output_right = output_left + 1 ; output_right < size ; ++output_right)
                    {
                        for (unsigned input_left = 0 ; input_left < size ; ++input_left)
                        {
                            for (unsigned input_right = input_left + 1; input_right < size; ++input_right)
                            {
                                // compute gradient using the algorithm
                                alg_diff.forward(
                                        [&] (const unsigned left, const unsigned right) -> float
                                        {
                                            return weights.at(left + right * size);
                                        }
                                );
                                alg_diff.backward(
                                        [&] (const unsigned left, const unsigned right) -> float
                                        {
                                            if (left == output_left && right == output_right)
                                                return 1.f;
                                            else
                                                return 0.f;
                                        }
                                );

                                const double computed_gradient = alg_diff.gradient(input_left, input_right);

                                // estimate the gradient
                                const float sensitivity = 1e-3;
                                const double original_weights = weights.at(input_left + input_right * size);

                                weights.at(input_left + input_right * size) = original_weights + sensitivity;
                                alg_diff.forward(
                                        [&] (const unsigned left, const unsigned right) -> float
                                        {
                                            return weights.at(left + right * size);
                                        }
                                );

                                const double output_a = alg_diff.output(output_left, output_right);

                                weights.at(input_left + input_right * size) = original_weights - sensitivity;
                                alg_diff.forward(
                                        [&] (const unsigned left, const unsigned right) -> float
                                        {
                                            return weights.at(left + right * size);
                                        }
                                );
                                const double output_b = alg_diff.output(output_left, output_right);

                                // restore
                                weights.at(input_left + input_right * size) = original_weights;

                                const double estimated_gradient = (output_a - output_b) / (2.f * sensitivity);

                                BOOST_CHECK(check_grad(computed_gradient, estimated_gradient));
                            }
                        }
                    }
                }
        }