#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EntropyRegularizedEisner"

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include "diffdp/algorithm/eisner.h"

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

    diffdp::AlgorithmicDifferentiableEisner alg_diff_eisner(size);

    // check gradient
    std::vector<float> weights(size * size);
    for (unsigned output_head = 0 ; output_head < size ; ++output_head)
    {
        for (unsigned output_mod = 1 ; output_mod < size ; ++output_mod)
        {
            if (output_head == output_mod)
            continue;

            for (unsigned input_head = 0 ; input_head < size ; ++input_head)
            {
                for (unsigned input_mod = 1; input_mod < size; ++input_mod)
                {
                    if (input_head == input_mod)
                        continue;

                    // compute gradient using the algorithm
                    alg_diff_eisner.forward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            return weights.at(head + mod * size);
                        }
                        );
                    alg_diff_eisner.backward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            if (head == output_head && mod == output_mod)
                                return 1.f;
                            else
                                return 0.f;
                        }
                    );

                    const double computed_gradient = alg_diff_eisner.gradient(input_head, input_mod);

                    // estimate the gradient
                    const float sensitivity = 1e-3;
                    const double original_weights = weights.at(input_head + input_mod * size);

                    weights.at(input_head + input_mod * size) = original_weights + sensitivity;
                    alg_diff_eisner.forward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            return weights.at(head + mod * size);
                        }
                        );

                    const double output_a = alg_diff_eisner.output(output_head, output_mod);

                    weights.at(input_head + input_mod * size) = original_weights - sensitivity;
                        alg_diff_eisner.forward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            return weights.at(head + mod * size);
                        }
                    );
                    const double output_b = alg_diff_eisner.output(output_head, output_mod);

                    // restore
                    weights.at(input_head + input_mod * size) = original_weights;

                    const double estimated_gradient = (output_a - output_b) / (2.f * sensitivity);

                    BOOST_CHECK(check_grad(computed_gradient, estimated_gradient));

                }
            }
        }
    }
}