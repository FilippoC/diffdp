#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EntropyRegularizedBinaryPhrase"

#include <cmath>
#include <algorithm>
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

BOOST_AUTO_TEST_CASE(first_order_gradient, * utf::tolerance(1e-2f))
{
    const unsigned size = 10;
    const float sensitivity = 1e-2;

    std::vector<float> weights(size * size);
    for (unsigned i = 0 ; i < size ; ++i)
        weights.at(i) = size;

    diffdp::EntropyRegularizedBinaryPhraseStructure parser(size);
        parser.forward(
        [&] (unsigned left, unsigned right) -> float
        {
            return weights.at(left + right * size);
        }
    );

    for (unsigned left = 0 ; left < size ; ++left)
    {
        for (unsigned right = left + 1; right < size ; ++right)
        {
            const float computed_arc = parser.output(left, right);

            // estimate the gradient
            const float original_weights = weights.at(left + right * size);

            weights.at(left + right * size) = original_weights + sensitivity;
            diffdp::EntropyRegularizedBinaryPhraseStructure parser2(size);
                parser2.forward(
                [&] (const unsigned left, const unsigned right) -> float
                {
                    return weights.at(left + right * size);
                }
            );
            const float output_a = parser2.chart_forward->weight(0, size-1);

            weights.at(left + right * size) = original_weights - sensitivity;
            parser2.forward(
            [&] (const unsigned left, const unsigned right) -> float
                {
                    return weights.at(left + right * size);
                }
            );
            const float output_b = parser2.chart_forward->weight(0, size-1);

            // restore
            weights.at(left + right * size) = original_weights;

            const float estimated_arc = (output_a - output_b) / (2.f * sensitivity);

            BOOST_CHECK(check_grad(computed_arc, estimated_arc));
        }
    }
}


BOOST_AUTO_TEST_CASE(second_order_gradient, * utf::tolerance(1e-2f))
{
    const unsigned size = 10;
    const float sensitivity = 1e-5;

    std::vector<float> weights(size * size);
    for (unsigned i = 0 ; i < size ; ++i)
    weights.at(i) = size;

    diffdp::EntropyRegularizedBinaryPhraseStructure parser(size);
    parser.forward(
        [&] (unsigned left, unsigned right) -> float
        {
            return weights.at(left + right * size);
        }
    );

    for (unsigned input_left = 0 ; input_left < size ; ++input_left)
    {
        for (unsigned input_right = input_left + 1; input_right < size ; ++input_right)
        {
            for (unsigned output_left = 0 ; output_left < size ; ++output_left)
            {
                for (unsigned output_right = output_left + 1; output_right < size; ++output_right)
                {
                    parser.backward(
                        [&](const unsigned left, const unsigned right)
                        {
                            if (left == output_left && right == output_right)
                                return 1.f;
                            else
                                return 0.f;
                        }
                    );
                    const float computed_gradient = parser.gradient(input_left, input_right);

                    // estimate the gradient
                    const float sensitivity = 1e-3;
                    const float original_weights = weights.at(input_left + input_right * size);

                    weights.at(input_left + input_right * size) = original_weights + sensitivity;
                        diffdp::EntropyRegularizedBinaryPhraseStructure parser2(size);
                        parser2.forward(
                        [&](const unsigned left, const unsigned right) -> float
                        {
                            return weights.at(left + right * size);
                        }
                    );
                    const float output_a = parser2.output(output_left, output_right);

                    weights.at(input_left + input_right * size) = original_weights - sensitivity;
                        parser2.forward(
                        [&](const unsigned left, const unsigned right) -> float
                        {
                            return weights.at(left + right * size);
                        }
                    );
                    const float output_b = parser2.output(output_left, output_right);

                    // restore
                    weights.at(input_left + input_right * size) = original_weights;

                    const double estimated_gradient = (output_a - output_b) / (2.f * sensitivity);
                    BOOST_CHECK(check_grad(computed_gradient, estimated_gradient));
                }
            }
        }
    }
}
