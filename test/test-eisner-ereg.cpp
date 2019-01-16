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

BOOST_AUTO_TEST_CASE(first_order_gradient, * utf::tolerance(1e-2f))
{
        const unsigned size = 10;
        const float sensitivity = 1e-2;

        std::vector<float> weights(size * size);
        for (unsigned i = 0 ; i < size ; ++i)
            weights.at(i) = size;

        diffdp::EntropyRegularizedEisner parser(size);
        parser.forward(
            [&] (unsigned head, unsigned mod) -> float
                    {
                        return weights.at(head + mod * size);
                    }
            );

        for (unsigned head = 0 ; head < size ; ++head)
        {
            for (unsigned mod = 1; mod < size ; ++mod)
            {
                if (head == mod)
                    continue;

                const float computed_arc = parser.output(head, mod);

                // estimate the gradient
                const float original_weights = weights.at(head + mod * size);

                weights.at(head + mod * size) = original_weights + sensitivity;
                diffdp::EntropyRegularizedEisner parser2(size);
                parser2.forward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            return weights.at(head + mod * size);
                        }
                );
                const float output_a = parser2.chart_forward->c_cright(0, size-1);

                weights.at(head + mod * size) = original_weights - sensitivity;
                parser2.forward(
                        [&] (const unsigned head, const unsigned mod) -> float
                        {
                            return weights.at(head + mod * size);
                        }
                );
                const float output_b = parser2.chart_forward->c_cright(0, size-1);

                // restore
                weights.at(head + mod * size) = original_weights;

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

    diffdp::EntropyRegularizedEisner parser(size);
        parser.forward(
        [&] (unsigned head, unsigned mod) -> float
        {
            return weights.at(head + mod * size);
        }
    );

    for (unsigned input_head = 0 ; input_head < size ; ++input_head)
    {
        for (unsigned input_mod = 1; input_mod < size ; ++input_mod)
        {
            if (input_head == input_mod)
                continue;

            for (unsigned output_head = 0 ; output_head < size ; ++output_head)
            {
                for (unsigned output_mod = 1; output_mod < size; ++output_mod)
                {
                    if (output_head == output_mod)
                        continue;

                    parser.backward(
                            [&](const unsigned head, const unsigned mod)
                            {
                                if (head == output_head && mod == output_mod)
                                    return 1.f;
                                else
                                    return 0.f;
                            }
                    );
                    const float computed_gradient = parser.gradient(input_head, input_mod);

                    // estimate the gradient
                    const float sensitivity = 1e-3;
                    const float original_weights = weights.at(input_head + input_mod * size);

                    weights.at(input_head + input_mod * size) = original_weights + sensitivity;
                    diffdp::EntropyRegularizedEisner parser2(size);
                    parser2.forward(
                            [&](const unsigned head, const unsigned mod) -> float
                            {
                                return weights.at(head + mod * size);
                            }
                    );
                    const float output_a = parser2.output(output_head, output_mod);

                    weights.at(input_head + input_mod * size) = original_weights - sensitivity;
                    parser2.forward(
                            [&](const unsigned head, const unsigned mod) -> float
                            {
                                return weights.at(head + mod * size);
                            }
                    );
                    const float output_b = parser2.output(output_head, output_mod);

                    // restore
                    weights.at(input_head + input_mod * size) = original_weights;

                    const double estimated_gradient = (output_a - output_b) / (2.f * sensitivity);

                    BOOST_CHECK(check_grad(computed_gradient, estimated_gradient));
                }
            }
        }
    }
}