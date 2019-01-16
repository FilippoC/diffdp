#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EntropyRegularizedEisner"

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <vector>

#include "diffdp/algorithm/eisner.h"
#include "diffdp/math.h"
#include "dynet/expr.h"

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

BOOST_AUTO_TEST_CASE(test_softmax)
{
    int argc = 1;
    char **argv;
    dynet::initialize(argc, argv);

    std::vector<float> input(10);
    std::vector<float> output(10);
    std::vector<float> input_grad(10);
    std::vector<float> output_grad(10);
    for (unsigned i = 0 ; i < input.size() ; ++i)
    input.at(i) = i;

    {
        diffdp::softmax(output.begin(), input.begin(), input.size());

        dynet::ComputationGraph cg;
        auto e_output = dynet::softmax(dynet::input(cg, {10}, input));
        auto dynet_output = as_vector(cg.forward(e_output));

        for (unsigned i = 0 ; i < 10 ; ++i)
            BOOST_CHECK(check_grad(output.at(i), dynet_output.at(i)));
    }

    for (unsigned input_id = 0 ; input_id < input.size() ; ++input_id)
    {
        for (unsigned output_id = 0 ; output_id < input.size() ; ++output_id)
        {

            // compute gradient

            std::fill(input_grad.begin(), input_grad.end(), 0.f);
            std::fill(output_grad.begin(), output_grad.end(), 0.f);
            //std::fill(output.begin(), output.end(), 0.f);

            diffdp::softmax(output.begin(), input.begin(), input.size());
            output_grad.at(output_id) = 1.f;
            diffdp::backprop_softmax(
                    input_grad.begin(), output_grad.begin(),
                    input.begin(), output.begin(),
                    input.size()
            );
            const float computed_gradient = input_grad.at(input_id);

            // dynet gradient
            dynet::ComputationGraph cg;

            auto e_input = dynet::input(cg, {10}, input);
            auto e_softmax = dynet::softmax(e_input);
            auto e_output = dynet::pick(e_softmax, output_id);
            cg.forward(e_output);
            cg.backward(e_output, true);

            auto dynet_gradient_all = as_vector(e_input.gradient());
            float dynet_gradient = dynet_gradient_all.at(input_id);

            BOOST_CHECK(check_grad(computed_gradient, dynet_gradient));
        }

    }
}