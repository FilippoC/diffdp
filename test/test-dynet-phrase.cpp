/**
 * TODO split in two different test cases
 */
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "DynetPhrase"

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <vector>

#include "dynet/expr.h"
#include "dynet/param-init.h"
#include "dynet/grad-check.h"

#include "diffdp/dynet/binary_phrase.h"

BOOST_AUTO_TEST_CASE(test_dynet_phrase)
        {
                const unsigned size = 10u;

                int argc = 1;
                char **argv;
                dynet::initialize(argc, argv);

                dynet::ParameterCollection pc;

                std::vector<float> weights(size * size);
                //for (unsigned i = 0 ; i < weights.size() ; ++i)
                //    weights.at(i) = (float) i;
                auto p_weights = pc.add_parameters(dynet::Dim({size, size}), dynet::ParameterInitFromVector(weights));

                {
                    for (unsigned head = 0u; head < size; ++head)
                    {
                        for (unsigned mod = 0u; mod < size; ++mod)
                        {
                            dynet::ComputationGraph cg;
                            auto e_weights = dynet::parameter(cg, p_weights);
                            auto e_arcs = dynet::algorithmic_differentiable_binary_phrase_structure(
                                    e_weights,
                                    diffdp::DiscreteMode::ForwardRegularized
                            );
                            auto e_output = dynet::strided_select(
                                    e_arcs,
                                    {(int) 1u, (int) 1u},
                                    {(int) head, (int) mod},
                                    {(int) head + 1, (int) mod + 1} // not included
                            );

                            BOOST_CHECK(check_grad(pc, e_output, 0));
                        }
                    }
                }
                {

                    for (unsigned head = 0u; head < size; ++head)
                    {
                        for (unsigned mod = 0u; mod < size; ++mod)
                        {
                            dynet::ComputationGraph cg;
                            auto e_weights = dynet::parameter(cg, p_weights);

                            auto e_arcs = dynet::entropy_regularized_binary_phrase_structure(
                                    e_weights,
                                    diffdp::DiscreteMode::ForwardRegularized
                            );
                            auto e_output = dynet::strided_select(
                                    e_arcs,
                                    {(int) 1u, (int) 1u},
                                    {(int) head, (int) mod},
                                    {(int) head + 1, (int) mod + 1} // not included
                            );

                            BOOST_CHECK(check_grad(pc, e_output, 0));
                        }
                    }
                }
        }