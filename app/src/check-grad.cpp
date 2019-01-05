#include <vector>
#include <iostream>
#include <stdlib.h>

#include "dynet/dynet.h"
#include "dynet/grad-check.h"
#include "dynet/param-init.h"
#include "diffdp/continuous_eisner.h"

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Wrong number of arguments" << std::endl;
        return 1;
    }

    char *pCh;
    const unsigned size = strtoul(argv[1], &pCh, 10);
    const unsigned batch = 1;

    // Ensure argument was okay.

    if ((pCh == argv[1]) || (*pCh != '\0'))
    {
        std::cerr << "Invalid argument: the size of the sentence should be provided" << std::endl;
        return 1;
    }

    std::vector<dynet::real> weights(batch * size * size);
    for (unsigned b = 0 ; b < batch ; ++b)
        for (unsigned i = 0 ; i < size ; ++i)
            for (unsigned j = 0 ; j < size ; ++j)
                weights.at(b * size * size + i + j * size) = batch + i + j * size + 1;

    dynet::initialize(argc, argv);
    dynet::ParameterCollection pc;
    auto p_weights = pc.add_parameters({size * size * batch}, dynet::ParameterInitFromVector(weights));

    for (bool with_root_arcs : {true, false})
    {
        std::cerr << "Checking " << (with_root_arcs ? "with" : "without") << " root arcs\n";
        for (unsigned i = 0u ; i < 4u ; ++i)
        {
            dynet::ComputationGraph cg;
            //cg.set_immediate_compute(true);
            //cg.set_check_validity(true);

            auto e_weights = dynet::parameter(cg, p_weights);
            e_weights = dynet::reshape(e_weights, dynet::Dim({size, size}, batch));

            std::vector<unsigned> b_sizes;
            b_sizes.push_back(2);
            b_sizes.push_back(3);
            dynet::Expression e_arcs;
            if (i == 0u)
            {
                std::cerr << "Input/output modes: Adjacency/Adjacency\n";
                e_arcs = dynet::continuous_eisner(
                        e_weights,
                        1.0,
                        diffdp::DiscreteMode::ForwardRegularized,
                        diffdp::GraphMode::Adjacency, diffdp::GraphMode::Adjacency,
                        with_root_arcs
                        //&b_sizes
                );
            }
            else if (i == 1u)
            {
                std::cerr << "Input/output modes: Compact/Compact\n";
                e_arcs = dynet::continuous_eisner(
                        e_weights,
                        1.0,
                        diffdp::DiscreteMode::ForwardRegularized,
                        diffdp::GraphMode::Compact, diffdp::GraphMode::Compact,
                        with_root_arcs
                );
            }
            else if (i == 2u)
            {
                std::cerr << "Input/output modes: Adjacency/Compact\n";
                e_arcs = dynet::continuous_eisner(
                        e_weights,
                        1.0,
                        diffdp::DiscreteMode::ForwardRegularized,
                        diffdp::GraphMode::Adjacency, diffdp::GraphMode::Compact,
                        with_root_arcs
                );
            }
            else
            {
                std::cerr << "Input/output modes: Compact/Adjacency\n";
                e_arcs = dynet::continuous_eisner(
                        e_weights,
                        1.0,
                        diffdp::DiscreteMode::ForwardRegularized,
                        diffdp::GraphMode::Compact, diffdp::GraphMode::Adjacency,
                        with_root_arcs
                );
            }
            for (unsigned b = 0u ; b < batch ; ++ b)
            {
                for (unsigned head = 0u ; head < e_arcs.dim().rows() ; ++ head)
                {
                    for (unsigned mod = 0u ; mod < e_arcs.dim().cols() ; ++ mod)
                    {
                        auto e_output = dynet::strided_select(
                                e_arcs,
                                {(int) 1u, (int) 1u, (int) 1u},
                                {(int) head, (int) mod, (int) b},
                                {(int) head+1, (int) mod+1, (int) b+1} // not included
                        )
                        ;
                        auto v = check_grad(pc, e_output, 0);
                        if (v == 0)
                        {
                            std::cerr << "failed: " << head << "->" << mod << std::endl;
                            check_grad(pc, e_output, 2);
                            return 1;
                        }
                    }
                }
            }
        }
    }
    std::cerr << "All tests were successful\n";
    return 0;
}