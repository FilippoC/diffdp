#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>
#include <random>

#include "dynet/nodes-def-macros.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/tensor-eigen.h"
#include "diffdp/dynet/matrix_tree_theorem.h"
#include "dytools/functions/rooted_arborescence_marginals.h"

int main(int argc, char* argv[])
{
    const auto size = 3;
    dynet::initialize(argc, argv);

    std::vector<float> v_weights(size * size, 0.f);
    v_weights.at(0 + 1 * size) = 1.f;
    v_weights.at(0 + 2 * size) = 4.f;
    v_weights.at(1 + 2 * size) = 1.f;
    v_weights.at(2 + 1 * size) = 1.f;

    dynet::ComputationGraph cg;


    const auto e_weights = dynet::input(cg, {size, size}, v_weights);

    std::vector<unsigned> sizes{2};
    const auto e_marginals = dytools::rooted_arborescence_marginals(cg, e_weights, sizes);

    const auto v_output = as_vector(cg.forward(e_marginals));
    for (unsigned i = 0  ; i < size ; ++i)
    {
        for (unsigned j = 0 ; j < size ; ++j)
            std::cerr << v_output.at(i + j * size) << "\t";
        std::cerr << "\n";
    }
}
