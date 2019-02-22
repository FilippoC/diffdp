#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>
#include <random>

#include "dynet/expr.h"
#include "diffdp/dynet/matrix_tree_theorem.h"

int main(int argc, char* argv[])
{
    dynet::initialize(argc, argv);

    const unsigned size = 3;
    std::vector<float> v_input(3 * 3, 1.f);

    dynet::ComputationGraph cg;
    auto e_input = dynet::input(cg, {size, size}, v_input);
    auto e_output = dynet::matrix_tree_theorem(e_input);

    auto v_output = as_vector(cg.forward(e_output));

    for (unsigned i = 0 ; i < size ; ++i)
    {
        for (unsigned j = 0 ; j < size ; ++j)
        {
            std::cout << v_output.at(i + j * size) << "\t";
        }
        std::cout << "\n";
    }
}