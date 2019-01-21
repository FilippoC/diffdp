#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>
#include <random>

#include "diffdp/chart.h"

#include "diffdp/math.h"
#include "diffdp/deduction_operations.h"
#include "diffdp/algorithm/eisner.h"

int main(int argc, char* argv[])
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    const unsigned size = 5;
    std::vector<float> weights(size * size);
    for (unsigned i = 0 ; i < size ; ++i)
        weights.at(i) = distribution(generator);

    std::cerr << "Entropy reg / Alg diff\n";
    diffdp::EntropyRegularizedEisner entrop_reg_eisner(size);
    entrop_reg_eisner.forward(
            [&] (unsigned head, unsigned mod) -> float
            {
                return weights.at(head + mod * size);
            }
    );
    diffdp::AlgorithmicDifferentiableEisner algo_diff_eisner(size);
    algo_diff_eisner.forward(
            [&] (unsigned head, unsigned mod) -> float
            {
                return weights.at(head + mod * size);
            });

    for (unsigned head = 0 ; head < size ; ++head)
    {
        for (unsigned mod = 1 ; mod < size ; ++mod)
        {
            if (head == mod)
                continue;

            std::cerr
                << entrop_reg_eisner.output(head, mod)
                << "\t"
                << algo_diff_eisner.output(head, mod)
                << "\n";
        }
    }
}