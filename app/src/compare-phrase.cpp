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
#include "diffdp/algorithm/binary_phrase.h"

int main(int argc, char* argv[])
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);

    const unsigned size = 5;
    std::vector<float> weights(size * size);
    for (unsigned i = 0 ; i < weights.size() ; ++i)
        weights.at(i) = distribution(generator);

    diffdp::AlgorithmicDifferentiableBinaryPhraseStructure algo_diff(size);
    algo_diff.forward(
            [&] (unsigned head, unsigned mod) -> float
            {
                return weights.at(head + mod * size);
            });

    for (unsigned left = 0 ; left < size ; ++left)
    {
        for (unsigned right = left+1 ; right < size ; ++right)
        {
            std::cerr
                    << left << "," << right
                    << "\t"
                    << algo_diff.output(left, right)
                    << "\n";
        }
    }
}