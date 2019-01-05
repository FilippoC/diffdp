# Differentiable Perturb-and-Parse operator

This repository contains the code for the continuous relaxation of the Eisner algorithm presented in:
"Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder",
Caio Corro, Ivan Titov
See: https://openreview.net/forum?id=BJlgNh0qKQ

To cite:
@InProceedings{perturb-and-parse,
  author = "Corro, Caio and Titov, Ivan",
  title = "Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder",
  booktitle = "Proceedings of Seventh International Conference on Learning Representations",
  year = "2019"
}

The full VAE code and model will be released after the proceedings release.
If any question, please contact me at following mail address: c.f.corro@uva.nl


## Usage

```
#include "diffdp/continuous_eisner.h"

auto arcs = dynet::continuous_eisner(
        weights, // input : matrix of arc weights
        temp, // temperature of the relaxation
        diffdp::DiscreteMode::ForwardRegularized, // relaxation mode
        diffdp::GraphMode::Adjacency, diffdp::GraphMode::Adjacency, // input/output format
        true // set to false to remove root arcs
);
```


## Arguments

The following arguments must be provided:
1. the arc-factored weights of dependencies
2. the temperature of the relaxation
3. the relaxation mode: diffdp::DiscreteMode::BackwardRegularized output the discrete structure and us
   the relaxtion only for backward, diffdp::DiscreteMode::ForwardRegularized use the relaxation during forward
4. the input format: diffdp::GraphMode::Adjacency use a adjacency matrix as input format, i.e. the main diagonal
   represent self connections and is never used, diffdp::GraphMode::Compact use the main diagonal to represent the weights
   of root dependencies
5. the output format
6. set to false to remove root arcs from the output


## Batching

This computational node can be used with mini-batches.
However, it does not implement the auto-batch functionnality of Dynet, so mini-batches should be constructed manually.

If sentences are of different sizes, a pointer of type "std::vector<unsigned>*" can be given as the last argument.
This compatible with static graph (i.e. each forward call will check sentence sizes in the vector)

WARNING: the size of batch input *must not* include the root node.

## TODO
- The memory usage could be divided by 2
- There is a lot of code that could be moved from headers to cpp files