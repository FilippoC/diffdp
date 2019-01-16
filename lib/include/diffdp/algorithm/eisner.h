#pragma once

#include <cassert>
#include <functional>
#include <memory>

#include "diffdp/chart.h"
#include "diffdp/deduction_operations.h"

namespace diffdp
{

struct EisnerChart
{
    const unsigned size;
    const unsigned size_3d;
    const unsigned size_2d;
    float* _memory = nullptr;
    const bool _erase_memory;

    Tensor3D<float>
        a_cleft, a_cright, a_uleft, a_uright,
        b_cleft, b_cright, b_uleft, b_uright;

    Matrix<float>
        c_cleft, c_cright, c_uleft, c_uright,
        soft_c_cleft, soft_c_cright, soft_c_uleft, soft_c_uright
        ;

    EisnerChart(unsigned size);
    EisnerChart(unsigned size, float* mem);
    ~EisnerChart();

    void zeros();

    static std::size_t required_memory(const unsigned size);
    static unsigned required_cells(const unsigned size);
};

/*
 * Continuous relaxation of "Differentiable Perturb-and-Parse: Semi-Supervised Parsing with a Structured Variational Autoencoder, Corro & Titov"
 */
struct AlgorithmicDifferentiableEisner
{
    unsigned _size;

    std::shared_ptr<EisnerChart> chart_forward;
    std::shared_ptr<EisnerChart> chart_backward;

    explicit AlgorithmicDifferentiableEisner(const unsigned t_size);
    AlgorithmicDifferentiableEisner(std::shared_ptr<EisnerChart> chart_forward, std::shared_ptr<EisnerChart> chart_backward);

    template<class Functor>
    void forward(Functor&& weight_callback);

    template<class Functor>
    void backward(Functor&& gradient_callback);

    static void forward_maximize(std::shared_ptr<EisnerChart>& chart_forward);
    static void forward_backtracking(std::shared_ptr<EisnerChart>& chart_forward);

    static void backward_maximize(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward);
    static void backward_backtracking(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward);

    float output(const unsigned head, const unsigned mod) const;
    float gradient(const unsigned head, const unsigned mod) const;

    unsigned size() const;
};



/*
 * Continuous relaxation of "Differentiable Dynamic Programming for Structured Prediction and Attention, Mensch & Blondel"
 * This is equivalent to structured attention (i.e. marginalization),
 * but it has better numerically stability in practice (i.e. no underflow/overflow issue)
 */
struct EntropyRegularizedEisner
{
    unsigned _size;

    std::shared_ptr<EisnerChart> chart_forward;
    std::shared_ptr<EisnerChart> chart_backward;

    explicit EntropyRegularizedEisner(const unsigned t_size);
    EntropyRegularizedEisner(std::shared_ptr<EisnerChart> chart_forward, std::shared_ptr<EisnerChart> chart_backward);


    template<class Functor>
    void forward(Functor&& weight_callback);

    template<class Functor>
    void backward(Functor&& gradient_callback);

    static void forward_maximize(std::shared_ptr<EisnerChart>& chart_forward);
    static void forward_backtracking(std::shared_ptr<EisnerChart>& chart_forward);

    //static void backward_maximize(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward);
    //static void backward_backtracking(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward);

    float output(const unsigned head, const unsigned mod) const;
    float gradient(const unsigned head, const unsigned mod) const;

    unsigned size() const;
};


// templates implementations

template<class Functor>
void AlgorithmicDifferentiableEisner::forward(Functor&& weight_callback)
{
    const unsigned size = chart_forward->size;

    chart_forward->zeros(); // we could skip some zeros here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 1; j < size; ++j)
        {
            if (i < j)
                chart_forward->c_uright(i, j) = weight_callback(i, j);
            else if (j < i)
                chart_forward->c_uleft(j, i) = weight_callback(i, j);
        }
    }

    AlgorithmicDifferentiableEisner::forward_maximize(chart_forward);
    AlgorithmicDifferentiableEisner::forward_backtracking(chart_forward);
}

template<class Functor>
void AlgorithmicDifferentiableEisner::backward(Functor&& gradient_callback)
{
    const unsigned size = chart_forward->size;

    chart_backward->zeros();
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 1; j < size; ++j)
        {
            if (i < j)
                chart_backward->soft_c_uright(i, j) = gradient_callback(i, j);
            else if (j < i)
                chart_backward->soft_c_uleft(j, i) = gradient_callback(i, j);
        }
    }

    AlgorithmicDifferentiableEisner::backward_backtracking(chart_forward, chart_backward);
    AlgorithmicDifferentiableEisner::backward_maximize(chart_forward, chart_backward);
}

template<class Functor>
void EntropyRegularizedEisner::forward(Functor&& weight_callback)
{
    const unsigned size = chart_forward->size;

    // this initialization seems ok, but check why it works!
    chart_forward->zeros(); // we could skip some zeros here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 1; j < size; ++j)
        {
            if (i < j)
                chart_forward->c_uright(i, j) = weight_callback(i, j);
            else if (j < i)
                chart_forward->c_uleft(j, i) = weight_callback(i, j);
        }
    }

    EntropyRegularizedEisner::forward_maximize(chart_forward);
    EntropyRegularizedEisner::forward_backtracking(chart_forward);
}

template<class Functor>
void EntropyRegularizedEisner::backward(Functor&& gradient_callback)
{
    const unsigned size = chart_forward->size;

    // check if this init is correct
    chart_backward->zeros();
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 1; j < size; ++j)
        {
            if (i < j)
                chart_backward->soft_c_uright(i, j) = gradient_callback(i, j);
            else if (j < i)
                chart_backward->soft_c_uleft(j, i) = gradient_callback(i, j);
        }
    }

    // backpropagate throught backtracking
    for (unsigned l = 1; l < size ; ++l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            if (i > 0u)
            {
                diffdp::backward_backtracking(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uleft(i, j),
                        chart_forward->b_uleft.iter3(i, j, i),

                        chart_backward->soft_c_cright.iter2(i, i), chart_backward->soft_c_cleft.iter1(i + 1, j),
                        &chart_backward->soft_c_uleft(i, j),
                        chart_backward->b_uleft.iter3(i, j, i),

                        l
                );
            }

            diffdp::backward_backtracking(
                    chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                    chart_forward->soft_c_uright(i, j),
                    chart_forward->b_uright.iter3(i, j, i),

                    chart_backward->soft_c_cright.iter2(i, i), chart_backward->soft_c_cleft.iter1(i + 1, j),
                    &chart_backward->soft_c_uright(i, j),
                    chart_backward->b_uright.iter3(i, j, i),

                    l
            );

            if (i > 0u)
            {
                diffdp::backward_backtracking(
                        chart_forward->soft_c_cleft.iter2(i, i), chart_forward->soft_c_uleft.iter1(i, j),
                        chart_forward->soft_c_cleft(i, j),
                        chart_forward->b_cleft.iter3(i, j, i),

                        chart_backward->soft_c_cleft.iter2(i, i), chart_backward->soft_c_uleft.iter1(i, j),
                        &chart_backward->soft_c_cleft(i, j),
                        chart_backward->b_cleft.iter3(i, j, i),

                        l
                );
            }

            diffdp::backward_backtracking(
                    chart_forward->soft_c_uright.iter2(i, i+1), chart_forward->soft_c_cright.iter1(i+1, j),
                    chart_forward->soft_c_cright(i, j),
                    chart_forward->b_cright.iter3(i, j, i + 1),

                    chart_backward->soft_c_uright.iter2(i, i+1), chart_backward->soft_c_cright.iter1(i+1, j),
                    &chart_backward->soft_c_cright(i, j),
                    chart_backward->b_cright.iter3(i, j, i + 1),

                    l
            );
        }
    }
    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            if (i > 0u)
            {
                backward_entropy_reg(
                        chart_forward->c_cleft.iter2(i, i), chart_forward->c_uleft.iter1(i, j),
                        chart_forward->a_cleft.iter3(i, j, i),
                        chart_forward->b_cleft.iter3(i, j, i),

                        chart_backward->c_cleft.iter2(i, i), chart_backward->c_uleft.iter1(i, j),
                        chart_backward->c_cleft(i, j),
                        chart_backward->a_cleft.iter3(i, j, i),
                        chart_backward->b_cleft.iter3(i, j, i),

                        l
                );
            }

            backward_entropy_reg(
                    chart_forward->c_uright.iter2(i, i + 1), chart_forward->c_cright.iter1(i + 1, j),
                    chart_forward->a_cright.iter3(i, j, i + 1),
                    chart_forward->b_cright.iter3(i, j, i + 1),

                    chart_backward->c_uright.iter2(i, i + 1), chart_backward->c_cright.iter1(i + 1, j),
                    chart_backward->c_cright(i, j),
                    chart_backward->a_cright.iter3(i, j, i + 1),
                    chart_backward->b_cright.iter3(i, j, i + 1),

                    l
            );

            if (i > 0u)
            {
                backward_entropy_reg(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uleft.iter3(i, j, i),
                        chart_forward->b_uleft.iter3(i, j, i),

                        chart_backward->c_cright.iter2(i, i), chart_backward->c_cleft.iter1(i + 1, j),
                        chart_backward->c_uleft(i, j),
                        chart_backward->a_uleft.iter3(i, j, i),
                        chart_backward->b_uleft.iter3(i, j, i),

                        l
                );
            }

            backward_entropy_reg(
                    chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                    chart_forward->a_uright.iter3(i, j, i),
                    chart_forward->b_uright.iter3(i, j, i),

                    chart_backward->c_cright.iter2(i, i), chart_backward->c_cleft.iter1(i + 1, j),
                    chart_backward->c_uright(i, j),
                    chart_backward->a_uright.iter3(i, j, i),
                    chart_backward->b_uright.iter3(i, j, i),

                    l
            );
        }
    }
}




}

