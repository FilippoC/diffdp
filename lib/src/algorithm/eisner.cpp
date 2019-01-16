#include "diffdp/algorithm/eisner.h"

namespace diffdp
{

EisnerChart::EisnerChart(unsigned size) :
    size(size),
    size_3d(size*size*size),
    size_2d(size*size),
    _memory(new float[size_3d * 8 + size_2d * 8]),
    _erase_memory(true),
    a_cleft(size, _memory),
    a_cright(size, _memory + 1u*size_3d),
    a_uleft(size, _memory + 2u*size_3d),
    a_uright(size, _memory + 3u*size_3d),
    b_cleft(size, _memory + 4u*size_3d),
    b_cright(size, _memory + 5u*size_3d),
    b_uleft(size, _memory + 6u*size_3d),
    b_uright(size, _memory + 7u*size_3d),
    c_cleft(size, _memory + 8u*size_3d),
    c_cright(size, _memory + 8u*size_3d + 1u*size_2d),
    c_uleft(size, _memory + 8u*size_3d + 2u*size_2d),
    c_uright(size, _memory + 8u*size_3d + 3u*size_2d),
    soft_c_cleft(size, _memory + 8u*size_3d + 4u*size_2d),
    soft_c_cright(size, _memory + 8u*size_3d + 5u*size_2d),
    soft_c_uleft(size, _memory + 8u*size_3d + 6u*size_2d),
    soft_c_uright(size, _memory + 8u*size_3d + 7u*size_2d)
{}

EisnerChart::EisnerChart(unsigned size, float* mem) :
    size(size),
    size_3d(size*size*size),
    size_2d(size*size),
    _memory(mem),
    _erase_memory(false),
    a_cleft(size, mem),
    a_cright(size, mem + 1u*size_3d),
    a_uleft(size, mem + 2u*size_3d),
    a_uright(size, mem + 3u*size_3d),
    b_cleft(size, mem + 4u*size_3d),
    b_cright(size, mem + 5u*size_3d),
    b_uleft(size, mem + 6u*size_3d),
    b_uright(size, mem + 7u*size_3d),
    c_cleft(size, mem + 8u*size_3d),
    c_cright(size, mem + 8u*size_3d + 1u*size_2d),
    c_uleft(size, mem + 8u*size_3d + 2u*size_2d),
    c_uright(size, mem + 8u*size_3d + 3u*size_2d),
    soft_c_cleft(size, mem + 8u*size_3d + 4u*size_2d),
    soft_c_cright(size, mem + 8u*size_3d + 5u*size_2d),
    soft_c_uleft(size, mem + 8u*size_3d + 6u*size_2d),
    soft_c_uright(size, mem + 8u*size_3d + 7u*size_2d)
{}

EisnerChart::~EisnerChart()
{
    if (_erase_memory)
        delete[] _memory;
}

void EisnerChart::zeros()
{
    std::fill(_memory, _memory + size_3d * 8 + size_2d * 8, float{});
}

std::size_t EisnerChart::required_memory(const unsigned size)
{
    return
            8 * Tensor3D<float>::required_memory(size)
            + 8 * Matrix<float>::required_memory(size)
            ;
}

unsigned EisnerChart::required_cells(const unsigned size)
{
    return
            8 * Tensor3D<float>::required_cells(size)
            + 8 * Matrix<float>::required_cells(size)
            ;
}


AlgorithmicDifferentiableEisner::AlgorithmicDifferentiableEisner(const unsigned t_size) :
    _size(t_size),
    chart_forward(std::make_shared<EisnerChart>(_size)),
    chart_backward(std::make_shared<EisnerChart>(_size))
{}

AlgorithmicDifferentiableEisner::AlgorithmicDifferentiableEisner(std::shared_ptr<EisnerChart> chart_forward, std::shared_ptr<EisnerChart> chart_backward) :
        _size(chart_forward->size),
        chart_forward(chart_forward),
        chart_backward(chart_backward)
{}

void AlgorithmicDifferentiableEisner::forward_maximize(std::shared_ptr<EisnerChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            // use += because we initialized them with arc weights
            chart_forward->c_uright(i, j) += forward_algorithmic_softmax(
                    chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                    chart_forward->a_uright.iter3(i, j, i),
                    chart_forward->b_uright.iter3(i, j, i),
                    l
            );

            if (i > 0u) // because the root cannot be the modifier
            {
                chart_forward->c_uleft(i, j) += forward_algorithmic_softmax(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uleft.iter3(i, j, i),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }

            chart_forward->c_cright(i, j) = forward_algorithmic_softmax(
                    chart_forward->c_uright.iter2(i, i + 1), chart_forward->c_cright.iter1(i + 1, j),
                    chart_forward->a_cright.iter3(i, j, i + 1),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                chart_forward->c_cleft(i, j) = forward_algorithmic_softmax(
                        chart_forward->c_cleft.iter2(i, i), chart_forward->c_uleft.iter1(i, j),
                        chart_forward->a_cleft.iter3(i, j, i),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}

void AlgorithmicDifferentiableEisner::forward_backtracking(std::shared_ptr<EisnerChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    chart_forward->soft_c_cright(0, size - 1) = 1.0f;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            diffdp::forward_backtracking(
                    chart_forward->soft_c_uright.iter2(i, i + 1), chart_forward->soft_c_cright.iter1(i + 1, j),
                    chart_forward->soft_c_cright(i, j),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cleft.iter2(i, i), chart_forward->soft_c_uleft.iter1(i, j),
                        chart_forward->soft_c_cleft(i, j),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }

            diffdp::forward_backtracking(
                    chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                    chart_forward->soft_c_uright(i, j),
                    chart_forward->b_uright.iter3(i, j, i),
                    l
            );


            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uleft(i, j),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}

void AlgorithmicDifferentiableEisner::backward_backtracking(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

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

}

void AlgorithmicDifferentiableEisner::backward_maximize(std::shared_ptr<EisnerChart>& chart_forward, std::shared_ptr<EisnerChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            if (i > 0u)
            {
                backward_algorithmic_softmax(
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

            backward_algorithmic_softmax(
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
                backward_algorithmic_softmax(
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

            backward_algorithmic_softmax(
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

unsigned AlgorithmicDifferentiableEisner::size() const
{
    return _size;
}

float AlgorithmicDifferentiableEisner::output(const unsigned head, const unsigned mod) const
{
    if (head < mod)
        return chart_forward->soft_c_uright(head, mod);
    else if (mod < head)
        return chart_forward->soft_c_uleft(mod, head);
    else
        return std::nanf("");
}

float AlgorithmicDifferentiableEisner::gradient(const unsigned head, const unsigned mod) const
{
    if (head < mod)
        return chart_backward->c_uright(head, mod);
    else if (mod < head)
        return chart_backward->c_uleft(mod, head);
    else
        return std::nanf("");
}




EntropyRegularizedEisner::EntropyRegularizedEisner(const unsigned t_size) :
        _size(t_size),
        chart_forward(std::make_shared<EisnerChart>(_size)),
        chart_backward(std::make_shared<EisnerChart>(_size))
{}

EntropyRegularizedEisner::EntropyRegularizedEisner(std::shared_ptr<EisnerChart> chart_forward, std::shared_ptr<EisnerChart> chart_backward) :
        _size(chart_forward->size),
        chart_forward(chart_forward),
        chart_backward(chart_backward)
{}


void EntropyRegularizedEisner::forward_maximize(std::shared_ptr<EisnerChart>& chart_forward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            chart_forward->c_uright(i, j) += forward_entropy_reg(
                    chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                    chart_forward->a_uright.iter3(i, j, i),
                    chart_forward->b_uright.iter3(i, j, i),
                    l
            );

            if (i > 0u) // because the root cannot be the modifier
            {
                chart_forward->c_uleft(i, j) += forward_entropy_reg(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uleft.iter3(i, j, i),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }

            chart_forward->c_cright(i, j) = forward_entropy_reg(
                    chart_forward->c_uright.iter2(i, i + 1), chart_forward->c_cright.iter1(i + 1, j),
                    chart_forward->a_cright.iter3(i, j, i + 1),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                chart_forward->c_cleft(i, j) = forward_entropy_reg(
                        chart_forward->c_cleft.iter2(i, i), chart_forward->c_uleft.iter1(i, j),
                        chart_forward->a_cleft.iter3(i, j, i),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}

void EntropyRegularizedEisner::forward_backtracking(std::shared_ptr<EisnerChart>& chart_forward)
{
    const unsigned size = chart_forward->size;

    chart_forward->soft_c_cright(0, size - 1) = 1.0f;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            diffdp::forward_backtracking(
                    chart_forward->soft_c_uright.iter2(i, i + 1), chart_forward->soft_c_cright.iter1(i + 1, j),
                    chart_forward->soft_c_cright(i, j),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cleft.iter2(i, i), chart_forward->soft_c_uleft.iter1(i, j),
                        chart_forward->soft_c_cleft(i, j),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }

            diffdp::forward_backtracking(
                    chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                    chart_forward->soft_c_uright(i, j),
                    chart_forward->b_uright.iter3(i, j, i),
                    l
            );


            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uleft(i, j),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}


unsigned EntropyRegularizedEisner::size() const
{
    return _size;
}

float EntropyRegularizedEisner::output(const unsigned head, const unsigned mod) const
{
    if (head < mod)
        return chart_forward->soft_c_uright(head, mod);
    else if (mod < head)
        return chart_forward->soft_c_uleft(mod, head);
    else
        return std::nanf("");
}

float EntropyRegularizedEisner::gradient(const unsigned head, const unsigned mod) const
{
    if (head < mod)
        return chart_backward->c_uright(head, mod);
    else if (mod < head)
        return chart_backward->c_uleft(mod, head);
    else
        return std::nanf("");
}

}