#include "diffdp/algorithm/binary_phrase.h"

namespace diffdp
{

BinaryPhraseStructureChart::BinaryPhraseStructureChart(unsigned size) :
        size(size),
        size_3d(size*size*size),
        size_2d(size*size),
        _memory(new float[size_3d * 2 + size_2d * 2]),
        _erase_memory(true),
        split_weights(size, _memory),
        backptr(size, _memory + 1u*size_3d),
        weight(size, _memory + 2u*size_3d),
        soft_selection(size, _memory + 2u*size_3d + 1u*size_2d)
{}

BinaryPhraseStructureChart::BinaryPhraseStructureChart(unsigned size, float* mem) :
        size(size),
        size_3d(size*size*size),
        size_2d(size*size),
        _memory(mem),
        _erase_memory(false),
        split_weights(size, _memory),
        backptr(size, _memory + 1u*size_3d),
        weight(size, _memory + 2u*size_3d),
        soft_selection(size, _memory + 2u*size_3d + 1u*size_2d)
{}

BinaryPhraseStructureChart::~BinaryPhraseStructureChart()
{
    if (_erase_memory)
        delete[] _memory;
}

void BinaryPhraseStructureChart::zeros()
{
    std::fill(_memory, _memory + required_cells(size), float{});
}

std::size_t BinaryPhraseStructureChart::required_memory(const unsigned size)
{
    return
            2 * Tensor3D<float>::required_memory(size)
            + 2 * Matrix<float>::required_memory(size)
            ;
}

unsigned BinaryPhraseStructureChart::required_cells(const unsigned size)
{
    return
            2 * Tensor3D<float>::required_cells(size)
            + 2 * Matrix<float>::required_cells(size)
            ;
}


AlgorithmicDifferentiableBinaryPhraseStructure::AlgorithmicDifferentiableBinaryPhraseStructure(const unsigned t_size) :
        _size(t_size),
        chart_forward(std::make_shared<BinaryPhraseStructureChart>(_size)),
        chart_backward(std::make_shared<BinaryPhraseStructureChart>(_size))
{}

AlgorithmicDifferentiableBinaryPhraseStructure::AlgorithmicDifferentiableBinaryPhraseStructure(std::shared_ptr<BinaryPhraseStructureChart> chart_forward, std::shared_ptr<BinaryPhraseStructureChart> chart_backward) :
        _size(chart_forward->size),
        chart_forward(chart_forward),
        chart_backward(chart_backward)
{}

void AlgorithmicDifferentiableBinaryPhraseStructure::forward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            // use += because we initialized them with arc weights
            chart_forward->weight(i, j) += forward_algorithmic_softmax(
                    chart_forward->weight.iter2(i, i), chart_forward->weight.iter1(i + 1, j),
                    chart_forward->split_weights.iter3(i, j, i),
                    chart_forward->backptr.iter3(i, j, i),
                    l
            );
        }
    }
}

void AlgorithmicDifferentiableBinaryPhraseStructure::forward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    chart_forward->soft_selection(0, size - 1) = 1.0f;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;
            diffdp::forward_backtracking(
                    chart_forward->soft_selection.iter2(i, i), chart_forward->soft_selection.iter1(i + 1, j),
                    chart_forward->soft_selection(i, j),
                    chart_forward->backptr.iter3(i, j, i),
                    l
            );
        }
    }
}

void AlgorithmicDifferentiableBinaryPhraseStructure::backward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = 1; l < size ; ++l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            diffdp::backward_backtracking(
                    chart_forward->soft_selection.iter2(i, i), chart_forward->soft_selection.iter1(i + 1, j),
                    chart_forward->soft_selection(i, j),
                    chart_forward->backptr.iter3(i, j, i),

                    chart_backward->soft_selection.iter2(i, i), chart_backward->soft_selection.iter1(i + 1, j),
                    &chart_backward->soft_selection(i, j),
                    chart_backward->backptr.iter3(i, j, i),

                    l
            );
        }
    }

}

void AlgorithmicDifferentiableBinaryPhraseStructure::backward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            backward_algorithmic_softmax(
                    chart_forward->weight.iter2(i, i), chart_forward->weight.iter1(i + 1, j),
                    chart_forward->split_weights.iter3(i, j, i),
                    chart_forward->backptr.iter3(i, j, i),

                    chart_backward->weight.iter2(i, i), chart_backward->weight.iter1(i + 1, j),
                    chart_backward->weight(i, j),
                    chart_backward->split_weights.iter3(i, j, i),
                    chart_backward->backptr.iter3(i, j, i),

                    l
            );
        }
    }
}

unsigned AlgorithmicDifferentiableBinaryPhraseStructure::size() const
{
    return _size;
}

float AlgorithmicDifferentiableBinaryPhraseStructure::output(const unsigned left, const unsigned right) const
{
    return chart_forward->soft_selection(left, right);
}

float AlgorithmicDifferentiableBinaryPhraseStructure::gradient(const unsigned left, const unsigned right) const
{
    return chart_backward->weight(left, right);
}



EntropyRegularizedBinaryPhraseStructure::EntropyRegularizedBinaryPhraseStructure(const unsigned t_size) :
        _size(t_size),
        chart_forward(std::make_shared<BinaryPhraseStructureChart>(_size)),
        chart_backward(std::make_shared<BinaryPhraseStructureChart>(_size))
{}

EntropyRegularizedBinaryPhraseStructure::EntropyRegularizedBinaryPhraseStructure(std::shared_ptr<BinaryPhraseStructureChart> chart_forward, std::shared_ptr<BinaryPhraseStructureChart> chart_backward) :
        _size(chart_forward->size),
        chart_forward(chart_forward),
        chart_backward(chart_backward)
{}


void EntropyRegularizedBinaryPhraseStructure::forward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            // use += because we initialized them with arc weights
            chart_forward->weight(i, j) += forward_entropy_reg(
                    chart_forward->weight.iter2(i, i), chart_forward->weight.iter1(i + 1, j),
                    chart_forward->split_weights.iter3(i, j, i),
                    chart_forward->backptr.iter3(i, j, i),
                    l
            );
        }
    }
}

void EntropyRegularizedBinaryPhraseStructure::forward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward)
{
    const unsigned size = chart_forward->size;
    chart_forward->soft_selection(0, size - 1) = 1.0f;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;
            diffdp::forward_backtracking(
                    chart_forward->soft_selection.iter2(i, i), chart_forward->soft_selection.iter1(i + 1, j),
                    chart_forward->soft_selection(i, j),
                    chart_forward->backptr.iter3(i, j, i),
                    l
            );
        }
    }
}


unsigned EntropyRegularizedBinaryPhraseStructure::size() const
{
    return _size;
}

float EntropyRegularizedBinaryPhraseStructure::output(const unsigned left, const unsigned right) const
{
    return chart_forward->soft_selection(left, right);
}

float EntropyRegularizedBinaryPhraseStructure::gradient(const unsigned left, const unsigned right) const
{
    return chart_backward->weight(left, right);
}

void EntropyRegularizedBinaryPhraseStructure::backward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = 1; l < size ; ++l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            diffdp::backward_backtracking(
                    chart_forward->soft_selection.iter2(i, i), chart_forward->soft_selection.iter1(i + 1, j),
                    chart_forward->soft_selection(i, j),
                    chart_forward->backptr.iter3(i, j, i),

                    chart_backward->soft_selection.iter2(i, i), chart_backward->soft_selection.iter1(i + 1, j),
                    &chart_backward->soft_selection(i, j),
                    chart_backward->backptr.iter3(i, j, i),

                    l
            );
        }
    }
}
void EntropyRegularizedBinaryPhraseStructure::backward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward)
{
    const unsigned size = chart_forward->size;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0; i < size - l; ++i)
        {
            unsigned j = i + l;

            backward_entropy_reg(
                    chart_forward->weight.iter2(i, i), chart_forward->weight.iter1(i + 1, j),
                    chart_forward->split_weights.iter3(i, j, i),
                    chart_forward->backptr.iter3(i, j, i),

                    chart_backward->weight.iter2(i, i), chart_backward->weight.iter1(i + 1, j),
                    chart_backward->weight(i, j),
                    chart_backward->split_weights.iter3(i, j, i),
                    chart_backward->backptr.iter3(i, j, i),

                    l
            );
        }
    }
}
}