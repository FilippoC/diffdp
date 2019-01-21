#pragma once

#include <cassert>
#include <functional>
#include <memory>

#include "diffdp/chart.h"
#include "diffdp/deduction_operations.h"

namespace diffdp
{


struct BinaryPhraseStructureChart
{
    const unsigned size;
    const unsigned size_3d;
    const unsigned size_2d;
    float* _memory = nullptr;
    const bool _erase_memory;

    Tensor3D<float> split_weights, backptr;
    Matrix<float> weight, soft_selection;

    BinaryPhraseStructureChart(unsigned size);
    BinaryPhraseStructureChart(unsigned size, float* mem);
    ~BinaryPhraseStructureChart();

    void zeros();

    static std::size_t required_memory(const unsigned size);
    static unsigned required_cells(const unsigned size);
};



struct AlgorithmicDifferentiableBinaryPhraseStructure
{
    unsigned _size;

    std::shared_ptr<BinaryPhraseStructureChart> chart_forward;
    std::shared_ptr<BinaryPhraseStructureChart> chart_backward;

    explicit AlgorithmicDifferentiableBinaryPhraseStructure(const unsigned t_size);
    AlgorithmicDifferentiableBinaryPhraseStructure(std::shared_ptr<BinaryPhraseStructureChart> chart_forward, std::shared_ptr<BinaryPhraseStructureChart> chart_backward);

    template<class Functor>
    void forward(Functor&& weight_callback);

    template<class Functor>
    void backward(Functor&& gradient_callback);

    static void forward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward);
    static void forward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward);

    static void backward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward);
    static void backward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward);

    float output(const unsigned head, const unsigned mod) const;
    float gradient(const unsigned left, const unsigned right) const;

    unsigned size() const;
};


struct EntropyRegularizedBinaryPhraseStructure
{
    unsigned _size;

    std::shared_ptr<BinaryPhraseStructureChart> chart_forward;
    std::shared_ptr<BinaryPhraseStructureChart> chart_backward;

    explicit EntropyRegularizedBinaryPhraseStructure(const unsigned t_size);
    EntropyRegularizedBinaryPhraseStructure(std::shared_ptr<BinaryPhraseStructureChart> chart_forward, std::shared_ptr<BinaryPhraseStructureChart> chart_backward);


    template<class Functor>
    void forward(Functor&& weight_callback);

    template<class Functor>
    void backward(Functor&& gradient_callback);

    static void forward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward);
    static void forward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward);

    static void backward_maximize(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward);
    static void backward_backtracking(std::shared_ptr<BinaryPhraseStructureChart>& chart_forward, std::shared_ptr<BinaryPhraseStructureChart>& chart_backward);

    float output(const unsigned head, const unsigned mod) const;
    float gradient(const unsigned head, const unsigned mod) const;

    unsigned size() const;
};


// templates implementations

template<class Functor>
void AlgorithmicDifferentiableBinaryPhraseStructure::forward(Functor&& weight_callback)
{
    const unsigned size = chart_forward->size;

    chart_forward->zeros(); // we could skip some zeros here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = i + 1; j < size; ++j)
        {
            chart_forward->weight(i, j) = weight_callback(i, j);
        }
    }

    AlgorithmicDifferentiableBinaryPhraseStructure::forward_maximize(chart_forward);
    AlgorithmicDifferentiableBinaryPhraseStructure::forward_backtracking(chart_forward);
}

template<class Functor>
void AlgorithmicDifferentiableBinaryPhraseStructure::backward(Functor&& gradient_callback)
{
    const unsigned size = chart_forward->size;

    chart_backward->zeros();
    // init gradient here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = i + 1; j < size; ++j)
        {
            chart_backward->soft_selection(i, j) = gradient_callback(i, j);
        }
    }

    AlgorithmicDifferentiableBinaryPhraseStructure::backward_backtracking(chart_forward, chart_backward);
    AlgorithmicDifferentiableBinaryPhraseStructure::backward_maximize(chart_forward, chart_backward);
}

template<class Functor>
void EntropyRegularizedBinaryPhraseStructure::forward(Functor&& weight_callback)
{
    const unsigned size = chart_forward->size;

    chart_forward->zeros(); // we could skip some zeros here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = i + 1; j < size; ++j)
        {
            chart_forward->weight(i, j) = weight_callback(i, j);
        }
    }

    EntropyRegularizedBinaryPhraseStructure::forward_maximize(chart_forward);
    EntropyRegularizedBinaryPhraseStructure::forward_backtracking(chart_forward);
}

template<class Functor>
void EntropyRegularizedBinaryPhraseStructure::backward(Functor&& gradient_callback)
{
    const unsigned size = chart_forward->size;

    chart_backward->zeros();
    // init gradient here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = i + 1; j < size; ++j)
        {
            chart_backward->soft_selection(i, j) = gradient_callback(i, j);
        }
    }

    EntropyRegularizedBinaryPhraseStructure::backward_backtracking(chart_forward, chart_backward);
    EntropyRegularizedBinaryPhraseStructure::backward_maximize(chart_forward, chart_backward);
}


}