#pragma once

#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>
#include <boost/iterator/counting_iterator.hpp>

#include "dynet/expr.h"
#include "dynet/tensor-eigen.h"
#include "dynet/nodes-impl-macros.h"
#include "dynet/nodes-def-macros.h"

#include "diffdp/chart.h"
//#include "semiring/types/signed_log_value.h"
//#include "np/nodes/projective.h"

namespace diffdp
{

enum struct DiscreteMode
{
Null, // do not backpropagate
StraightThrough, // discrete output, copy input gradient
ForwardRegularized, // entropy regularization (ie. softmax)
BackwardRegularized // forward: discrete, backward: entropy regularization
};

enum struct GraphMode
{
    Adjacency, // adjacency matrix
    Compact
};

std::pair<unsigned, unsigned> from_adjacency(const std::pair<unsigned, unsigned> dep, const diffdp::GraphMode mode);
std::pair<unsigned, unsigned> from_compact(const std::pair<unsigned, unsigned> dep, const diffdp::GraphMode mode);

namespace
{

inline
void softmax(
    float* first, float* last,
    float* output,
    const float temp=1.0f
) noexcept
{
    float m = *(std::max_element(first, last)) / temp;
    float z = 0.f;
    std::transform(
        first, last,
        output,
        [&](const float a) -> float
        {
            const float v = std::exp((a / temp) - m);
            z += v;
            return v;
        }
    );

    for (auto it = first ; it != last ; ++it)
    {
        *output = *output / z;
        ++output;
    }
    /*std::transform(
        output, output + (last - first),
        output,
        std::bind2nd(std::divides<float>(), z)
    );
    */
}

/*float affine(const float a, const float b, const float c)
{
    return a + b*c;
}*/

}

struct BackPointer
{
    unsigned* _memory = nullptr;

    ChartMatrix2D<unsigned> uright;
    ChartMatrix2D<unsigned> uleft;
    ChartMatrix2D<unsigned> cright;
    ChartMatrix2D<unsigned> cleft;

    BackPointer(const unsigned size) :
        _memory(ChartMemory::getInstance().get_unsigned(size*size*4u)),
        uright(size, _memory),
        uleft(size, _memory + size*size),
        cright(size, _memory + 2u*size*size),
        cleft(size, _memory + 3u*size*size)
    {
        std::fill(_memory, _memory + 4*ChartMatrix2D<unsigned>::required_cells(size), unsigned{});
    }

    ~BackPointer()
    {
        if (_memory != nullptr)
            ChartMemory::getInstance().release(_memory);
    }

    BackPointer(const unsigned size, unsigned* mem) :
        uright(size, mem),
        uleft(size, mem + ChartMatrix2D<unsigned>::required_cells(size)),
        cright(size, mem + 2u*ChartMatrix2D<unsigned>::required_cells(size)),
        cleft(size, mem + 3u*ChartMatrix2D<unsigned>::required_cells(size))
    {
        std::fill(mem, mem + 4*ChartMatrix2D<unsigned>::required_cells(size), unsigned{});
    }

    inline static
    std::size_t required_memory(const unsigned size)
    {
        return ChartMatrix2D<unsigned>::required_memory(size) * 4;
    }

    inline static
    unsigned required_cells(const unsigned size)
    {
        return ChartMatrix2D<unsigned>::required_cells(size) * 4;
    }
};

typedef BackPointer SubGradient;

struct Chart
{
    const unsigned size_3d;
    const unsigned size_2d;

    float* _memory = nullptr;

    ChartMatrix3D a_cleft;
    ChartMatrix3D a_cright;
    ChartMatrix3D a_uleft;
    ChartMatrix3D a_uright;
    ChartMatrix3D b_cleft;
    ChartMatrix3D b_cright;
    ChartMatrix3D b_uleft;
    ChartMatrix3D b_uright;
    ChartMatrix2D<float> c_cleft;
    ChartMatrix2D<float> c_cright;
    ChartMatrix2D<float> c_uleft;
    ChartMatrix2D<float> c_uright;
    ChartMatrix2D<float> soft_c_cleft;
    ChartMatrix2D<float> soft_c_cright;
    ChartMatrix2D<float> soft_c_uleft;
    ChartMatrix2D<float> soft_c_uright;

    Chart(unsigned size) :
        size_3d(size*size*size),
        size_2d(size*size),
        _memory(ChartMemory::getInstance().get_float(size_3d * 8 + size_2d * 8)),
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
    {
        std::fill(_memory, _memory + size_3d * 8 + size_2d * 8, float{});
    }

    ~Chart()
    {
        if (_memory != nullptr)
            ChartMemory::getInstance().release(_memory);
    }

    Chart(unsigned size, float* mem) :
        //size_3d(ChartMatrix3D::required_cells(size)),
        //size_2d(ChartMatrix2D<float>::required_cells(size)),
        size_3d(size*size*size),
        size_2d(size*size),
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
    {
        std::fill(mem, mem + size_3d * 8 + size_2d * 8, float{});
    }

    inline static
    std::size_t required_memory(const unsigned size)
    {
        return
            8 * ChartMatrix3D::required_memory(size)
            + 8 * ChartMatrix2D<float>::required_memory(size)
        ;
    }

    inline static
    unsigned required_cells(const unsigned size)
    {
        return
            8 * ChartMatrix3D::required_cells(size)
            + 8 * ChartMatrix2D<float>::required_cells(size)
        ;
    }
};


struct ContinuousEisner
{
    unsigned _size;
    float _temp;

    Chart forward;
    Chart backward;
    BackPointer backptr;
    SubGradient subg;

    std::vector<unsigned> hard_heads;

    template<class Lambda>
    ContinuousEisner(
        const unsigned t_size,
        const Lambda& Weights,
        float t_temp
    ) :
        _size(t_size),
        _temp(t_temp),
        forward(_size),
        backward(_size),
        backptr(_size),
        subg(_size),
        hard_heads(_size, 0u)
    {
        _run_inside(Weights);
        _run_outside();
    }

    template<class Lambda>
    ContinuousEisner(
        const unsigned t_size,
        const Lambda& Weights,
        float t_temp,
        float* fmem,
        unsigned* umem
    ) :
        _size(t_size),
        _temp(t_temp),
        forward(_size, fmem),
        backward(_size, fmem + Chart::required_cells(_size)),
        backptr(_size, umem),
        subg(_size, umem + BackPointer::required_cells(_size)),
        hard_heads(_size, 0u)
    {
        _run_inside(Weights);
        _run_outside();
    }

    static inline
    std::size_t required_memory(unsigned size)
    {
        return
            2 * Chart::required_memory(size)
            + 2 * BackPointer::required_memory(size)
        ;
    }

    static inline
    std::size_t required_memory_float(unsigned size)
    {
        return 2 * Chart::required_memory(size);
    }

    static inline
    unsigned required_float_cells(unsigned size)
    {
        return 2 * Chart::required_cells(size);
    }

    static inline
    unsigned required_unsigned_cells(unsigned size)
    {
        return 2 * BackPointer::required_cells(size);
    }

    template<class InputGradientType, class OutputGradientType>
    void backpropagate(InputGradientType InputGradient, OutputGradientType OutputGradient) noexcept
    {
        _backprop_outside(InputGradient);
        _backprop_inside();

        for (unsigned i = 0 ; i < size() ; ++i)
        {
            for (unsigned j = 1 ; j < size() ; ++j)
            {
                if (i < j)
                    OutputGradient(i, j, backward.c_uright(i, j));
                else if (j < i)
                    OutputGradient(i, j, backward.c_uleft(j, i));
            }
        }
    }
    
    template <class Lambda>
    void _run_inside(Lambda& Weights) noexcept
    {
        for (unsigned l = 1u ; l < size() ; ++l)
        {
            for (unsigned i = 0u ; i < size() - l ; ++ i)
            {
                unsigned j = i + l;

                for(unsigned k = i ; k < j ; ++k)
                    forward.a_uright(i, j, k) = forward.c_cright(i, k) + forward.c_cleft(k+1, j);//+ Weights(i, j);
                softmax(
                    forward.a_uright.iter3(i, j, i), forward.a_uright.iter3(i, j, j),
                    forward.b_uright.iter3(i, j, i),
                    temp()
                );
                forward.c_uright(i, j) = Weights(i, j);
                for(unsigned k = i ; k < j ; ++k)
                    forward.c_uright(i, j) += forward.b_uright(i, j, k) * forward.a_uright(i, j, k);

                backptr.uright(i, j) = *std::max_element(
                    boost::counting_iterator<unsigned>(i), boost::counting_iterator<unsigned>(j),
                    [&](const boost::counting_iterator<unsigned>& k1, const boost::counting_iterator<unsigned>& k2)
                    {
                        return forward.a_uright(i, j, *k1) < forward.a_uright(i, j, *k2);
                    }
                );

                if (i > 0u)
                {
                    for(unsigned k = i ; k < j ; ++k)
                        forward.a_uleft(i, j, k) = forward.c_cright(i, k) + forward.c_cleft(k+1, j); //+ Weights(j, i);
                    softmax(
                        forward.a_uleft.iter3(i, j, i), forward.a_uleft.iter3(i, j, j),
                        forward.b_uleft.iter3(i, j, i),
                        temp()
                    );
                    forward.c_uleft(i, j) = Weights(j, i);
                    for(unsigned k = i ; k < j ; ++k)
                        forward.c_uleft(i, j) += forward.b_uleft(i, j, k) * forward.a_uleft(i, j, k);

                    backptr.uleft(i, j) = *std::max_element(
                        boost::counting_iterator<unsigned>(i), boost::counting_iterator<unsigned>(j),
                        [&](const boost::counting_iterator<unsigned>& k1, const boost::counting_iterator<unsigned>& k2)
                        {
                            return forward.a_uleft(i, j, *k1) < forward.a_uleft(i, j, *k2);
                        }
                    );
                }

                for(unsigned k = i + 1 ; k <= j ; ++k)
                    forward.a_cright(i, j, k) = forward.c_uright(i, k) + forward.c_cright(k, j);
                softmax(
                    forward.a_cright.iter3(i, j, i+1), forward.a_cright.iter3(i, j, j+1),
                    forward.b_cright.iter3(i, j, i+1),
                    temp()
                );
                for(unsigned k = i + 1 ; k <= j ; ++k)
                    forward.c_cright(i, j) += forward.b_cright(i, j, k) * forward.a_cright(i, j, k);
                backptr.cright(i, j) = *std::max_element(
                    boost::counting_iterator<unsigned>(i+1), boost::counting_iterator<unsigned>(j+1),
                    [&](const boost::counting_iterator<unsigned>& k1, const boost::counting_iterator<unsigned>& k2)
                    {
                        return forward.a_cright(i, j, *k1) < forward.a_cright(i, j, *k2);
                    }
                );

                if (i > 0u)
                {
                    for(unsigned k = i ; k < j ; ++k)
                        forward.a_cleft(i, j, k) = forward.c_cleft(i, k) + forward.c_uleft(k, j);
                    softmax(
                        forward.a_cleft.iter3(i, j, i), forward.a_cleft.iter3(i, j, j),
                        forward.b_cleft.iter3(i, j, i),
                        temp()
                    );
                    for(unsigned k = i ; k < j ; ++k)
                        forward.c_cleft(i, j) += forward.b_cleft(i, j, k) * forward.a_cleft(i, j, k);
                    backptr.cleft(i, j) = *std::max_element(
                        boost::counting_iterator<unsigned>(i), boost::counting_iterator<unsigned>(j),
                        [&](const boost::counting_iterator<unsigned>& k1, const boost::counting_iterator<unsigned>& k2)
                        {
                            return forward.a_cleft(i, j, *k1) < forward.a_cleft(i, j, *k2);
                        }
                    );
                }
            }
        }
}

void _run_outside() noexcept
{
    forward.soft_c_cright(0, size() - 1) = 1.0f;
    subg.cright(0, size() - 1) = 1u;

    for (unsigned l = size() - 1 ; l >= 1 ; --l)
    {
        for (unsigned i = 0u ; i < size() - l ; ++i)
        {
            unsigned j = i + l;

            for (unsigned k = i+1 ; k <= j ; ++k)
            {
                forward.soft_c_uright(i, k) += forward.soft_c_cright(i, j) * forward.b_cright(i, j, k);
                forward.soft_c_cright(k, j) += forward.soft_c_cright(i, j) * forward.b_cright(i, j, k);
            }

            {
            const unsigned k = backptr.cright(i, j);
            subg.uright(i, k) += subg.cright(i, j);
            subg.cright(k, j) += subg.cright(i, j);
            }

            if (i > 0u)
            {
                for (unsigned k = i ; k < j ; ++k)
                {
                    forward.soft_c_cleft(i, k) += forward.soft_c_cleft(i, j) * forward.b_cleft(i, j, k);
                    forward.soft_c_uleft(k, j) += forward.soft_c_cleft(i, j) * forward.b_cleft(i, j, k);
                }

                {
                const unsigned k = backptr.cleft(i, j);
                subg.cleft(i, k) += subg.cleft(i, j);
                subg.uleft(k, j) += subg.cleft(i, j);
                }
            }

            for (unsigned k = i ; k < j ; ++k)
            {
                forward.soft_c_cright(i, k) += forward.soft_c_uright(i, j) * forward.b_uright(i, j, k);
                forward.soft_c_cleft(k+1, j) += forward.soft_c_uright(i, j) * forward.b_uright(i, j, k);
            }
            {
            const unsigned k = backptr.uright(i, j);
            subg.cright(i, k) += subg.uright(i, j);
            subg.cleft(k+1, j) += subg.uright(i, j);
            }

            if (subg.uright(i, j) != 0u)
                hard_heads.at(i) = j;

            if (i > 0u)
            {
                for (unsigned k = i ; k < j ; ++k)
                {
                    forward.soft_c_cright(i, k) += forward.soft_c_uleft(i, j) * forward.b_uleft(i, j, k);
                    forward.soft_c_cleft(k+1, j) += forward.soft_c_uleft(i , j) * forward.b_uleft(i, j, k);
                }

                {
                const unsigned k = backptr.uleft(i, j);
                subg.cright(i, k) += subg.uleft(i, j);
                subg.cleft(k+1, j) += subg.uleft(i, j);
                }

                if (subg.uleft(i, j) != 0u)
                    hard_heads.at(j) = i;
            }
        }
    }
}

    template<class InputGradientType>
    void _backprop_outside(InputGradientType InputGradient) noexcept
    {
        for (unsigned i = 0 ; i < size() ; ++i)
        {
            for (unsigned j = 1 ; j < size() ; ++j)
            {
                if (i < j)
                    backward.soft_c_uright(i, j) = InputGradient(i, j);
                else if (j < i)
                    backward.soft_c_uleft(j, i) = InputGradient(i, j);
            }
        }

        for (unsigned l = 1 ; l < size() ; ++l)
        {
            for (unsigned i = 0 ; i < size() - l ; ++i)
            {
                unsigned j = i + l;

                if (i > 0u)
                {
                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.soft_c_uleft(i, j) += 
                            backward.soft_c_cright(i, k) * forward.b_uleft(i, j, k)
                            + 
                            backward.soft_c_cleft(k+1, j) * forward.b_uleft(i, j, k)
                        ;
                        backward.b_uleft(i, j, k) +=
                            backward.soft_c_cright(i, k) * forward.soft_c_uleft(i, j)
                            + 
                            backward.soft_c_cleft(k+1, j) * forward.soft_c_uleft(i, j)
                        ;
                    }
                }

                for (unsigned k = i ; k < j ; ++k)
                {
                    backward.soft_c_uright(i, j) += 
                        backward.soft_c_cright(i, k) * forward.b_uright(i, j, k)
                        + 
                        backward.soft_c_cleft(k+1, j) * forward.b_uright(i, j, k)
                    ;
                    backward.b_uright(i, j, k) +=
                        backward.soft_c_cright(i, k) * forward.soft_c_uright(i, j)
                        + 
                        backward.soft_c_cleft(k+1, j) * forward.soft_c_uright(i, j)
                    ;
                }

                if (i > 0u)
                {
                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.soft_c_cleft(i, j) += 
                            backward.soft_c_cleft(i, k) * forward.b_cleft(i, j, k)
                            + 
                            backward.soft_c_uleft(k, j) * forward.b_cleft(i, j, k)
                        ;
                        backward.b_cleft(i, j, k) +=
                            backward.soft_c_cleft(i, k) * forward.soft_c_cleft(i, j)
                            + 
                            backward.soft_c_uleft(k, j) * forward.soft_c_cleft(i, j)
                        ;
                    }
                }

                for (unsigned k = i+1 ; k <= j ; ++k)
                {
                    backward.soft_c_cright(i, j) += 
                        backward.soft_c_uright(i, k) * forward.b_cright(i, j, k)
                        + 
                        backward.soft_c_cright(k, j) * forward.b_cright(i, j, k)
                    ;
                    backward.b_cright(i, j, k) +=
                        backward.soft_c_uright(i, k) * forward.soft_c_cright(i, j)
                        + 
                        backward.soft_c_cright(k, j) * forward.soft_c_cright(i, j)
                    ;
                }
            }
        }
    }

    void _backprop_inside() noexcept
    {
        for (unsigned l = size() - 1 ; l >= 1 ; --l)
        {
            for (unsigned i = 0 ; i < size() - l ; ++i)
            {
                unsigned j = i + l;

                if (i > 0u)
                {
                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.b_cleft(i, j, k) += backward.c_cleft(i, j) * forward.a_cleft(i, j, k);
                        backward.a_cleft(i, j, k) += backward.c_cleft(i, j) * forward.b_cleft(i, j, k);
                    }
                    float s = backward.b_cleft(i, j, i) * forward.b_cleft(i, j, i);
                    for (unsigned k = i + 1 ; k < j ; ++k)
                        s += backward.b_cleft(i, j, k) * forward.b_cleft(i, j, k);
                    for (unsigned k = i ; k < j ; ++k)
                        backward.a_cleft(i, j, k) += forward.b_cleft(i, j, k) * (backward.b_cleft(i, j, k) - s) / temp();

                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.c_cleft(i, k) += backward.a_cleft(i, j, k);
                        backward.c_uleft(k, j) += backward.a_cleft(i, j, k);
                    }
                }

                for (unsigned k = i + 1 ; k <= j ; ++k)
                {
                    backward.b_cright(i, j, k) += backward.c_cright(i, j) * forward.a_cright(i, j, k);
                    backward.a_cright(i, j, k) += backward.c_cright(i, j) * forward.b_cright(i, j, k);
                }
                float s = backward.b_cright(i, j, i+1) * forward.b_cright(i, j, i+1);
                for (unsigned k = i + 2 ; k <= j ; ++k)
                    s += backward.b_cright(i, j, k) * forward.b_cright(i, j, k);
                for (unsigned k = i + 1 ; k <= j ; ++k)
                    backward.a_cright(i, j, k) += forward.b_cright(i, j, k) * (backward.b_cright(i, j, k) - s) / temp();
                for (unsigned k = i+1 ; k <= j ; ++k)
                {
                    backward.c_uright(i, k) += backward.a_cright(i, j, k);
                    backward.c_cright(k, j) += backward.a_cright(i, j, k);
                }

                if (i > 0u)
                {
                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.b_uleft(i, j, k) += backward.c_uleft(i, j) * forward.a_uleft(i, j, k);
                        backward.a_uleft(i, j, k) += backward.c_uleft(i, j) * forward.b_uleft(i, j, k);
                    }
                    s = backward.b_uleft(i, j, i) * forward.b_uleft(i, j, i);
                    for (unsigned k = i + 1; k < j ; ++k)
                        s += backward.b_uleft(i, j, k) * forward.b_uleft(i, j, k);
                    for (unsigned k = i ; k < j ; ++k)
                        backward.a_uleft(i, j, k) += forward.b_uleft(i, j, k) * (backward.b_uleft(i, j, k) - s) / temp();
                    for (unsigned k = i ; k < j ; ++k)
                    {
                        backward.c_cright(i, k) += backward.a_uleft(i, j, k);
                        backward.c_cleft(k+1, j) += backward.a_uleft(i, j, k);
                    }
                }

                for (unsigned k = i ; k < j ; ++k)
                {
                    backward.b_uright(i, j, k) += backward.c_uright(i, j) * forward.a_uright(i, j, k);
                    backward.a_uright(i, j, k) += backward.c_uright(i, j) * forward.b_uright(i, j, k);
                }
                s = backward.b_uright(i, j, i) * forward.b_uright(i, j, i);
                for (unsigned k = i+1 ; k < j ; ++k)
                    s += backward.b_uright(i, j, k) * forward.b_uright(i, j, k);
                for (unsigned k = i ; k < j ; ++k)
                    backward.a_uright(i, j, k) += forward.b_uright(i, j, k) * (backward.b_uright(i, j, k) - s) / temp();
                for (unsigned k = i ; k < j ; ++k)
                {
                    backward.c_cright(i, k) += backward.a_uright(i, j, k);
                    backward.c_cleft(k+1, j) += backward.a_uright(i, j, k);
                }
            }
        }
    }

    inline
    unsigned size() const
    {
        return _size;
    }

    inline
    float temp() const
    {
        return _temp;
    }

    float arc_value(const unsigned head, const unsigned mod) const
    {
        assert(mod >= 1);
        assert(head < size());
        assert(mod < size());
        assert(head != mod);
        if (head < mod)
            return forward.soft_c_uright(head, mod);
        else
            return forward.soft_c_uleft(mod, head);
    }

    float discrete_arc_value(const unsigned head, const unsigned mod) const
    {
        assert(mod >= 1);
        assert(head < size());
        assert(mod < size());
        assert(head != mod);
        if (head < mod)
            return (float) subg.uright(head, mod);
        else
            return (float) subg.uleft(mod, head);
    }
};

}

namespace dynet
{
Expression continuous_eisner(
    const Expression& x,
    float temp,
    diffdp::DiscreteMode mode,
    diffdp::GraphMode input_graph = diffdp::GraphMode::Compact,
    diffdp::GraphMode output_graph = diffdp::GraphMode::Compact,
    bool with_root_arcs=true,
    std::vector<unsigned>* batch_sizes = nullptr
);

struct ContinuousEisner :
    public dynet::Node
{
    const diffdp::DiscreteMode mode;
    const diffdp::GraphMode input_graph;
    const diffdp::GraphMode output_graph;
    const float _temp;
    bool with_root_arcs;
    std::vector<unsigned>* batch_sizes = nullptr;

    std::vector<diffdp::ContinuousEisner*> _ce_ptr;

    explicit ContinuousEisner(
        const std::initializer_list<VariableIndex>& a,
        float temp,
        diffdp::DiscreteMode mode,
        diffdp::GraphMode input_graph,
        diffdp::GraphMode output_graph,
        bool with_root_arcs,
        std::vector<unsigned>* batch_sizes
    ) :
        Node(a),
        mode(mode),
        input_graph(input_graph),
        output_graph(output_graph),
        _temp(temp),
        with_root_arcs(with_root_arcs),
        batch_sizes(batch_sizes)
    {
        this->has_cuda_implemented = false;
    }

    DYNET_NODE_DEFINE_DEV_IMPL()

    virtual bool supports_multibatch() const override { return true; }
    size_t aux_storage_size() const override;

    virtual ~ContinuousEisner()
    {
        for (auto*& ptr : _ce_ptr)
            if (ptr != nullptr)
            {
                delete ptr;
                ptr = nullptr;
            }
    }
};
}
