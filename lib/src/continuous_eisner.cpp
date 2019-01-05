#include <omp.h>
#include <iostream>


#include "diffdp/continuous_eisner.h"

namespace diffdp
{
std::pair<unsigned, unsigned> from_adjacency(const std::pair<unsigned, unsigned> dep, const diffdp::GraphMode mode)
{
    unsigned head = dep.first;
    unsigned mod = dep.second;
    if (mode == diffdp::GraphMode::Compact)
    {
        mod -= 1u;
        if (head == 0u)
            head = mod;
        else
            head -= 1u;
    }

    return {head, mod};
}

std::pair<unsigned, unsigned> from_compact(const std::pair<unsigned, unsigned> dep, const diffdp::GraphMode mode)
{
    unsigned head = dep.first;
    unsigned mod = dep.second;
    if (mode == diffdp::GraphMode::Adjacency)
    {
        if (head == mod)
            head = 0u;
        else
            head += 1u;
        mod += 1u;
    }

    return {head, mod};
}
}

namespace dynet
{


std::string ContinuousEisner::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "continuous_eisner(" << arg_names[0] << ")";
    return s.str();
}

Dim ContinuousEisner::dim_forward(const std::vector<Dim>& xs) const {
    DYNET_ARG_CHECK(
        xs.size() == 1 && xs[0].nd == 2 && xs[0].rows() == xs[0].cols(),
        "Bad input dimensions in ContinuousEisner: " << xs
    );
    if (input_graph == diffdp::GraphMode::Compact)
        DYNET_ARG_CHECK(
            xs[0].rows() >= 1,
            "Bad input dimensions in ContinuousEisner: " << xs
        )
    else
        DYNET_ARG_CHECK(
            xs[0].rows() >= 2,
            "Bad input dimensions in ContinuousEisner: " << xs
        )
        
    unsigned dim;
    if (input_graph == output_graph)
        dim = xs[0].rows();
    else if (input_graph == diffdp::GraphMode::Compact)
        dim = xs[0].rows() + 1; // from compact to adj
    else
        dim = xs[0].rows() - 1; // from adj to compact
    
    return dynet::Dim({dim, dim}, xs[0].batch_elems());
}

size_t ContinuousEisner::aux_storage_size() const {
    const unsigned eisner_dim = dim.rows() + (output_graph == diffdp::GraphMode::Compact ? 1 : 0);
    const size_t eisner_mem = diffdp::ContinuousEisner::required_memory(eisner_dim);
    return dim.batch_elems() * eisner_mem;
}

template<class MyDevice>
void ContinuousEisner::forward_dev_impl(
    const MyDevice&,
    const std::vector<const Tensor*>& xs,
    Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("ContinuousEisner::forward");
#else
    TensorTools::zero(fx);

    std::vector<diffdp::ContinuousEisner*>& _ce_ptr2 = const_cast<std::vector<diffdp::ContinuousEisner*>&>(_ce_ptr);
    for (auto*& ptr : _ce_ptr2)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
    if (_ce_ptr2.size() != xs[0]->d.batch_elems())
        _ce_ptr2.resize(xs[0]->d.batch_elems(), nullptr);

    const unsigned max_eisner_dim = xs[0]->d.rows() + (input_graph == diffdp::GraphMode::Compact ? 1 : 0);
    float* aux_fmem = static_cast<float*>(aux_mem);
    unsigned * aux_umem =
        (unsigned*)
        (aux_fmem + xs[0]->d.batch_elems() * diffdp::ContinuousEisner::required_float_cells(max_eisner_dim))
    ;

    #pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        const unsigned eisner_dim = (
            batch_sizes == nullptr
            ? xs[0]->d.rows() + (input_graph == diffdp::GraphMode::Compact ? 1 : 0)
            : batch_sizes->at(batch) + 1
        );
        const unsigned output_dim = (
            batch_sizes == nullptr
            ? fx.d.rows()
            : batch_sizes->at(batch) + (output_graph == diffdp::GraphMode::Compact ? 0 : 1)
        );

        auto input = batch_matrix(*(xs[0]), batch);

        if (mode == diffdp::DiscreteMode::Null || mode == diffdp::DiscreteMode::StraightThrough)
        {
            throw std::runtime_error("Not implemented");
        }
        else
        {
            float* fmem = aux_fmem + batch * diffdp::ContinuousEisner::required_float_cells(max_eisner_dim);
            unsigned* umem = aux_umem + batch * diffdp::ContinuousEisner::required_unsigned_cells(max_eisner_dim);

            _ce_ptr2.at(batch) = new diffdp::ContinuousEisner(
                eisner_dim,
                [&] (const unsigned head, const unsigned mod)
                {
                    if (mod == 0u)
                        throw std::runtime_error("Illegal arc");
                    if (head == 0u && !with_root_arcs)
                    {
                        return 0.f;
                    }
                    else
                    {
                        auto arc = from_adjacency({head, mod}, input_graph);
                        const float v = input(arc.first, arc.second);
                        return v;
                    }
                },
                _temp,
                fmem,
                umem
            );

            auto output = batch_matrix(fx, batch);
            for (unsigned head = 0u ; head < output_dim ; ++head)
            {
                for (unsigned mod = 0u ; mod < output_dim ; ++mod)
                {
                    if (output_graph == diffdp::GraphMode::Adjacency)
                    {
                        if (mod == 0u)
                        {
                            output(head, mod) = 0.f;
                            continue;
                        }
                        if (head == mod)
                        {
                            output(head, mod) = 0.f;
                            continue;
                        }
                    }
                    const auto arc = (
                        output_graph == diffdp::GraphMode::Adjacency
                        ? std::make_pair(head, mod)
                        : from_compact({head, mod}, diffdp::GraphMode::Adjacency)
                    );
                    const float a = (
                        mode == diffdp::DiscreteMode::ForwardRegularized
                        ? _ce_ptr2[batch]->arc_value(arc.first, arc.second)
                        // mode == BackwardRegularized
                        // We use discrete values during forward
                        : _ce_ptr2[batch]->discrete_arc_value(arc.first, arc.second)
                    );

                    if (!std::isfinite(a))
                        throw std::runtime_error("BAD eisner output");

                    if (arc.first == 0u && !with_root_arcs)
                        output(head, mod) = 0.f;
                    else
                        output(head, mod) = a;
                }
            }
        }
    }
#endif
    //std::cerr << "F aux: " << aux_mem << "\n";
}

template<class MyDevice>
    void ContinuousEisner::backward_dev_impl(
            const MyDevice &,
            const std::vector<const Tensor*>& xs,
            const Tensor&,
            const Tensor& dEdf,
            unsigned,
            Tensor& dEdxi
) const {
#ifdef __CUDACC__
        DYNET_NO_CUDA_IMPL_ERROR("ContinuousEisner::backward");
#else
    if (mode == diffdp::DiscreteMode::Null)
        return;

    #pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        auto output_grad = batch_matrix(dEdxi, batch);
        auto input_grad = batch_matrix(dEdf, batch);

        if (mode == diffdp::DiscreteMode::StraightThrough)
        {
            // just copy the incoming sensitivity
            const unsigned input_dim = (
                batch_sizes == nullptr
                ? dEdf.d.rows()
                : batch_sizes->at(batch) + (input_graph == diffdp::GraphMode::Compact ? 0 : 1)
            );
            for (unsigned head = 0u ; head < input_dim ; ++head)
            {
                for (unsigned mod = 0u ; mod < input_dim ; ++mod)
                {
                    if (input_graph == diffdp::GraphMode::Adjacency)
                    {
                        if (mod == 0u)
                            continue;
                        if (head == mod)
                            continue;
                        if (head == 0u && !with_root_arcs)
                            continue;
                    }
                    else
                    {
                        if (head == mod && !with_root_arcs)
                            continue;
                    }

                    const auto arc = (
                        input_graph == diffdp::GraphMode::Adjacency
                        ? from_adjacency({head, mod}, output_graph)
                        : from_compact({head, mod}, output_graph)
                    );
                    output_grad(head, mod) += input_grad(arc.first, arc.second);
                }
            }
        }
        else
        {
            _ce_ptr.at(batch)->backpropagate(
                [&] (unsigned head, unsigned mod) -> float
                {
                    if (head == 0u && !with_root_arcs)
                        return 0.f;
                    auto arc = from_adjacency({head, mod}, output_graph);
                    const float v = input_grad(arc.first, arc.second);
                    if (!std::isfinite(v))
                        throw std::runtime_error("BAD eisner input grad");
                    return v;
                },
                [&] (unsigned head, unsigned mod, float v)
                {
                    if (head == 0u && !with_root_arcs)
                        return;
                    if (!std::isfinite(v))
                        throw std::runtime_error("BAD eisner output grad");

                    if (mod != 0u && head != mod)
                    {
                        auto arc = from_adjacency({head, mod}, input_graph);
                        output_grad(arc.first, arc.second) += v;
                    }
                }
            );
        }
    }
#endif
}

DYNET_NODE_INST_DEV_IMPL(ContinuousEisner)

Expression continuous_eisner(const Expression& x, float temp, diffdp::DiscreteMode mode, diffdp::GraphMode input_graph, diffdp::GraphMode output_graph, bool with_root_arcs, std::vector<unsigned>* batch_sizes)
{
    return Expression(x.pg, x.pg->add_function<ContinuousEisner>({x.i}, temp, mode, input_graph, output_graph, with_root_arcs, batch_sizes));
}

}
