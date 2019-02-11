/**
 * TODO: there is a lot a duplicate code between the two nodes (almost identical)
 */
#include "diffdp/dynet/eisner.h"
#include "dynet/tensor-eigen.h"

namespace diffdp
{

std::pair<unsigned, unsigned> from_adjacency(const std::pair<unsigned, unsigned> dep, const diffdp::DependencyGraphMode mode)
{
    unsigned head = dep.first;
    unsigned mod = dep.second;
    if (mode == diffdp::DependencyGraphMode::Compact)
    {
        mod -= 1u;
        if (head == 0u)
            head = mod;
        else
            head -= 1u;
    }

    return {head, mod};
}

std::pair<unsigned, unsigned> from_compact(const std::pair<unsigned, unsigned> dep, const diffdp::DependencyGraphMode mode)
{
    unsigned head = dep.first;
    unsigned mod = dep.second;
    if (mode == diffdp::DependencyGraphMode::Adjacency)
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

Expression algorithmic_differentiable_eisner(const Expression& x, diffdp::DiscreteMode mode, diffdp::DependencyGraphMode input_graph, diffdp::DependencyGraphMode output_graph, bool with_root_arcs, std::vector<unsigned>* batch_sizes)
{
    return Expression(x.pg, x.pg->add_function<AlgorithmicDifferentiableEisner>({x.i}, mode, input_graph, output_graph, with_root_arcs, batch_sizes));
}

Expression entropy_regularized_eisner(const Expression& x, diffdp::DiscreteMode mode, diffdp::DependencyGraphMode input_graph, diffdp::DependencyGraphMode output_graph, bool with_root_arcs, std::vector<unsigned>* batch_sizes)
{
    return Expression(x.pg, x.pg->add_function<EntropyRegularizedEisner>({x.i}, mode, input_graph, output_graph, with_root_arcs, batch_sizes));
}

AlgorithmicDifferentiableEisner::AlgorithmicDifferentiableEisner(
        const std::initializer_list<VariableIndex>& a,
        diffdp::DiscreteMode mode,
        diffdp::DependencyGraphMode input_graph,
        diffdp::DependencyGraphMode output_graph,
        bool with_root_arcs,
        std::vector<unsigned>* batch_sizes
) :
        Node(a),
        mode(mode),
        input_graph(input_graph),
        output_graph(output_graph),
        with_root_arcs(with_root_arcs),
        batch_sizes(batch_sizes)
{
    this->has_cuda_implemented = false;
}

bool AlgorithmicDifferentiableEisner::supports_multibatch() const
{
    return true;
}

AlgorithmicDifferentiableEisner::~AlgorithmicDifferentiableEisner()
{
    for (auto*& ptr : _ce_ptr)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
}

std::string AlgorithmicDifferentiableEisner::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "algorithmic_differentiable_eisner(" << arg_names[0] << ")";
    return s.str();
}

Dim AlgorithmicDifferentiableEisner::dim_forward(const std::vector<Dim>& xs) const {
    DYNET_ARG_CHECK(
            xs.size() == 1 && xs[0].nd == 2 && xs[0].rows() == xs[0].cols(),
            "Bad input dimensions in AlgorithmicDifferentiableEisner: " << xs
    );
    if (input_graph == diffdp::DependencyGraphMode::Compact)
        DYNET_ARG_CHECK(
                xs[0].rows() >= 1,
                "Bad input dimensions in AlgorithmicDifferentiableEisner: " << xs
        )
    else
        DYNET_ARG_CHECK(
                xs[0].rows() >= 2,
                "Bad input dimensions in AlgorithmicDifferentiableEisner: " << xs
        )

    unsigned dim;
    if (input_graph == output_graph)
        dim = xs[0].rows();
    else if (input_graph == diffdp::DependencyGraphMode::Compact)
        dim = xs[0].rows() + 1; // from compact to adj
    else
        dim = xs[0].rows() - 1; // from adj to compact

    return dynet::Dim({dim, dim}, xs[0].batch_elems());
}

size_t AlgorithmicDifferentiableEisner::aux_storage_size() const {
    const unsigned eisner_dim = dim.rows() + (output_graph == diffdp::DependencyGraphMode::Compact ? 1 : 0);
    // 2 times because we have a forward and a backward chart
    const size_t eisner_mem = 2 * diffdp::EisnerChart::required_memory(eisner_dim);
    return dim.batch_elems() * eisner_mem;
}




template<class MyDevice>
void AlgorithmicDifferentiableEisner::forward_dev_impl(
        const MyDevice&,
        const std::vector<const Tensor*>& xs,
        Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("AlgorithmicDifferentiableEisner::forward");
#else
    // TODO call zero only when necessary
    TensorTools::zero(fx);

    std::vector<diffdp::AlgorithmicDifferentiableEisner*>& _ce_ptr2 =
            const_cast<std::vector<diffdp::AlgorithmicDifferentiableEisner*>&>(_ce_ptr);

    for (auto*& ptr : _ce_ptr2)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }

    if (_ce_ptr2.size() != xs[0]->d.batch_elems())
        _ce_ptr2.resize(xs[0]->d.batch_elems(), nullptr);

    const unsigned max_eisner_dim = xs[0]->d.rows() + (input_graph == diffdp::DependencyGraphMode::Compact ? 1 : 0);
    float* aux_fmem = static_cast<float*>(aux_mem);

    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        const unsigned eisner_dim = (
                batch_sizes == nullptr
                ? max_eisner_dim
                : batch_sizes->at(batch) + 1
        );

        auto input = batch_matrix(*(xs[0]), batch);

        if (mode == diffdp::DiscreteMode::ForwardRegularized)
        {
            float* fmem = aux_fmem + batch * 2 * diffdp::EisnerChart::required_cells(max_eisner_dim);
            //auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            //auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem + diffdp::EisnerChart::required_cells(max_eisner_dim));
            auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim);

            _ce_ptr2.at(batch) = new diffdp::AlgorithmicDifferentiableEisner(forward_chart, backward_chart);


            _ce_ptr2.at(batch)->forward(
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
                            const auto arc = diffdp::from_adjacency({head, mod}, input_graph);
                            const float v = input(arc.first, arc.second);
                            return v;
                        }
                    }
            );

            auto output = batch_matrix(fx, batch);

            for (unsigned head = 0u ; head < eisner_dim ; ++head)
            {
                for (unsigned mod = 1u; mod < eisner_dim ; ++mod)
                {
                    const auto arc = diffdp::from_adjacency({head, mod}, output_graph);
                    if (head == mod)
                    {
                        output(arc.first, arc.second) = 0.f;
                        continue;
                    }

                    if (head == 0u && !with_root_arcs)
                    {
                        output(arc.first, arc.second) = 0.f;
                        continue;
                    }

                    const float a = _ce_ptr2[batch]->output(head, mod);

                    if (!std::isfinite(a))
                        throw std::runtime_error("BAD eisner output");

                    output(arc.first, arc.second) = a;
                }
            }
        }
        else
        {
            throw std::runtime_error("Not implemented: only ForwardRegularized can be used at the moment");
        }
    }
#endif
}

template<class MyDevice>
void AlgorithmicDifferentiableEisner::backward_dev_impl(
        const MyDevice &,
        const std::vector<const Tensor*>& xs,
        const Tensor&,
        const Tensor& dEdf,
        unsigned,
        Tensor& dEdxi
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("AlgorithmicDifferentiableEisner::backward");
#else
    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        auto output_grad = batch_matrix(dEdxi, batch);
        auto input_grad = batch_matrix(dEdf, batch);

        auto& eisner = *(_ce_ptr.at(batch));

        eisner.backward(
                [&] (unsigned head, unsigned mod) -> float
                {
                    if (head == 0u && !with_root_arcs)
                        return 0.f;
                    auto arc = diffdp::from_adjacency({head, mod}, output_graph);
                    const float v = input_grad(arc.first, arc.second);
                    if (!std::isfinite(v))
                        throw std::runtime_error("BAD eisner input grad");
                    return v;
                }
        );

        for (unsigned head = 0u ; head < eisner.size() ; ++head)
        {
            for (unsigned mod = 1u; mod < eisner.size(); ++mod)
            {
                if (head == mod)
                    continue;

                if (head == 0u && !with_root_arcs)
                    return;

                auto const v = eisner.gradient(head, mod);
                if (!std::isfinite(v))
                    throw std::runtime_error("BAD eisner output grad");

                auto arc = diffdp::from_adjacency({head, mod}, input_graph);
                output_grad(arc.first, arc.second) += v;
            }
        }
    }
#endif
}


DYNET_NODE_INST_DEV_IMPL(AlgorithmicDifferentiableEisner)




// ENTROPY Regularized


EntropyRegularizedEisner::EntropyRegularizedEisner(
        const std::initializer_list<VariableIndex>& a,
        diffdp::DiscreteMode mode,
        diffdp::DependencyGraphMode input_graph,
        diffdp::DependencyGraphMode output_graph,
        bool with_root_arcs,
        std::vector<unsigned>* batch_sizes
) :
        Node(a),
        mode(mode),
        input_graph(input_graph),
        output_graph(output_graph),
        with_root_arcs(with_root_arcs),
        batch_sizes(batch_sizes)
{
    this->has_cuda_implemented = false;
}

bool EntropyRegularizedEisner::supports_multibatch() const
{
    return true;
}

EntropyRegularizedEisner::~EntropyRegularizedEisner()
{
    for (auto*& ptr : _ce_ptr)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
}

std::string EntropyRegularizedEisner::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "entropy_regularized_eisner(" << arg_names[0] << ")";
    return s.str();
}

Dim EntropyRegularizedEisner::dim_forward(const std::vector<Dim>& xs) const {
    DYNET_ARG_CHECK(
            xs.size() == 1 && xs[0].nd == 2 && xs[0].rows() == xs[0].cols(),
            "Bad input dimensions in EntropyRegularizedEisner: " << xs
    );
    if (input_graph == diffdp::DependencyGraphMode::Compact)
        DYNET_ARG_CHECK(
                xs[0].rows() >= 1,
                "Bad input dimensions in EntropyRegularizedEisner: " << xs
        )
    else
        DYNET_ARG_CHECK(
                xs[0].rows() >= 2,
                "Bad input dimensions in EntropyRegularizedEisner: " << xs
        )

    unsigned dim;
    if (input_graph == output_graph)
        dim = xs[0].rows();
    else if (input_graph == diffdp::DependencyGraphMode::Compact)
        dim = xs[0].rows() + 1; // from compact to adj
    else
        dim = xs[0].rows() - 1; // from adj to compact

    return dynet::Dim({dim, dim}, xs[0].batch_elems());
}

size_t EntropyRegularizedEisner::aux_storage_size() const {
    const unsigned eisner_dim = dim.rows() + (output_graph == diffdp::DependencyGraphMode::Compact ? 1 : 0);
    // 2 times because we have a forward and a backward chart
    const size_t eisner_mem = 2 * diffdp::EisnerChart::required_memory(eisner_dim);
    return dim.batch_elems() * eisner_mem;
}

template<class MyDevice>
void EntropyRegularizedEisner::forward_dev_impl(
        const MyDevice&,
        const std::vector<const Tensor*>& xs,
        Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("EntropyRegularizedEisner::forward");
#else
    // TODO call zero only when necessary
    TensorTools::zero(fx);

    std::vector<diffdp::EntropyRegularizedEisner*>& _ce_ptr2 =
            const_cast<std::vector<diffdp::EntropyRegularizedEisner*>&>(_ce_ptr);

    for (auto*& ptr : _ce_ptr2)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }

    if (_ce_ptr2.size() != xs[0]->d.batch_elems())
        _ce_ptr2.resize(xs[0]->d.batch_elems(), nullptr);

    const unsigned max_eisner_dim = xs[0]->d.rows() + (input_graph == diffdp::DependencyGraphMode::Compact ? 1 : 0);
    float* aux_fmem = static_cast<float*>(aux_mem);

    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        const unsigned eisner_dim = (
                batch_sizes == nullptr
                ? max_eisner_dim
                : batch_sizes->at(batch) + 1
        );

        auto input = batch_matrix(*(xs[0]), batch);

        if (mode == diffdp::DiscreteMode::ForwardRegularized)
        {
            float* fmem = aux_fmem + batch * 2 * diffdp::EisnerChart::required_cells(max_eisner_dim);
            //auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            //auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem + diffdp::EisnerChart::required_cells(max_eisner_dim));
            auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim);

            _ce_ptr2.at(batch) = new diffdp::EntropyRegularizedEisner(forward_chart, backward_chart);


            _ce_ptr2.at(batch)->forward(
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
                            const auto arc = diffdp::from_adjacency({head, mod}, input_graph);
                            const float v = input(arc.first, arc.second);
                            return v;
                        }
                    }
            );

            auto output = batch_matrix(fx, batch);

            for (unsigned head = 0u ; head < eisner_dim ; ++head)
            {
                for (unsigned mod = 1u; mod < eisner_dim ; ++mod)
                {
                    const auto arc = diffdp::from_adjacency({head, mod}, output_graph);
                    if (head == mod)
                    {
                        output(arc.first, arc.second) = 0.f;
                        continue;
                    }

                    if (head == 0u && !with_root_arcs)
                    {
                        output(arc.first, arc.second) = 0.f;
                        continue;
                    }

                    const float a = _ce_ptr2[batch]->output(head, mod);

                    if (!std::isfinite(a))
                        throw std::runtime_error("BAD eisner output");

                    output(arc.first, arc.second) = a;
                }
            }
        }
        else
        {
            throw std::runtime_error("Not implemented: only ForwardRegularized can be used at the moment");
        }
    }
#endif
}

template<class MyDevice>
void EntropyRegularizedEisner::backward_dev_impl(
        const MyDevice &,
        const std::vector<const Tensor*>& xs,
        const Tensor&,
        const Tensor& dEdf,
        unsigned,
        Tensor& dEdxi
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("EntropyRegularizedEisner::backward");
#else
    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        auto output_grad = batch_matrix(dEdxi, batch);
        auto input_grad = batch_matrix(dEdf, batch);

        auto& eisner = *(_ce_ptr.at(batch));

        eisner.backward(
                [&] (unsigned head, unsigned mod) -> float
                {
                    if (head == 0u && !with_root_arcs)
                        return 0.f;
                    auto arc = diffdp::from_adjacency({head, mod}, output_graph);
                    const float v = input_grad(arc.first, arc.second);
                    if (!std::isfinite(v))
                        throw std::runtime_error("BAD eisner input grad");
                    return v;
                }
        );

        for (unsigned head = 0u ; head < eisner.size() ; ++head)
        {
            for (unsigned mod = 1u; mod < eisner.size(); ++mod)
            {
                if (head == mod)
                    continue;

                if (head == 0u && !with_root_arcs)
                    return;

                auto const v = eisner.gradient(head, mod);
                if (!std::isfinite(v))
                    throw std::runtime_error("BAD eisner output grad");

                auto arc = diffdp::from_adjacency({head, mod}, input_graph);
                output_grad(arc.first, arc.second) += v;
            }
        }
    }
#endif
}

DYNET_NODE_INST_DEV_IMPL(EntropyRegularizedEisner)

}