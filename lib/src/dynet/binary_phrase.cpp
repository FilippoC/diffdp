#include "diffdp/dynet/binary_phrase.h"
#include "dynet/tensor-eigen.h"

namespace dynet
{

Expression algorithmic_differentiable_binary_phrase_structure(const Expression& x, diffdp::DiscreteMode mode, std::vector<unsigned>* batch_sizes)
{
    return Expression(x.pg, x.pg->add_function<AlgorithmicDifferentiableBinaryPhraseStructure>({x.i}, mode, batch_sizes));
}

Expression entropy_regularized_binary_phrase_structure(const Expression& x, diffdp::DiscreteMode mode, std::vector<unsigned>* batch_sizes)
{
    return Expression(x.pg, x.pg->add_function<EntropyRegularizedBinaryPhraseStructure>({x.i}, mode, batch_sizes));
}

AlgorithmicDifferentiableBinaryPhraseStructure::AlgorithmicDifferentiableBinaryPhraseStructure(
        const std::initializer_list<VariableIndex>& a,
        diffdp::DiscreteMode mode,
        std::vector<unsigned>* batch_sizes
) :
        Node(a),
        mode(mode),
        batch_sizes(batch_sizes)
{
    this->has_cuda_implemented = false;
}

bool AlgorithmicDifferentiableBinaryPhraseStructure::supports_multibatch() const
{
    return true;
}

AlgorithmicDifferentiableBinaryPhraseStructure::~AlgorithmicDifferentiableBinaryPhraseStructure()
{
    for (auto*& ptr : _ce_ptr)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
}

std::string AlgorithmicDifferentiableBinaryPhraseStructure::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "algorithmic_differentiable_eisner(" << arg_names[0] << ")";
    return s.str();
}

Dim AlgorithmicDifferentiableBinaryPhraseStructure::dim_forward(const std::vector<Dim>& xs) const {
    DYNET_ARG_CHECK(
            xs.size() == 1 && xs[0].nd == 2 && xs[0].rows() == xs[0].cols(),
            "Bad input dimensions in AlgorithmicDifferentiableEisner: " << xs
    );

    return dynet::Dim(xs[0]);
}

size_t AlgorithmicDifferentiableBinaryPhraseStructure::aux_storage_size() const {
    // 2 times because we have a forward and a backward chart
    const size_t dp_mem = 2 * diffdp::BinaryPhraseStructureChart::required_memory(dim.rows());
    return dim.batch_elems() * dp_mem;
}

template<class MyDevice>
void AlgorithmicDifferentiableBinaryPhraseStructure::forward_dev_impl(
        const MyDevice&,
        const std::vector<const Tensor*>& xs,
        Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("AlgorithmicDifferentiableEisner::forward");
#else
    // TODO call zero only when necessary
    TensorTools::zero(fx);

    std::vector<diffdp::AlgorithmicDifferentiableBinaryPhraseStructure*>& _ce_ptr2 =
            const_cast<std::vector<diffdp::AlgorithmicDifferentiableBinaryPhraseStructure*>&>(_ce_ptr);

    for (auto*& ptr : _ce_ptr2)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }

    if (_ce_ptr2.size() != xs[0]->d.batch_elems())
        _ce_ptr2.resize(xs[0]->d.batch_elems(), nullptr);

    const unsigned max_input_dim = xs[0]->d.rows();
    float* aux_fmem = static_cast<float*>(aux_mem);

    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        const unsigned eisner_dim = (
                batch_sizes == nullptr
                ? max_input_dim
                : batch_sizes->at(batch)
        );

        auto input = batch_matrix(*(xs[0]), batch);

        if (mode == diffdp::DiscreteMode::ForwardRegularized)
        {
            float* fmem = aux_fmem + batch * 2 * diffdp::BinaryPhraseStructureChart::required_cells(max_input_dim);
            //auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            //auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem + diffdp::EisnerChart::required_cells(max_eisner_dim));
            auto forward_chart = std::make_shared<diffdp::BinaryPhraseStructureChart>(eisner_dim, fmem);
            auto backward_chart = std::make_shared<diffdp::BinaryPhraseStructureChart>(eisner_dim);

            _ce_ptr2.at(batch) = new diffdp::AlgorithmicDifferentiableBinaryPhraseStructure(forward_chart, backward_chart);


            _ce_ptr2.at(batch)->forward(
                    [&] (const unsigned left, const unsigned right)
                    {
                        return input(left, right);
                    }
            );

            auto output = batch_matrix(fx, batch);

            for (unsigned left = 0u ; left < eisner_dim ; ++left)
            {
                for (unsigned right = left+1; right < eisner_dim ; ++right)
                {
                    const float a = _ce_ptr2[batch]->output(left, right);
                    output(left, right) = a;
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
void AlgorithmicDifferentiableBinaryPhraseStructure::backward_dev_impl(
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

        auto& dp = *(_ce_ptr.at(batch));

        dp.backward(
                [&] (unsigned left, unsigned right) -> float
                {
                    return input_grad(left, right);
                }
        );

        for (unsigned left = 0u ; left < dp.size() ; ++left)
            for (unsigned right = left + 1u; right < dp.size(); ++right)
                output_grad(left, right) += dp.gradient(left, right);
    }
#endif
}

DYNET_NODE_INST_DEV_IMPL(AlgorithmicDifferentiableBinaryPhraseStructure)




// ENTROPY Regularized

EntropyRegularizedBinaryPhraseStructure::EntropyRegularizedBinaryPhraseStructure(
        const std::initializer_list<VariableIndex>& a,
        diffdp::DiscreteMode mode,
        std::vector<unsigned>* batch_sizes
) :
        Node(a),
        mode(mode),
        batch_sizes(batch_sizes)
{
    this->has_cuda_implemented = false;
}

bool EntropyRegularizedBinaryPhraseStructure::supports_multibatch() const
{
    return true;
}

EntropyRegularizedBinaryPhraseStructure::~EntropyRegularizedBinaryPhraseStructure()
{
    for (auto*& ptr : _ce_ptr)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }
}

std::string EntropyRegularizedBinaryPhraseStructure::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "algorithmic_differentiable_eisner(" << arg_names[0] << ")";
    return s.str();
}

Dim EntropyRegularizedBinaryPhraseStructure::dim_forward(const std::vector<Dim>& xs) const {
    DYNET_ARG_CHECK(
            xs.size() == 1 && xs[0].nd == 2 && xs[0].rows() == xs[0].cols(),
            "Bad input dimensions in AlgorithmicDifferentiableEisner: " << xs
    );

    return dynet::Dim(xs[0]);
}

size_t EntropyRegularizedBinaryPhraseStructure::aux_storage_size() const {
    // 2 times because we have a forward and a backward chart
    const size_t dp_mem = 2 * diffdp::BinaryPhraseStructureChart::required_memory(dim.rows());
    return dim.batch_elems() * dp_mem;
}

template<class MyDevice>
void EntropyRegularizedBinaryPhraseStructure::forward_dev_impl(
        const MyDevice&,
        const std::vector<const Tensor*>& xs,
        Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("AlgorithmicDifferentiableEisner::forward");
#else
    // TODO call zero only when necessary
    TensorTools::zero(fx);

    std::vector<diffdp::EntropyRegularizedBinaryPhraseStructure*>& _ce_ptr2 =
            const_cast<std::vector<diffdp::EntropyRegularizedBinaryPhraseStructure*>&>(_ce_ptr);

    for (auto*& ptr : _ce_ptr2)
        if (ptr != nullptr)
        {
            delete ptr;
            ptr = nullptr;
        }

    if (_ce_ptr2.size() != xs[0]->d.batch_elems())
        _ce_ptr2.resize(xs[0]->d.batch_elems(), nullptr);

    const unsigned max_input_dim = xs[0]->d.rows();
    float* aux_fmem = static_cast<float*>(aux_mem);

    //#pragma omp parallel for
    for (unsigned batch = 0u ; batch < xs[0]->d.batch_elems() ; ++batch)
    {
        const unsigned eisner_dim = (
                batch_sizes == nullptr
                ? max_input_dim
                : batch_sizes->at(batch)
        );

        auto input = batch_matrix(*(xs[0]), batch);

        if (mode == diffdp::DiscreteMode::ForwardRegularized)
        {
            float* fmem = aux_fmem + batch * 2 * diffdp::BinaryPhraseStructureChart::required_cells(max_input_dim);
            //auto forward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem);
            //auto backward_chart = std::make_shared<diffdp::EisnerChart>(eisner_dim, fmem + diffdp::EisnerChart::required_cells(max_eisner_dim));
            auto forward_chart = std::make_shared<diffdp::BinaryPhraseStructureChart>(eisner_dim, fmem);
            auto backward_chart = std::make_shared<diffdp::BinaryPhraseStructureChart>(eisner_dim, fmem + diffdp::BinaryPhraseStructureChart::required_cells(max_input_dim));

            _ce_ptr2.at(batch) = new diffdp::EntropyRegularizedBinaryPhraseStructure(forward_chart, backward_chart);


            _ce_ptr2.at(batch)->forward(
                    [&] (const unsigned left, const unsigned right)
                    {
                        return input(left, right);
                    }
            );

            auto output = batch_matrix(fx, batch);

            for (unsigned left = 0u ; left < eisner_dim ; ++left)
            {
                for (unsigned right = left+1; right < eisner_dim ; ++right)
                {
                    const float a = _ce_ptr2[batch]->output(left, right);
                    output(left, right) = a;
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
void EntropyRegularizedBinaryPhraseStructure::backward_dev_impl(
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

        auto& dp = *(_ce_ptr.at(batch));

        dp.backward(
                [&] (unsigned left, unsigned right) -> float
                {
                    return input_grad(left, right);
                }
        );

        for (unsigned left = 0u ; left < dp.size() ; ++left)
            for (unsigned right = left + 1u; right < dp.size(); ++right)
            {
                output_grad(left, right) += dp.gradient(left, right);
            }
    }
#endif
}

DYNET_NODE_INST_DEV_IMPL(EntropyRegularizedBinaryPhraseStructure)



}