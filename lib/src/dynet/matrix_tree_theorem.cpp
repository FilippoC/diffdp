#include "diffdp/dynet/matrix_tree_theorem.h"

#include "dynet/tensor-eigen.h"
#include "dynet/nodes-impl-macros.h"

namespace dynet
{

Expression matrix_tree_theorem(const Expression &weights)
{
    return Expression(weights.pg, weights.pg->add_function<MatrixTreeTheorem>({weights.i}));
}


MatrixTreeTheorem::MatrixTreeTheorem(
        const std::initializer_list<VariableIndex>& a
) :
        Node(a)
{
    this->has_cuda_implemented = false;
}

bool MatrixTreeTheorem::supports_multibatch() const
{
    return false;
}


std::string MatrixTreeTheorem::as_string(const std::vector<std::string>& arg_names) const {
    std::ostringstream s;
    s << "matrix_tree_theorem(" << arg_names[0] << ")";
    return s.str();
}

Dim MatrixTreeTheorem::dim_forward(const std::vector<Dim>& xs) const {
    return xs[0];
}

size_t MatrixTreeTheorem::aux_storage_size() const
{
    const auto matrix_size = dim.rows() * dim.cols();
    // 1. exp weights
    // 2. laplacian inverse
    // 3. output1
    // 4. output2
    return sizeof(float) * matrix_size * 4;
}

template<class MyDevice>
void MatrixTreeTheorem::forward_dev_impl(
        const MyDevice& dev,
        const std::vector<const Tensor*>& xs,
        Tensor& fx
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("MatrixTreeTheorem::forward");
#else
    // aux mem
    const Dim matrix_dim({fx.d.cols(), fx.d.rows()});
    const unsigned matrix_size = fx.d.cols() * fx.d.rows();

    float* f_aux_mem = (float*) aux_mem;
    Tensor tensor_exp_weights(matrix_dim, f_aux_mem, fx.device, DeviceMempool::FXS);
    Tensor tensor_laplacian(matrix_dim, f_aux_mem + matrix_size, fx.device, DeviceMempool::FXS);
    Tensor tensor_output1(matrix_dim, f_aux_mem + 2*matrix_size, fx.device, DeviceMempool::FXS);
    Tensor tensor_output2(matrix_dim, f_aux_mem + 3*matrix_size, fx.device, DeviceMempool::FXS);

    // temp mem
    AlignedMemoryPool* scratch_allocator = fx.device->pools[(int)DeviceMempool::SCS];
    Tensor tensor_col_sum(Dim({fx.d.cols()}), nullptr, fx.device, DeviceMempool::FXS);
    tensor_col_sum.v = static_cast<float*>(scratch_allocator->allocate(tensor_col_sum.d.size() * sizeof(float)));

    auto weights = mat(*xs[0]);
    auto exp_weights = mat(tensor_exp_weights);
    auto col_sum = vec(tensor_col_sum);
    auto laplacian = mat(tensor_laplacian);
    auto output1 = mat(tensor_output1);
    auto output2 = mat(tensor_output2);
    auto marginals = mat(fx);

    exp_weights = weights.array().exp();

    // sum over columns
    col_sum = exp_weights.colwise().sum();

    // set fx = laplacian
    laplacian = -exp_weights;
    laplacian.diagonal() += col_sum;

    laplacian.row(0).setZero();
    laplacian(0, 0) = 1.f; // anythinhg > 0 will work here

    for (unsigned i = 0 ; i < 3 ; ++i)
    {
        for (unsigned j = 0 ; j < 3 ; ++j)
            std::cerr << laplacian(i, j) << "\t";
        std::cerr << std::endl;
    }
    // inverse
    laplacian = laplacian.inverse();

    // on gpu it may be faster to use masked matrix?
    output1.col(0).setZero();
    for (unsigned i = 1 ; i < fx.d.rows() ; ++i)
        output1.col(i) = exp_weights.col(i) * laplacian(i, i);

    // array because it a cwise product, not a matrix product
    output2 = exp_weights.array() * laplacian.transpose().array();
    output2.row(0).setZero();

    marginals = output1 - output2;

    scratch_allocator->free();
#endif
}

template<class MyDevice>
void MatrixTreeTheorem::backward_dev_impl(
        const MyDevice &,
        const std::vector<const Tensor*>& xs,
        const Tensor& fx,
        const Tensor& dEdf,
        unsigned,
        Tensor& dEdxi
) const {
#ifdef __CUDACC__
    DYNET_NO_CUDA_IMPL_ERROR("AlgorithmicDifferentiableEisner::backward");
#else

    const Dim matrix_dim({fx.d.cols(), fx.d.rows()});
    const unsigned matrix_size = fx.d.cols() * fx.d.rows();

    float* f_aux_mem = (float*) aux_mem;
    Tensor tensor_exp_weights(matrix_dim, f_aux_mem, fx.device, DeviceMempool::FXS);
    Tensor tensor_laplacian(matrix_dim, f_aux_mem + matrix_size, fx.device, DeviceMempool::FXS);
    Tensor tensor_output1(matrix_dim, f_aux_mem + 2*matrix_size, fx.device, DeviceMempool::FXS);
    Tensor tensor_output2(matrix_dim, f_aux_mem + 3*matrix_size, fx.device, DeviceMempool::FXS);

    auto weights = mat(*xs[0]);
    auto exp_weights = mat(tensor_exp_weights);
    auto laplacian = mat(tensor_laplacian);
    auto output1 = mat(tensor_output1);
    auto output2 = mat(tensor_output2);
    auto marginals = mat(fx);

    auto d_marginals = mat(dEdf);
    auto d_weights = mat(dEdxi);


#endif
}


DYNET_NODE_INST_DEV_IMPL(MatrixTreeTheorem)

}