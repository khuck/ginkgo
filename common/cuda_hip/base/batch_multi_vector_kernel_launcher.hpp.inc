/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const batch::MultiVector<ValueType>* const alpha,
           batch::MultiVector<ValueType>* const x)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    if (alpha->get_common_size()[1] == 1) {
        scale_kernel<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            alpha_ub, x_ub, [] __device__(int col) { return 0; });
    } else {
        scale_kernel<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            alpha_ub, x_ub, [] __device__(int col) { return col; });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const batch::MultiVector<ValueType>* const alpha,
                const batch::MultiVector<ValueType>* const x,
                batch::MultiVector<ValueType>* const y)
{
    const auto num_blocks = x->get_num_batch_items();
    const size_type nrhs = x->get_common_size()[1];
    const auto alpha_ub = get_batch_struct(alpha);
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    if (alpha->get_common_size()[1] == 1) {
        add_scaled_kernel<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(
            alpha_ub, x_ub, y_ub, [] __device__(int col) { return 0; });
    } else {
        add_scaled_kernel<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(
            alpha_ub, x_ub, y_ub, [] __device__(int col) { return col; });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DefaultExecutor> exec,
                 const batch::MultiVector<ValueType>* x,
                 const batch::MultiVector<ValueType>* y,
                 batch::MultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);
    compute_gen_dot_product_kernel<<<num_blocks, default_block_size, 0,
                                     exec->get_stream()>>>(
        x_ub, y_ub, res_ub, [] __device__(auto val) { return val; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const batch::MultiVector<ValueType>* x,
                      const batch::MultiVector<ValueType>* y,
                      batch::MultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto y_ub = get_batch_struct(y);
    const auto res_ub = get_batch_struct(result);
    compute_gen_dot_product_kernel<<<num_blocks, default_block_size, 0,
                                     exec->get_stream()>>>(
        x_ub, y_ub, res_ub, [] __device__(auto val) { return conj(val); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const batch::MultiVector<ValueType>* const x,
                   batch::MultiVector<remove_complex<ValueType>>* const result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto num_rhs = x->get_common_size()[1];
    const auto x_ub = get_batch_struct(x);
    const auto res_ub = get_batch_struct(result);
    compute_norm2_kernel<<<num_blocks, default_block_size, 0,
                           exec->get_stream()>>>(x_ub, res_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const batch::MultiVector<ValueType>* x,
          batch::MultiVector<ValueType>* result)
{
    const auto num_blocks = x->get_num_batch_items();
    const auto result_ub = get_batch_struct(result);
    const auto x_ub = get_batch_struct(x);
    copy_kernel<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
        x_ub, result_ub);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);
