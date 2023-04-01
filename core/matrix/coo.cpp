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

#include <ginkgo/core/matrix/coo.hpp>


#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/absolute_array_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_kernels.hpp"
#include "core/matrix/coo_kernels.hpp"


namespace gko {
namespace matrix {
namespace coo {
namespace {


GKO_REGISTER_OPERATION(spmv, coo::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, coo::advanced_spmv);
GKO_REGISTER_OPERATION(spmv2, coo::spmv2);
GKO_REGISTER_OPERATION(advanced_spmv2, coo::advanced_spmv2);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(fill_in_dense, coo::fill_in_dense);
GKO_REGISTER_OPERATION(convert_to_bccoo, coo::convert_to_bccoo);
GKO_REGISTER_OPERATION(extract_diagonal, coo::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);
GKO_REGISTER_OPERATION(mem_size_bccoo, coo::mem_size_bccoo);


}  // anonymous namespace
}  // namespace coo


namespace bccoo {


GKO_REGISTER_OPERATION(get_default_block_size, bccoo::get_default_block_size);
GKO_REGISTER_OPERATION(get_default_compression, bccoo::get_default_compression);


}  // namespace bccoo


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->get_executor()->run(coo::make_advanced_spmv(
                dense_alpha, this, dense_b, dense_beta, dense_x));
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(coo::make_spmv2(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::apply2_impl(const LinOp* alpha, const LinOp* b,
                                            LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_x) {
            this->get_executor()->run(
                coo::make_advanced_spmv2(dense_alpha, this, dense_b, dense_x));
        },
        alpha, b, x);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Coo<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->row_idxs_ = this->row_idxs_;
    result->col_idxs_ = this->col_idxs_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(
    Coo<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Bccoo<ValueType, IndexType>* result) const
{
    // Definition of executors
    auto exec = this->get_executor();
    auto exec_master = exec->get_master();

    // Compression. If the initial value is def_value, the default is chosen
    bccoo::compression compression = result->get_compression();
    if (result->use_default_compression()) {
        exec->run(bccoo::make_get_default_compression(&compression));
    }

    // Block partitioning. If the initial value is 0, the default is chosen
    IndexType block_size = result->get_block_size();
    if (block_size == 0) {
        // exec->run(bccoo::make_get_default_block_size(&block_size));
        size_type aux;
        exec->run(bccoo::make_get_default_block_size(&aux));
        block_size = aux;
    }

    // Computation of nnz
    IndexType num_stored_elements = this->get_num_stored_elements();

    // Creating the result
    size_type mem_size{};
    if (exec == exec_master) {
        exec->run(
            coo::make_mem_size_bccoo(this, block_size, compression, &mem_size));
        auto tmp = Bccoo<ValueType, IndexType>::create(
            exec, this->get_size(), num_stored_elements, block_size, mem_size,
            compression);
        exec->run(coo::make_convert_to_bccoo(this, tmp.get()));
        *result = *tmp;
    } else {
        auto host_coo = this->clone(exec_master);
        exec_master->run(coo::make_mem_size_bccoo(host_coo.get(), block_size,
                                                  compression, &mem_size));
        auto tmp = Bccoo<ValueType, IndexType>::create(
            exec_master, host_coo->get_size(), num_stored_elements, block_size,
            mem_size, compression);
        exec_master->run(coo::make_convert_to_bccoo(host_coo.get(), tmp.get()));
        *result = *tmp;
    }
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Bccoo<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();
    result->set_size(this->get_size());
    result->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
    result->col_idxs_ = this->col_idxs_;
    result->values_ = this->values_;
    exec->run(coo::make_convert_idxs_to_ptrs(
        this->get_const_row_idxs(), this->get_num_stored_elements(),
        this->get_size()[0],
        make_temporary_clone(exec, &result->row_ptrs_)->get_data()));
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    auto exec = this->get_executor();
    const auto nnz = this->get_num_stored_elements();
    result->set_size(this->get_size());
    result->row_ptrs_.resize_and_reset(this->get_size()[0] + 1);
    result->col_idxs_ = std::move(this->col_idxs_);
    result->values_ = std::move(this->values_);
    exec->run(coo::make_convert_idxs_to_ptrs(
        this->get_const_row_idxs(), nnz, this->get_size()[0],
        make_temporary_clone(exec, &result->row_ptrs_)->get_data()));
    result->make_srow();
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp_result = make_temporary_output_clone(exec, result);
    tmp_result->resize(this->get_size());
    tmp_result->fill(zero<ValueType>());
    exec->run(coo::make_fill_in_dense(this, tmp_result.get()));
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::resize(dim<2> new_size, size_type nnz)
{
    this->set_size(new_size);
    this->row_idxs_.resize_and_reset(nnz);
    this->col_idxs_.resize_and_reset(nnz);
    this->values_.resize_and_reset(nnz);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_from_host(this->get_executor(), data));
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(const device_mat_data& data)
{
    // make a copy, read the data in
    this->read(device_mat_data{this->get_executor(), data});
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::read(device_mat_data&& data)
{
    this->set_size(data.get_size());
    auto arrays = data.empty_out();
    this->values_ = std::move(arrays.values);
    this->col_idxs_ = std::move(arrays.col_idxs);
    this->row_idxs_ = std::move(arrays.row_idxs);
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::write(mat_data& data) const
{
    auto tmp = make_temporary_clone(this->get_executor()->get_master(), this);

    data = {this->get_size(), {}};

    for (size_type i = 0; i < tmp->get_num_stored_elements(); ++i) {
        const auto row = tmp->row_idxs_.get_const_data()[i];
        const auto col = tmp->col_idxs_.get_const_data()[i];
        const auto val = tmp->values_.get_const_data()[i];
        data.nonzeros.emplace_back(row, col, val);
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Coo<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(coo::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(coo::make_extract_diagonal(this, diag.get()));
    return diag;
}


template <typename ValueType, typename IndexType>
void Coo<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(coo::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Coo<ValueType, IndexType>::absolute_type>
Coo<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_coo = absolute_type::create(exec, this->get_size(),
                                         this->get_num_stored_elements());

    abs_coo->col_idxs_ = col_idxs_;
    abs_coo->row_idxs_ = row_idxs_;
    exec->run(coo::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_coo->get_values()));

    return abs_coo;
}


#define GKO_DECLARE_COO_MATRIX(ValueType, IndexType) \
    class Coo<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_MATRIX);


}  // namespace matrix
}  // namespace gko
