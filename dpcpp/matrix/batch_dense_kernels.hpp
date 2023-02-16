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
#ifndef GKO_DPCPP_MATRIX_BATCH_DENSE_KERNELS_HPP_
#define GKO_DPCPP_MATRIX_BATCH_DENSE_KERNELS_HPP_


#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>

#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BatchDense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {

template <typename ValueType>
inline void single_matvec_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < a.num_rows; row += num_sg) {
        ValueType temp = zero<ValueType>();
        for (int j = sg.get_local_id(); j < a.num_rhs; j += sg_size) {
            const ValueType val = a.values[row * a.stride + j];
            temp += val * b[j];
        }
        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());
        if (sg.get_local_id() == 0) {
            c[row] = temp;
        }
    }
}


template <typename ValueType>
inline void single_advanced_matvec_kernel(
    sycl::nd_item<3>& item_ct1, const ValueType alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < a.num_rows; row += num_sg) {
        ValueType temp = zero<ValueType>();
        for (int j = sg.get_local_id(); j < a.num_rhs; j += sg_size) {
            const ValueType val = a.values[row * a.stride + j];
            temp += alpha * val * b[j];
        }
        temp = sycl::reduce_over_group(sg, temp, sycl::plus<>());
        if (sg.get_local_id() == 0) {
            c[row] = temp + beta * c[row];
        }
    }
}


template <typename ValueType>
inline void single_scale_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<ValueType>& x)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_linear_id(); li < max_li;
         li += item_ct1.get_local_range().size()) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            x.values[row * x.stride + col] =
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            x.values[row * x.stride + col] =
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}


template <typename ValueType>
inline void single_add_scaled_kernel(sycl::nd_item<3>& item_ct1,
                                     const int num_rows, const ValueType alpha,
                                     const ValueType* const x,
                                     ValueType* const y)
{
    for (int li = item_ct1.get_local_id(2); li < num_rows;
         li += item_ct1.get_local_range(2)) {
        y[li] += alpha * x[li];
    }
}

template <typename ValueType>
inline void add_scaled_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}


template <typename ValueType>
inline void add_scaled_advanced_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& beta,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    const int max_li = x.num_rows * x.num_rhs;
    for (int li = item_ct1.get_local_id(2); li < max_li;
         li += item_ct1.get_local_range(2)) {
        const int row = li / x.num_rhs;
        const int col = li % x.num_rhs;

        if (alpha.num_rhs == 1) {
            y.values[row * y.stride + col] *= beta.values[0];
            y.values[row * y.stride + col] +=
                alpha.values[0] * x.values[row * x.stride + col];
        } else {
            y.values[row * y.stride + col] *= beta.values[col];
            y.values[row * y.stride + col] +=
                alpha.values[col] * x.values[row * x.stride + col];
        }
    }
}


template <typename ValueType>
inline void compute_dot_product_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& y,
    const gko::batch_dense::BatchEntry<ValueType>& result)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        ValueType val = zero<ValueType>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size) {
            val += conj(x.values[r * x.stride + rhs_index]) *
                   y.values[r * y.stride + rhs_index];
        }

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) {
            result.values[rhs_index] = val;
        }
    }
}


template <typename ValueType>
inline void compute_norm2_kernel(
    sycl::nd_item<3>& item_ct1,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<remove_complex<ValueType>>& result)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    using real_type = typename gko::remove_complex<ValueType>;
    for (int rhs_index = sg_id; rhs_index < x.num_rhs; rhs_index += num_sg) {
        real_type val = zero<real_type>();

        for (int r = sg.get_local_id(); r < x.num_rows; r += sg_size)
            val += squared_norm(x.values[r * x.stride + rhs_index]);

        val = sycl::reduce_over_group(sg, val, sycl::plus<>());

        if (sg.get_local_id() == 0) result.values[rhs_index] = sqrt(val);
    }
}


template <typename Op, typename ValueType>
inline void transpose_kernel(sycl::nd_item<3>& item_ct1, const int src_nrows,
                             const int src_ncols, const size_type src_stride,
                             const ValueType* const src,
                             const size_type dest_stride, ValueType* const dest,
                             Op op)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int i_row = sg_id; i_row < src_nrows; i_row += num_sg) {
        for (int j = sg.get_local_id(); j < src_ncols; j += sg_size) {
            dest[j * dest_stride + i_row] = op(src[i_row * src_stride + j]);
        }
    }
}


template <typename ValueType>
inline void copy_kernel(sycl::nd_item<3>& item_ct1,
                        const gko::batch_dense::BatchEntry<const ValueType>& in,
                        const gko::batch_dense::BatchEntry<ValueType>& out)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < in.num_rows * in.num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


template <typename ValueType>
inline void batch_scale_kernel(sycl::nd_item<3>& item_ct1, const int num_rows,
                               const size_type stride, const int num_rhs,
                               const ValueType* const left_scale_vec,
                               const ValueType* const right_scale_vec,
                               ValueType* const a)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= left_scale_vec[row] * right_scale_vec[col];
    }
}


template <typename ValueType>
inline void add_scaled_identity_kernel(sycl::nd_item<3>& item_ct1,
                                       const int nrows, const int ncols,
                                       const size_type stride,
                                       ValueType* const __restrict__ values,
                                       const ValueType alpha,
                                       const ValueType beta)
{
    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    for (int row = sg_id; row < nrows; row += num_sg) {
        for (int col = sg.get_local_id(); col < ncols; col += sg_size) {
            values[row * stride + col] *= beta;
            if (col == row) {
                values[row * stride + row] += alpha;
            }
        }
    }
}


}  // namespace batch_dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif
