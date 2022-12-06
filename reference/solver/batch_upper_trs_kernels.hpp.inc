/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
inline void upper_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& U_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    const auto nrows = U_entry.num_rows;
    const auto nrhs = b_entry.num_rhs;
    const auto row_ptrs = U_entry.row_ptrs;
    const auto col_idxs = U_entry.col_idxs;

    // U_entry * x_entry = b_entry
    for (int rhs_idx = 0; rhs_idx < nrhs; rhs_idx++) {
        for (int row_idx = nrows - 1; row_idx >= 0; row_idx--) {
            ValueType sum = zero<ValueType>();

            for (int i = row_ptrs[row_idx + 1] - 1; i > row_ptrs[row_idx];
                 i--) {
                const int col_idx = col_idxs[i];
                sum += U_entry.values[i] *
                       x_entry.values[col_idx * x_entry.stride + rhs_idx];
            }

            ValueType diag_ele = U_entry.values[row_ptrs[row_idx]];
            x_entry.values[row_idx * x_entry.stride + rhs_idx] =
                (b_entry.values[row_idx * b_entry.stride + rhs_idx] - sum) /
                diag_ele;
        }
    }
}

template <typename ValueType>
inline void upper_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& U_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry) GKO_NOT_IMPLEMENTED;

template <typename ValueType>
inline void upper_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& U_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    const auto nrows = U_entry.num_rows;
    const auto nrhs = b_entry.num_rhs;

    for (int rhs_idx = 0; rhs_idx < nrhs; rhs_idx++) {
        for (int row_idx = nrows - 1; row_idx >= 0; row_idx--) {
            ValueType sum = zero<ValueType>();
            for (int col_idx = nrows - 1; col_idx > row_idx; col_idx--) {
                sum += U_entry.values[row_idx * U_entry.stride + col_idx] *
                       x_entry.values[col_idx * x_entry.stride + rhs_idx];
            }

            x_entry.values[row_idx * x_entry.stride + rhs_idx] =
                (b_entry.values[row_idx * b_entry.stride + rhs_idx] - sum) /
                U_entry.values[row_idx * U_entry.stride + row_idx];
        }
    }
}


template <typename BatchMatrixType, typename ValueType>
inline void batch_entry_upper_trsv_impl(
    const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_t ibatch,
    unsigned char* const local_space = nullptr)
{
    const typename BatchMatrixType::entry_type A_entry =
        gko::batch::batch_entry(a, ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(b, ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);

    upper_trsv(A_entry, b_entry, x_entry);
}