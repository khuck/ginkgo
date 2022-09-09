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
inline void batch_entry_factorize_impl(
    const int* const diag_locs,
    const batch_csr::BatchEntry<ValueType>& mat_factorized_entry)
{
    const int nrows = mat_factorized_entry.num_rows;
    const int nnz = mat_factorized_entry.num_nnz;
    const int* row_ptrs = mat_factorized_entry.row_ptrs;
    const int* col_idxs = mat_factorized_entry.col_idxs;
    ValueType* values = mat_factorized_entry.values;

    for (int curr_row_index = 0; curr_row_index < nrows; curr_row_index++) {
        for (int row_below_index = curr_row_index + 1; row_below_index < nrows;
             row_below_index++) {
            ValueType row_ele = zero<ValueType>();

            for (int i = row_ptrs[row_below_index];
                 i < row_ptrs[row_below_index + 1]; i++) {
                const int col_index = col_idxs[i];

                if (col_index == curr_row_index) {
                    ValueType diag_ele = values[diag_locs[curr_row_index]];
                    assert(diag_ele != zero<ValueType>());
                    row_ele = values[i] / diag_ele;
                    values[i] = row_ele;
                }

                if (col_index > curr_row_index) {
                    ValueType col_ele = zero<ValueType>();
                    for (int t = row_ptrs[curr_row_index];
                         t < row_ptrs[curr_row_index + 1]; t++) {
                        if (col_idxs[t] == col_index) {
                            col_ele = values[t];
                            break;
                        }
                    }

                    values[i] -= row_ele * col_ele;
                }
            }
        }
    }
}