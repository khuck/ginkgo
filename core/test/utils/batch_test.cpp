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

#include "core/test/utils/batch.hpp"


#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace {


class BatchGenerator : public ::testing::Test {
protected:
    using Mtx = gko::matrix::BatchDense<double>;
    BatchGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              bsize.get_num_batch_entries(), bsize.at()[0], bsize.at()[1],
              std::normal_distribution<double>(50, 5),
              std::normal_distribution<double>(20.0, 5.0), std::ranlux48(42),
              false, exec)),
          nnz_per_row_sample(bsize.get_num_batch_entries() * bsize.at()[0], 0),
          values_sample(0)
    {
        // collect samples of nnz/row and values from the matrix
        for (size_t ibatch = 0; ibatch < bsize.get_num_batch_entries();
             ibatch++) {
            for (int row = 0; row < mtx->get_size().at(ibatch)[0]; ++row) {
                for (int col = 0; col < mtx->get_size().at(ibatch)[1]; ++col) {
                    auto val = mtx->at(ibatch, row, col);
                    if (val != 0.0) {
                        ++nnz_per_row_sample[ibatch * bsize.at()[0] + row];
                        values_sample.push_back(val);
                    }
                }
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    const gko::batch_dim<2> bsize{8, gko::dim<2>(254, 100)};
    std::unique_ptr<Mtx> mtx;
    std::vector<int> nnz_per_row_sample;
    std::vector<double> values_sample;

    template <typename InputIterator, typename ValueType>
    ValueType get_nth_moment(int n, ValueType c, InputIterator sample_start,
                             InputIterator sample_end)
    {
        using std::pow;
        ValueType res = 0;
        ValueType num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(tmp - c, n);
            num_elems += 1;
        }
        return res / num_elems;
    }
};


TEST_F(BatchGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(mtx->get_size(), bsize);
}


TEST_F(BatchGenerator, OutputHasCorrectNonzeroAverageAndDeviation)
{
    using std::sqrt;
    auto average = get_nth_moment(1, 0.0, begin(nnz_per_row_sample),
                                  end(nnz_per_row_sample));
    auto deviation = sqrt(get_nth_moment(2, average, begin(nnz_per_row_sample),
                                         end(nnz_per_row_sample)));

    // check that average & deviation is within 10% of the required amount
    ASSERT_NEAR(average, 50.0, 5);
    ASSERT_NEAR(deviation, 5.0, 0.5);
}


TEST_F(BatchGenerator, OutputHasCorrectValuesAverageAndDeviation)
{
    using std::sqrt;
    auto average =
        get_nth_moment(1, 0.0, begin(values_sample), end(values_sample));
    auto deviation = sqrt(
        get_nth_moment(2, average, begin(values_sample), end(values_sample)));

    // check that average and deviation is within 10% of the required amount
    ASSERT_NEAR(average, 20.0, 2.0);
    ASSERT_NEAR(deviation, 5.0, 0.5);
}


TEST_F(BatchGenerator, OutputHasAllDiagonalEntriesWhenRequested)
{
    using Csr = gko::matrix::BatchCsr<double>;
    const gko::size_type nbatch = 3, nrows = 10, ncols = 10;
    auto dmtx = gko::test::generate_uniform_batch_random_matrix<Csr>(
        nbatch, nrows, ncols, std::uniform_int_distribution<>(1, 3),
        std::normal_distribution<double>(20.0, 5.0), std::ranlux48(42), true,
        exec);

    const int* const row_ptrs = dmtx->get_const_row_ptrs();
    const int* const col_idxs = dmtx->get_const_col_idxs();
    const double* const vals = dmtx->get_const_values();
    for (size_t row = 0; row < nrows; row++) {
        bool has_diag = false;
        for (int iz = row_ptrs[row]; iz < row_ptrs[row + 1]; iz++) {
            if (col_idxs[iz] == row) {
                has_diag = true;
                for (size_t ibatch = 0; ibatch < nbatch; ibatch++) {
                    if (vals[ibatch * row_ptrs[nrows] + iz] == 0.0) {
                        has_diag = false;
                    }
                }
            }
        }
        ASSERT_TRUE(has_diag);
    }
}


}  // namespace
