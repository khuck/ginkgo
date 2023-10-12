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

#include "core/matrix/batch_ell_kernels.hpp"


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


class Ell : public CommonTestFixture {
protected:
    using Mtx = gko::batch::matrix::Ell<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;

    Ell() : rand_engine(15) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(const gko::size_type num_batch_items,
                                     gko::size_type num_rows,
                                     gko::size_type num_cols,
                                     int num_elems_per_row)
    {
        return gko::test::generate_random_batch_matrix<MtxType>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_elems_per_row,
                                            num_elems_per_row),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref,
            num_elems_per_row);
    }

    std::unique_ptr<MVec> gen_mvec(const gko::size_type num_batch_items,
                                   gko::size_type num_rows,
                                   gko::size_type num_cols)
    {
        return gko::test::generate_random_batch_matrix<MVec>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(gko::size_type num_vecs = 1,
                           int num_elems_per_row = 5)
    {
        const int num_rows = 252;
        const int num_cols = 32;
        x = gen_mtx<Mtx>(batch_size, num_rows, num_cols, num_elems_per_row);
        y = gen_mvec(batch_size, num_cols, num_vecs);
        alpha = gen_mvec(batch_size, 1, 1);
        beta = gen_mvec(batch_size, 1, 1);
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        expected = MVec::create(
            ref,
            gko::batch_dim<2>(batch_size, gko::dim<2>{num_rows, num_vecs}));
        expected->fill(gko::one<value_type>());
        dresult = gko::clone(exec, expected);
    }

    std::ranlux48 rand_engine;

    const size_t batch_size = 11;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<MVec> y;
    std::unique_ptr<MVec> alpha;
    std::unique_ptr<MVec> beta;
    std::unique_ptr<MVec> expected;
    std::unique_ptr<MVec> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<MVec> dy;
    std::unique_ptr<MVec> dalpha;
    std::unique_ptr<MVec> dbeta;
};


TEST_F(Ell, SingleVectorApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(Ell, SingleVectorAdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data(1);

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}
