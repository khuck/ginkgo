/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/solver/batch_cg_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchCg : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options = gko::kernels::batch_cg::BatchCgOptions<real_type>;

    BatchCg()
        : exec(gko::ReferenceExecutor::create()),
          ompexec(gko::OmpExecutor::create()),
          xex_1(gko::batch_initialize<BDense>(nbatch, {1.0, 3.0, 2.0}, exec)),
          b_1(gko::batch_initialize<BDense>(nbatch, {-1.0, 3.0, 1.0}, exec)),
          d_b_1(gko::clone(ompexec, b_1)),
          xex_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
              exec)),
          b_m(gko::batch_initialize<BDense>(
              nbatch,
              std::initializer_list<std::initializer_list<value_type>>{
                  {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
              exec)),
          d_b_m(gko::clone(ompexec, b_m))
    {}

    // NOTE: Avoided extra copies b/w Reference and OmpExecutor while accessing
    // the objects' underneath arrays as the objects which have exec: omp can be
    // accessed directly, so copying of objects to the Reference Executor is not
    // required. At the same time, made sure that everything is on the same
    // executor while calling apply on linop or batchlinops.
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::OmpExecutor> ompexec;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::shared_ptr<const BDense> b_1;
    std::shared_ptr<const BDense> d_b_1;
    std::shared_ptr<const BDense> xex_1;
    std::shared_ptr<RBDense> bnorm_1;
    const Options opts_1{"none", 500, 1e-6, 1e-11,
                         gko::stop::batch::ToleranceType::absolute};

    const int nrhs = 2;
    std::shared_ptr<const BDense> b_m;
    std::shared_ptr<const BDense> d_b_m;
    std::shared_ptr<const BDense> xex_m;
    std::shared_ptr<RBDense> bnorm_m;
    const Options opts_m{"none", 500, 1e-6, 1e-11,
                         gko::stop::batch::ToleranceType::absolute};

    struct Result {
        std::shared_ptr<BDense> d_x;
        std::shared_ptr<RBDense> resnorm;
        gko::log::BatchLogData<value_type> d_logdata;
        std::shared_ptr<BDense> residual;
    };
    Result r_1;
    Result r_m;

    Result solve_poisson_uniform_1(const Options opts,
                                   const BDense *const d_left_scale = nullptr,
                                   const BDense *const d_right_scale = nullptr)
    {
        bnorm_1 = gko::batch_initialize<RBDense>(nbatch, {0.0}, exec);
        b_1->compute_norm2(bnorm_1.get());

        const int nrhs_1 = 1;

        auto orig_mtx = gko::test::create_poisson1d_batch<value_type>(
            this->exec, nrows, nbatch);

        auto d_mtx = gko::clone(ompexec, orig_mtx);

        Result result;

        result.residual = b_1->clone();
        result.d_x = gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0},
                                                   this->ompexec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs_1));
        result.d_logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->ompexec, sizes);
        result.d_logdata.iter_counts.set_executor(this->ompexec);
        result.d_logdata.iter_counts.resize_and_reset(nrhs_1 * nbatch);

        gko::kernels::omp::batch_cg::apply<value_type>(
            this->ompexec, opts, d_mtx.get(), d_left_scale, d_right_scale,
            d_b_1.get(), result.d_x.get(), result.d_logdata);

        result.resnorm =
            gko::batch_initialize<RBDense>(nbatch, {0.0}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        orig_mtx->apply(alpha.get(), gko::clone(exec, result.d_x).get(),
                        beta.get(), result.residual.get());
        result.residual->compute_norm2(result.resnorm.get());
        return result;
    }


    int single_iters_regression() const
    {
        if (std::is_same<real_type, float>::value) {
            return 5;
        } else if (std::is_same<real_type, double>::value) {
            return 3;
        } else {
            return -1;
        }
    }

    Result solve_poisson_uniform_mult()
    {
        bnorm_m = gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, exec);
        b_m->compute_norm2(bnorm_m.get());

        const int nrows = 3;
        auto mtx = gko::test::create_poisson1d_batch<value_type>(this->exec,
                                                                 nrows, nbatch);
        auto d_mtx = gko::clone(this->ompexec, mtx);

        Result result;
        result.d_x = gko::batch_initialize<BDense>(
            nbatch,
            std::initializer_list<std::initializer_list<value_type>>{
                {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
            this->ompexec);

        std::vector<gko::dim<2>> sizes(nbatch, gko::dim<2>(1, nrhs));
        result.d_logdata.res_norms =
            gko::matrix::BatchDense<real_type>::create(this->ompexec, sizes);
        result.d_logdata.iter_counts.set_executor(this->ompexec);
        result.d_logdata.iter_counts.resize_and_reset(nrhs * nbatch);

        gko::kernels::omp::batch_cg::apply<value_type>(
            this->ompexec, opts_m, d_mtx.get(), nullptr, nullptr, d_b_m.get(),
            result.d_x.get(), result.d_logdata);

        result.residual = b_m->clone();
        result.resnorm =
            gko::batch_initialize<RBDense>(nbatch, {{0.0, 0.0}}, this->exec);
        auto alpha = gko::batch_initialize<BDense>(nbatch, {-1.0}, this->exec);
        auto beta = gko::batch_initialize<BDense>(nbatch, {1.0}, this->exec);
        mtx->apply(alpha.get(), gko::clone(exec, result.d_x).get(), beta.get(),
                   result.residual.get());
        result.residual->compute_norm2(result.resnorm.get());
        return result;
    }


    std::vector<int> multiple_iters_regression() const
    {
        std::vector<int> iters(2);
        if (std::is_same<real_type, float>::value) {
            iters[0] = 5;
            iters[1] = 6;
        } else if (std::is_same<real_type, double>::value) {
            iters[0] = 3;
            iters[1] = 3;
        } else {
            iters[0] = -1;
            iters[1] = -1;
        }
        return iters;
    }
};

TYPED_TEST_SUITE(BatchCg, gko::test::ValueTypes);


TYPED_TEST(BatchCg, SolvesStencilSystemNone)
{
    this->r_1 = this->solve_poisson_uniform_1(this->opts_1);

    GKO_ASSERT_BATCH_MTX_NEAR(this->r_1.d_x, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}

TYPED_TEST(BatchCg, StencilSystemNoneLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    this->r_1 = this->solve_poisson_uniform_1(this->opts_1);

    const int ref_iters = this->single_iters_regression();

    const int *const iter_array =
        this->r_1.d_logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_1.d_logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        ASSERT_EQ(iter_array[i], ref_iters);
        ASSERT_LE(res_log_array[i], this->opts_1.abs_residual_tol);
        ASSERT_NEAR(res_log_array[i], this->r_1.resnorm->get_const_values()[i],
                    10 * r<value_type>::value);
    }
}


TYPED_TEST(BatchCg, SolvesStencilMultipleSystemNone)
{
    this->r_m = this->solve_poisson_uniform_mult();

    GKO_ASSERT_BATCH_MTX_NEAR(this->r_m.d_x, this->xex_m,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchCg, StencilMultipleSystemNoneLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    this->r_m = this->solve_poisson_uniform_mult();

    const std::vector<int> ref_iters = this->multiple_iters_regression();

    const int *const iter_array =
        this->r_m.d_logdata.iter_counts.get_const_data();
    const real_type *const res_log_array =
        this->r_m.d_logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        // test logger
        for (size_t j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(iter_array[i * this->nrhs + j], ref_iters[j]);
            ASSERT_LE(res_log_array[i * this->nrhs + j],
                      this->opts_m.abs_residual_tol);

            ASSERT_NEAR(
                res_log_array[i * this->nrhs + j],
                this->r_m.resnorm->get_const_values()[i * this->nrhs + j],
                10 * r<value_type>::value);
        }
    }
}


TYPED_TEST(BatchCg, UnitScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    auto d_left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->ompexec);
    auto d_right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->ompexec);

    Result result = this->solve_poisson_uniform_1(
        this->opts_1, d_left_scale.get(), d_right_scale.get());


    GKO_ASSERT_BATCH_MTX_NEAR(result.d_x, this->xex_1,
                              1e-6 /*r<value_type>::value*/);
}


TYPED_TEST(BatchCg, GeneralScalingDoesNotChangeResult)
{
    using Result = typename TestFixture::Result;
    using BDense = typename TestFixture::BDense;
    using Options = typename TestFixture::Options;
    auto d_left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->ompexec);
    auto d_right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->ompexec);


    Result result = this->solve_poisson_uniform_1(
        this->opts_1, d_left_scale.get(), d_right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.d_x, this->xex_1,
                              1e-06 /*r<value_type>::value*/);
}


}  // namespace