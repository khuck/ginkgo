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

#include <ginkgo/core/preconditioner/batch_exact_ilu.hpp>


#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/preconditioner/batch_exact_ilu_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchExactIlu : public ::testing::Test {
protected:
    using value_type = T;
    using index_type = int;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using prec_type =
        gko::preconditioner::BatchExactIlu<value_type, index_type>;

    BatchExactIlu()
        : exec(gko::ReferenceExecutor::create()),
          mtx(get_matrix()),
          diag_locs(get_corresponding_diag_locs())
    {}

    const size_t nbatch = 2;
    const index_type nrows = 3;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<const Mtx> mtx;
    const gko::array<index_type> diag_locs;


    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 6);
        index_type* const row_ptrs = mat->get_row_ptrs();
        index_type* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 2; row_ptrs[2] = 4; row_ptrs[3] = 6;
		col_idxs[0] = 0; col_idxs[1] = 1; col_idxs[2] = 0; col_idxs[3] = 1;
		col_idxs[4] = 0; col_idxs[5] = 2;
		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
		vals[10] = 2.0; vals[11] = -0.25;
        // clang-format on
        return mat;
    }

    gko::array<index_type> get_corresponding_diag_locs()
    {
        return gko::array<index_type>(
            exec, std::initializer_list<index_type>{0, 3, 5});
    }
};

TYPED_TEST_SUITE(BatchExactIlu, gko::test::ValueTypes);


template <typename ValueType, typename IndexType>
void assert_factored_a_is_same_as_l_u(
    std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>> l_factor,
    std::shared_ptr<const gko::matrix::Csr<ValueType, IndexType>> u_factor,
    const std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>& a_factorized,
    const gko::array<IndexType>& diag_locations)
{
    const int num_rows = a_factorized->get_size()[0];
    const auto l_row_ptrs = l_factor->get_const_row_ptrs();
    const auto l_col_idxs = l_factor->get_const_col_idxs();
    const auto l_vals = l_factor->get_const_values();
    const auto u_row_ptrs = u_factor->get_const_row_ptrs();
    const auto u_col_idxs = u_factor->get_const_col_idxs();
    const auto u_vals = u_factor->get_const_values();
    const auto a_row_ptrs = a_factorized->get_const_row_ptrs();
    const auto a_col_idxs = a_factorized->get_const_col_idxs();
    const auto a_vals = a_factorized->get_const_values();
    const auto diag_locations_arr = diag_locations.get_const_data();

    for (int row = 0; row < num_rows; row++) {
        assert(diag_locations_arr[row] + 1 - a_row_ptrs[row] ==
               l_row_ptrs[row + 1] - l_row_ptrs[row]);
        assert(a_row_ptrs[row + 1] - diag_locations_arr[row] ==
               u_row_ptrs[row + 1] - u_row_ptrs[row]);

        for (int i = a_row_ptrs[row]; i < diag_locations_arr[row]; i++) {
            const int idx_in_l = l_row_ptrs[row] + i - a_row_ptrs[row];
            GKO_ASSERT_NEAR(l_vals[idx_in_l], a_vals[i], 0.0);
            GKO_ASSERT_NEAR(l_col_idxs[idx_in_l], a_col_idxs[i], 0.0);
        }

        GKO_ASSERT_NEAR(
            l_vals[l_row_ptrs[row] + diag_locations_arr[row] - a_row_ptrs[row]],
            static_cast<ValueType>(1), 0.0);

        for (int i = diag_locations_arr[row]; i < a_row_ptrs[row + 1]; i++) {
            const int idx_in_u = u_row_ptrs[row] + i - diag_locations_arr[row];
            GKO_ASSERT_NEAR(u_vals[idx_in_u], a_vals[i], 0.0);
            GKO_ASSERT_NEAR(u_col_idxs[idx_in_u], a_col_idxs[i], 0.0);
        }
    }
}

TYPED_TEST(BatchExactIlu, GenerationIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;


    auto exec = this->exec;
    const auto sys_csr = this->mtx.get();
    const auto nbatch = this->nbatch;
    const auto nrows = this->nrows;
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;
    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size = gko::dim<2>(nrows, sys_csr->get_size().at(0)[1]);
    auto sys_rows_view = gko::array<index_type>::const_view(
        exec, nrows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view = gko::array<index_type>::const_view(
        exec, nnz, sys_csr->get_const_col_idxs());
    auto sys_vals_view = gko::array<value_type>::const_view(
        exec, nnz, sys_csr->get_const_values());
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));


    // initialize L and U factors (unbatched ilu needs this)
    gko::array<index_type> l_row_ptrs(exec, nrows + 1);
    gko::array<index_type> u_row_ptrs(exec, nrows + 1);
    gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
        exec, first_csr.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data());
    const auto l_nnz = exec->copy_val_to_host(&l_row_ptrs.get_data()[nrows]);
    const auto u_nnz = exec->copy_val_to_host(&u_row_ptrs.get_data()[nrows]);
    auto first_L = unbatch_type::create(exec, unbatch_size, l_nnz);
    auto first_U = unbatch_type::create(exec, unbatch_size, u_nnz);
    exec->copy(nrows + 1, l_row_ptrs.get_const_data(), first_L->get_row_ptrs());
    exec->copy(nrows + 1, u_row_ptrs.get_const_data(), first_U->get_row_ptrs());
    gko::kernels::reference::factorization::initialize_l_u(
        exec, first_csr.get(), first_L.get(), first_U.get());

    // unbatch for check
    auto mtxs = gko::test::share(sys_csr->unbatch());
    using unbatch_ilu_type = gko::factorization::Ilu<value_type>;
    auto ilu_fact = unbatch_ilu_type::build().with_skip_sorting(true).on(exec);
    std::vector<std::shared_ptr<const unbatch_type>> check_l_factors(
        mtxs.size());
    std::vector<std::shared_ptr<const unbatch_type>> check_u_factors(
        mtxs.size());
    for (size_t i = 0; i < mtxs.size(); i++) {
        auto facts = ilu_fact->generate(mtxs[i]);
        check_l_factors[i] = facts->get_l_factor();
        check_u_factors[i] = facts->get_u_factor();
    }

    auto factorize_sys_csr = gko::share(gko::clone(exec, sys_csr));
    gko::kernels::reference::batch_exact_ilu::compute_factorization(
        exec, this->diag_locs.get_const_data(), factorize_sys_csr.get());

    auto factorize_sys_csr_individual = factorize_sys_csr->unbatch();

    for (size_t i = 0; i < mtxs.size(); i++) {
        /*
        assert_factored_a_is_same_as_l_u(check_l_factors[i], check_u_factors[i],
                                         factorize_sys_csr_individual[i],
                                         this->diag_locs);
        */

        gko::array<index_type> l_factor_row_ptrs(exec, nrows + 1);
        gko::array<index_type> u_factor_row_ptrs(exec, nrows + 1);
        gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
            exec, factorize_sys_csr_individual[i].get(),
            l_factor_row_ptrs.get_data(), u_factor_row_ptrs.get_data());
        const auto l_factor_nnz = l_factor_row_ptrs.get_data()[nrows];
        const auto u_factor_nnz = u_factor_row_ptrs.get_data()[nrows];
        auto l_factor = unbatch_type::create(exec, unbatch_size, l_factor_nnz);
        auto u_factor = unbatch_type::create(exec, unbatch_size, u_factor_nnz);
        exec->copy(nrows + 1, l_factor_row_ptrs.get_const_data(),
                   l_factor->get_row_ptrs());
        exec->copy(nrows + 1, u_factor_row_ptrs.get_const_data(),
                   u_factor->get_row_ptrs());
        gko::kernels::reference::factorization::initialize_l_u(
            exec, factorize_sys_csr_individual[i].get(), l_factor.get(),
            u_factor.get());

        GKO_ASSERT_MTX_NEAR(l_factor, check_l_factors[i], 0.0);
        GKO_ASSERT_MTX_NEAR(u_factor, check_u_factors[i], 0.0);
    }
}


TYPED_TEST(BatchExactIlu, GenerationFromCoreIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;
    using unbatch_ilu_type = gko::factorization::Ilu<value_type>;
    using prec_type = typename TestFixture::prec_type;
    auto exec = this->exec;
    const auto sys_csr = this->mtx.get();
    const auto nbatch = this->nbatch;
    const auto nrows = this->nrows;
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    // unbatch for check
    const auto unbatch_size = gko::dim<2>(nrows, sys_csr->get_size().at(0)[1]);
    auto mtxs = gko::test::share(sys_csr->unbatch());
    auto ilu_fact = unbatch_ilu_type::build().with_skip_sorting(true).on(exec);
    std::vector<std::shared_ptr<const unbatch_type>> check_l_factors(
        mtxs.size());
    std::vector<std::shared_ptr<const unbatch_type>> check_u_factors(
        mtxs.size());
    for (size_t i = 0; i < mtxs.size(); i++) {
        auto facts = ilu_fact->generate(mtxs[i]);
        check_l_factors[i] = facts->get_l_factor();
        check_u_factors[i] = facts->get_u_factor();
    }

    auto prec_fact = prec_type::build().with_skip_sorting(true).on(exec);
    auto prec = prec_fact->generate(this->mtx);

    auto factorize_sys_csr_individual =
        prec->get_const_factorized_matrix()->unbatch();

    for (size_t i = 0; i < mtxs.size(); i++) {
        /*
        assert_factored_a_is_same_as_l_u(check_l_factors[i], check_u_factors[i],
                                         factorize_sys_csr_individual[i],
                                         this->diag_locs);
        */

        gko::array<index_type> l_factor_row_ptrs(exec, nrows + 1);
        gko::array<index_type> u_factor_row_ptrs(exec, nrows + 1);
        gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
            exec, factorize_sys_csr_individual[i].get(),
            l_factor_row_ptrs.get_data(), u_factor_row_ptrs.get_data());
        const auto l_factor_nnz = l_factor_row_ptrs.get_data()[nrows];
        const auto u_factor_nnz = u_factor_row_ptrs.get_data()[nrows];
        auto l_factor = unbatch_type::create(exec, unbatch_size, l_factor_nnz);
        auto u_factor = unbatch_type::create(exec, unbatch_size, u_factor_nnz);
        exec->copy(nrows + 1, l_factor_row_ptrs.get_const_data(),
                   l_factor->get_row_ptrs());
        exec->copy(nrows + 1, u_factor_row_ptrs.get_const_data(),
                   u_factor->get_row_ptrs());
        gko::kernels::reference::factorization::initialize_l_u(
            exec, factorize_sys_csr_individual[i].get(), l_factor.get(),
            u_factor.get());

        GKO_ASSERT_MTX_NEAR(l_factor, check_l_factors[i], 0.0);
        GKO_ASSERT_MTX_NEAR(u_factor, check_u_factors[i], 0.0);
    }
}


TYPED_TEST(BatchExactIlu, ApplyToSingleVectorIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    using prec_type = typename TestFixture::prec_type;
    using mtx_type = typename TestFixture::Mtx;
    using unbatch_type = typename mtx_type::unbatch_type;
    using unbatch_prec_type =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<value_type>,
                                 gko::solver::UpperTrs<value_type>>;

    auto exec = this->exec;
    auto b = gko::batch_initialize<BDense>({{-2.0, 9.0, 4.0}, {-3.0, 5.0, 3.0}},
                                           exec);
    auto x =
        BDense::create(exec, gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));
    auto umtxs = gko::test::share(this->mtx->unbatch());
    auto ub = b->unbatch();
    auto ux = x->unbatch();

    auto unbatch_prec_fact = unbatch_prec_type::build().on(exec);

    for (size_t i = 0; i < umtxs.size(); i++) {
        auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
        unbatch_prec->apply(ub[i].get(), ux[i].get());
    }

    auto prec_fact = prec_type::build().with_skip_sorting(true).on(exec);
    auto prec = prec_fact->generate(this->mtx);
    const auto factorized_sys_csr = prec->get_const_factorized_matrix();
    const auto diag_locs = prec->get_const_diag_locations();

    gko::kernels::reference::batch_exact_ilu::apply_exact_ilu(
        exec, factorized_sys_csr, diag_locs, b.get(), x.get());

    auto xs = x->unbatch();
    for (size_t i = 0; i < umtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
    }
}


}  // namespace
