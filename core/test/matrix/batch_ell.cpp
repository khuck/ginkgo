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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class Ell : public ::testing::Test {
protected:
    using value_type = T;
    using EllMtx = gko::matrix::Ell<value_type>;
    using size_type = gko::size_type;
    Ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch::initialize<gko::batch::matrix::Ell<value_type>>(
              {{{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec)),
          mvec(gko::batch::initialize<gko::batch::MultiVector<value_type>>(
              {{{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
               {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}},
              exec)),
          ell_mtx(gko::initialize<gko::matrix::Ell<value_type>>(
              {{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, exec))
    {}


    static void assert_equal_to_original_mtx(
        gko::batch::matrix::Ell<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 2);
        ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 2 * (2 * 3));
        EXPECT_EQ(m->at(0, 0, 0), value_type{-1.0});
        EXPECT_EQ(m->at(0, 0, 1), value_type{2.0});
        EXPECT_EQ(m->at(0, 0, 2), value_type{3.0});
        EXPECT_EQ(m->at(0, 1, 0), value_type{-1.5});
        EXPECT_EQ(m->at(0, 1, 1), value_type{2.5});
        ASSERT_EQ(m->at(0, 1, 2), value_type{3.5});
        EXPECT_EQ(m->at(1, 0, 0), value_type{1.0});
        EXPECT_EQ(m->at(1, 0, 1), value_type{2.5});
        EXPECT_EQ(m->at(1, 0, 2), value_type{3.0});
        EXPECT_EQ(m->at(1, 1, 0), value_type{1.0});
        EXPECT_EQ(m->at(1, 1, 1), value_type{2.0});
        ASSERT_EQ(m->at(1, 1, 2), value_type{3.0});
    }

    static void assert_empty(gko::batch::matrix::Ell<value_type>* m)
    {
        ASSERT_EQ(m->get_num_batch_items(), 0);
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::batch::matrix::Ell<value_type>> mtx;
    std::unique_ptr<gko::batch::MultiVector<value_type>> mvec;
    std::unique_ptr<gko::matrix::Ell<value_type>> ell_mtx;
};

TYPED_TEST_SUITE(Ell, gko::test::ValueTypes);


TYPED_TEST(Ell, KnowsItsSizeAndValues)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Ell, CanBeEmpty)
{
    auto empty = gko::batch::matrix::Ell<TypeParam>::create(this->exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(Ell, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty = gko::batch::matrix::Ell<TypeParam>::create(this->exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(Ell, CanGetValuesForEntry)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_EQ(this->mtx->get_values_for_item(1)[0], value_type{1.0});
}


TYPED_TEST(Ell, CanCreateEllItemView)
{
    GKO_ASSERT_MTX_NEAR(this->mtx->create_view_for_item(1), this->ell_mtx, 0.0);
}


TYPED_TEST(Ell, CanCreateMultiVectorView)
{
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx->create_multi_vector_view(), this->mvec,
                              0.0);
}


TYPED_TEST(Ell, CanBeCopied)
{
    auto mtx_copy = gko::batch::matrix::Ell<TypeParam>::create(this->exec);

    mtx_copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    this->mtx->at(0, 0, 0) = 7;
    this->mtx->at(0, 1) = 7;
    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Ell, CanBeMoved)
{
    auto mtx_copy = gko::batch::matrix::Ell<TypeParam>::create(this->exec);

    this->mtx->move_to(mtx_copy);

    this->assert_equal_to_original_mtx(mtx_copy.get());
}


TYPED_TEST(Ell, CanBeCloned)
{
    auto mtx_clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(
        dynamic_cast<decltype(this->mtx.get())>(mtx_clone.get()));
}


TYPED_TEST(Ell, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Ell, CanBeConstructedWithSize)
{
    using size_type = gko::size_type;

    auto m = gko::batch::matrix::Ell<TypeParam>::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>{5, 3}));

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(5, 3));
    ASSERT_EQ(m->get_num_stored_elements(), 30);
}


TYPED_TEST(Ell, CanBeConstructedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    // clang-format off
    value_type data[] = {
       1.0,  2.0,
      -1.0,  3.0,
       4.0, -1.0,
       3.0,  5.0,
       1.0,  5.0,
       6.0, -3.0};
    // clang-format on

    auto m = gko::batch::matrix::Ell<TypeParam>::create(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 2)),
        gko::array<value_type>::view(this->exec, 8, data));

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(0, 0, 0), value_type{1.0});
    ASSERT_EQ(m->at(0, 0, 1), value_type{2.0});
    ASSERT_EQ(m->at(0, 1, 0), value_type{-1.0});
    ASSERT_EQ(m->at(0, 1, 1), value_type{3.0});
    ASSERT_EQ(m->at(1, 0, 0), value_type{4.0});
    ASSERT_EQ(m->at(1, 0, 1), value_type{-1.0});
    ASSERT_EQ(m->at(1, 1, 0), value_type{3.0});
    ASSERT_EQ(m->at(1, 1, 1), value_type{5.0});
}


TYPED_TEST(Ell, CanBeConstructedFromExistingConstData)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    // clang-format off
    const value_type data[] = {
       1.0,  2.0,
      -1.0,  3.0,
       4.0, -1.0,
       3.0,  5.0,
       1.0,  5.0,
       6.0, -3.0};
    // clang-format on

    auto m = gko::batch::matrix::Ell<TypeParam>::create_const(
        this->exec, gko::batch_dim<2>(2, gko::dim<2>(2, 2)),
        gko::array<value_type>::const_view(this->exec, 8, data));

    ASSERT_EQ(m->get_const_values(), data);
    ASSERT_EQ(m->at(0, 0, 0), value_type{1.0});
    ASSERT_EQ(m->at(0, 0, 1), value_type{2.0});
    ASSERT_EQ(m->at(0, 1, 0), value_type{-1.0});
    ASSERT_EQ(m->at(0, 1, 1), value_type{3.0});
    ASSERT_EQ(m->at(1, 0, 0), value_type{4.0});
    ASSERT_EQ(m->at(1, 0, 1), value_type{-1.0});
    ASSERT_EQ(m->at(1, 1, 0), value_type{3.0});
    ASSERT_EQ(m->at(1, 1, 1), value_type{5.0});
}


TYPED_TEST(Ell, CanBeConstructedFromEllMatrices)
{
    using value_type = typename TestFixture::value_type;
    using EllMtx = typename TestFixture::EllMtx;
    using size_type = gko::size_type;

    auto mat1 = gko::initialize<EllMtx>({{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, this->exec);

    auto m = gko::batch::create_from_item<gko::batch::matrix::Ell<value_type>>(
        this->exec, std::vector<EllMtx*>{mat1.get(), mat2.get()});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Ell, CanBeConstructedFromEllMatricesByDuplication)
{
    using value_type = typename TestFixture::value_type;
    using EllMtx = typename TestFixture::EllMtx;
    using size_type = gko::size_type;

    auto mat1 = gko::initialize<EllMtx>(4, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, this->exec);

    auto bat_m =
        gko::batch::create_from_item<gko::batch::matrix::Ell<value_type>>(
            this->exec,
            std::vector<EllMtx*>{mat1.get(), mat1.get(), mat1.get()});
    auto m = gko::batch::create_from_item<gko::batch::matrix::Ell<value_type>>(
        this->exec, 3, mat1.get());

    GKO_ASSERT_BATCH_MTX_NEAR(bat_m.get(), m.get(), 1e-14);
}


TYPED_TEST(Ell, CanBeConstructedByDuplicatingEllMatrices)
{
    using value_type = typename TestFixture::value_type;
    using EllMtx = typename TestFixture::EllMtx;
    using size_type = gko::size_type;

    auto mat1 = gko::initialize<EllMtx>({{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, this->exec);

    auto m = gko::batch::create_from_item<gko::batch::matrix::Ell<value_type>>(
        this->exec, std::vector<EllMtx*>{mat1.get(), mat2.get()});
    auto m_ref =
        gko::batch::create_from_item<gko::batch::matrix::Ell<value_type>>(
            this->exec,
            std::vector<EllMtx*>{mat1.get(), mat2.get(), mat1.get(), mat2.get(),
                                 mat1.get(), mat2.get()});

    auto m2 = gko::batch::duplicate<gko::batch::matrix::Ell<value_type>>(
        this->exec, 3, m.get());

    GKO_ASSERT_BATCH_MTX_NEAR(m2.get(), m_ref.get(), 1e-14);
}


TYPED_TEST(Ell, CanBeUnbatchedIntoEllMatrices)
{
    using value_type = typename TestFixture::value_type;
    using EllMtx = typename TestFixture::EllMtx;
    using size_type = gko::size_type;
    auto mat1 = gko::initialize<EllMtx>(4, {{-1.0, 2.0, 3.0}, {-1.5, 2.5, 3.5}},
                                        this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{1.0, 2.5, 3.0}, {1.0, 2.0, 3.0}}, this->exec);

    auto ell_mats = gko::batch::unbatch<gko::batch::matrix::Ell<value_type>>(
        this->mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(ell_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(Ell, CanBeListConstructed)
{
    using value_type = typename TestFixture::value_type;
    auto m = gko::batch::initialize<gko::batch::matrix::Ell<TypeParam>>(
        {{1.0, 2.0}, {1.0, 3.0}}, this->exec);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    EXPECT_EQ(m->at(0, 0), value_type{1});
    EXPECT_EQ(m->at(0, 1), value_type{2});
    EXPECT_EQ(m->at(1, 0), value_type{1});
    EXPECT_EQ(m->at(1, 1), value_type{3});
}


TYPED_TEST(Ell, CanBeListConstructedByCopies)
{
    using value_type = typename TestFixture::value_type;

    auto m = gko::batch::initialize<gko::batch::matrix::Ell<TypeParam>>(
        2, I<value_type>({1.0, 2.0}), this->exec);

    ASSERT_EQ(m->get_num_batch_items(), 2);
    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 1));
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 0, 1), value_type{2.0});
}


TYPED_TEST(Ell, CanBeDoubleListConstructed)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;

    auto m = gko::batch::initialize<gko::batch::matrix::Ell<TypeParam>>(
        {{I<T>{1.0, 1.0, 0.0}, I<T>{2.0, 4.0, 3.0}, I<T>{3.0, 6.0, 1.0}},
         {I<T>{1.0, 2.0, -1.0}, I<T>{3.0, 4.0, -2.0}, I<T>{5.0, 6.0, -3.0}}},
        this->exec);

    ASSERT_EQ(m->get_common_size(), gko::dim<2>(3, 3));
    EXPECT_EQ(m->at(0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 1), value_type{1.0});
    EXPECT_EQ(m->at(0, 2), value_type{0.0});
    ASSERT_EQ(m->at(0, 3), value_type{2.0});
    EXPECT_EQ(m->at(0, 4), value_type{4.0});
    EXPECT_EQ(m->at(1, 0), value_type{1.0});
    EXPECT_EQ(m->at(1, 1), value_type{2.0});
    EXPECT_EQ(m->at(1, 2), value_type{-1.0});
    ASSERT_EQ(m->at(1, 3), value_type{3.0});
    EXPECT_EQ(m->at(1, 4), value_type{4.0});
}


TYPED_TEST(Ell, CanBeReadFromMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = int;

    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 2}, {{0, 0, 1.0}, {0, 1, 3.0}, {1, 0, 0.0}, {1, 1, 5.0}}));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 2}, {{0, 0, -1.0}, {0, 1, 0.5}, {1, 0, 0.0}, {1, 1, 9.0}}));

    auto m = gko::batch::read<value_type, index_type,
                              gko::batch::matrix::Ell<value_type>>(this->exec,
                                                                   vec_data);

    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 2));
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at(0, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 1, 1), value_type{5.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{-1.0});
    EXPECT_EQ(m->at(1, 0, 1), value_type{0.5});
    EXPECT_EQ(m->at(1, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(1, 1, 1), value_type{9.0});
}


TYPED_TEST(Ell, CanBeReadFromSparseMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = int;
    auto vec_data = std::vector<gko::matrix_data<value_type, index_type>>{};
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 2}, {{0, 0, 1.0}, {0, 1, 3.0}, {1, 1, 5.0}}));
    vec_data.emplace_back(gko::matrix_data<value_type, index_type>(
        {2, 2}, {{0, 0, -1.0}, {0, 1, 0.5}, {1, 1, 9.0}}));

    auto m = gko::batch::read<value_type, index_type,
                              gko::batch::matrix::Ell<value_type>>(this->exec,
                                                                   vec_data);

    ASSERT_EQ(m->get_common_size(), gko::dim<2>(2, 2));
    EXPECT_EQ(m->at(0, 0, 0), value_type{1.0});
    EXPECT_EQ(m->at(0, 0, 1), value_type{3.0});
    EXPECT_EQ(m->at(0, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(0, 1, 1), value_type{5.0});
    EXPECT_EQ(m->at(1, 0, 0), value_type{-1.0});
    EXPECT_EQ(m->at(1, 0, 1), value_type{0.5});
    EXPECT_EQ(m->at(1, 1, 0), value_type{0.0});
    EXPECT_EQ(m->at(1, 1, 1), value_type{9.0});
}


TYPED_TEST(Ell, GeneratesCorrectMatrixData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = int;
    using tpl = typename gko::matrix_data<TypeParam>::nonzero_type;

    auto data =
        gko::batch::write<value_type, index_type,
                          gko::batch::matrix::Ell<value_type>>(this->mtx.get());

    ASSERT_EQ(data[0].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[0].nonzeros.size(), 6);
    EXPECT_EQ(data[0].nonzeros[0], tpl(0, 0, value_type{-1.0}));
    EXPECT_EQ(data[0].nonzeros[1], tpl(0, 1, value_type{2.0}));
    EXPECT_EQ(data[0].nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data[0].nonzeros[3], tpl(1, 0, value_type{-1.5}));
    EXPECT_EQ(data[0].nonzeros[4], tpl(1, 1, value_type{2.5}));
    EXPECT_EQ(data[0].nonzeros[5], tpl(1, 2, value_type{3.5}));
    ASSERT_EQ(data[1].size, gko::dim<2>(2, 3));
    ASSERT_EQ(data[1].nonzeros.size(), 6);
    EXPECT_EQ(data[1].nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[1], tpl(0, 1, value_type{2.5}));
    EXPECT_EQ(data[1].nonzeros[2], tpl(0, 2, value_type{3.0}));
    EXPECT_EQ(data[1].nonzeros[3], tpl(1, 0, value_type{1.0}));
    EXPECT_EQ(data[1].nonzeros[4], tpl(1, 1, value_type{2.0}));
    EXPECT_EQ(data[1].nonzeros[5], tpl(1, 2, value_type{3.0}));
}
