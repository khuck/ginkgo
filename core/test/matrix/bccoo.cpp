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

#include <ginkgo/core/matrix/bccoo.hpp>


#include <gtest/gtest.h>


#include "core/base/unaligned_access.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Bccoo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Bccoo<value_type, index_type>;

    Bccoo()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Bccoo<value_type, index_type>::create(
              exec, gko::dim<2>{2, 3}, index_type{4}, index_type{10},
              index_type{4 * (sizeof(value_type) + sizeof(index_type) + 1)}))
    {
        mtx->read({{2, 3},
                   {{0, 0, 1.0},
                    {0, 1, 3.0},
                    {0, 2, 2.0},
                    {1, 0, 0.0},
                    {1, 1, 5.0},
                    {1, 2, 0.0}}});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;

    void assert_equal_to_original_mtx(const Mtx* m)
    {
        auto chunk_data = m->get_const_chunk();
        gko::size_type block_size = m->get_block_size();
        index_type ind = {};

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(chunk_data[ind], 0x00);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{1.0});
        ind += sizeof(value_type);

        EXPECT_EQ(chunk_data[ind], 0x01);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{3.0});
        ind += sizeof(value_type);

        if (block_size < 3) {
            EXPECT_EQ(chunk_data[ind], 0x02);
        } else {
            EXPECT_EQ(chunk_data[ind], 0x01);
        }
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{2.0});
        ind += sizeof(value_type);

        if ((block_size == 2) || (block_size >= 4)) {
            EXPECT_EQ(chunk_data[ind], 0xFF);
            ind++;
        }

        EXPECT_EQ(chunk_data[ind], 0x01);
        ind++;
        EXPECT_EQ(gko::get_value_chunk<value_type>(chunk_data, ind),
                  value_type{5.0});
        ind += sizeof(value_type);
    }

    void assert_empty(const Mtx* m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_block_size(), 0);
        ASSERT_EQ(m->get_num_blocks(), 0);
        ASSERT_EQ(m->get_num_bytes(), 0);
        ASSERT_EQ(m->get_const_rows(), nullptr);
        ASSERT_EQ(m->get_const_offsets(), nullptr);
        ASSERT_EQ(m->get_const_chunk(), nullptr);
    }
};

TYPED_TEST_SUITE(Bccoo, gko::test::ValueIndexTypes);


TYPED_TEST(Bccoo, KnowsItsSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(this->mtx->get_num_stored_elements(), 4);
}


TYPED_TEST(Bccoo, ContainsCorrectData)
{
    this->assert_equal_to_original_mtx(this->mtx.get());
}


TYPED_TEST(Bccoo, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = Mtx::create(this->exec);

    this->assert_empty(mtx.get());
}


TYPED_TEST(Bccoo, CanBeCreatedFromExistingData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    const index_type block_size = 10;
    const index_type num_bytes = 6 + 4 * sizeof(value_type);
    index_type ind = {};
    gko::uint8 chunk[num_bytes] = {};
    index_type offsets[] = {0, num_bytes};
    index_type rows[] = {0};

    chunk[ind++] = 0x00;
    gko::set_value_chunk<value_type>(chunk, ind, 1.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0x01;
    gko::set_value_chunk<value_type>(chunk, ind, 2.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0xFF;
    chunk[ind++] = 0x01;
    gko::set_value_chunk<value_type>(chunk, ind, 3.0);
    ind += sizeof(value_type);
    chunk[ind++] = 0xFF;
    chunk[ind++] = 0x00;
    gko::set_value_chunk<value_type>(chunk, ind, 4.0);
    ind += sizeof(value_type);

    auto mtx = gko::matrix::Bccoo<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 2},
        gko::Array<gko::uint8>::view(this->exec, num_bytes, chunk),
        gko::Array<index_type>::view(this->exec, 2, offsets),
        gko::Array<index_type>::view(this->exec, 1, rows), 4, block_size);

    ASSERT_EQ(mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(mtx->get_block_size(), block_size);
    ASSERT_EQ(mtx->get_const_offsets(), offsets);
    ASSERT_EQ(mtx->get_const_rows(), rows);
}


TYPED_TEST(Bccoo, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(this->mtx.get());

    this->assert_equal_to_original_mtx(this->mtx.get());
    *((value_type*)(this->mtx->get_chunk() + (2 + sizeof(value_type)))) = 5.0;
    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Bccoo, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = Mtx::create(this->exec);

    copy->copy_from(std::move(this->mtx));

    this->assert_equal_to_original_mtx(copy.get());
}


TYPED_TEST(Bccoo, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto clone = this->mtx->clone();

    this->assert_equal_to_original_mtx(this->mtx.get());
    *((value_type*)(this->mtx->get_chunk() + (2 + sizeof(value_type)))) = 5.0;
    this->assert_equal_to_original_mtx(dynamic_cast<Mtx*>(clone.get()));
}


TYPED_TEST(Bccoo, CanBeCleared)
{
    this->mtx->clear();

    this->assert_empty(this->mtx.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    auto m = Mtx::create(this->exec);
    m->read({{2, 3},
             {{0, 0, 1.0},
              {0, 1, 3.0},
              {0, 2, 2.0},
              {1, 0, 0.0},
              {1, 1, 5.0},
              {1, 2, 0.0}}});

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Bccoo, CanBeReadFromMatrixAssemblyData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto m = Mtx::create(this->exec);
    gko::matrix_assembly_data<value_type, index_type> data(gko::dim<2>{2, 3});
    data.set_value(0, 0, 1.0);
    data.set_value(0, 1, 3.0);
    data.set_value(0, 2, 2.0);
    data.set_value(1, 0, 0.0);
    data.set_value(1, 1, 5.0);
    data.set_value(1, 2, 0.0);

    m->read(data);

    this->assert_equal_to_original_mtx(m.get());
}


TYPED_TEST(Bccoo, GeneratesCorrectMatrixData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    gko::matrix_data<value_type, index_type> data;

    this->mtx->write(data);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    ASSERT_EQ(data.nonzeros.size(), 4);
    EXPECT_EQ(data.nonzeros[0], tpl(0, 0, value_type{1.0}));
    EXPECT_EQ(data.nonzeros[1], tpl(0, 1, value_type{3.0}));
    EXPECT_EQ(data.nonzeros[2], tpl(0, 2, value_type{2.0}));
    EXPECT_EQ(data.nonzeros[3], tpl(1, 1, value_type{5.0}));
}


}  // namespace
