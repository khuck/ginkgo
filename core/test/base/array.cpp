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

#include <ginkgo/core/base/array.hpp>


#include <algorithm>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Array : public ::testing::Test {
protected:
    Array() : exec(gko::ReferenceExecutor::create()), x(exec, 2)
    {
        x.data()[0] = 5;
        x.data()[1] = 2;
    }

    static void assert_equal_to_original_x(gko::array<T>& a,
                                           bool check_zero = true)
    {
        ASSERT_EQ(a.size(), 2);
        if (check_zero) EXPECT_EQ(a.data()[0], T{5});
        EXPECT_EQ(a.data()[1], T{2});
        if (check_zero) EXPECT_EQ(a.const_data()[0], T{5});
        EXPECT_EQ(a.const_data()[1], T{2});
    }

    std::shared_ptr<const gko::Executor> exec;
    gko::array<T> x;
};

TYPED_TEST_SUITE(Array, gko::test::ValueAndIndexTypes, TypenameNameGenerator);


TYPED_TEST(Array, CanBeCreatedWithoutAnExecutor)
{
    gko::array<TypeParam> a;

    ASSERT_EQ(a.get_executor(), nullptr);
    ASSERT_EQ(a.size(), 0);
}


TYPED_TEST(Array, CanBeEmpty)
{
    gko::array<TypeParam> a(this->exec);

    ASSERT_EQ(a.size(), 0);
}


TYPED_TEST(Array, ReturnsNullWhenEmpty)
{
    gko::array<TypeParam> a(this->exec);

    EXPECT_EQ(a.const_data(), nullptr);
    ASSERT_EQ(a.data(), nullptr);
}


TYPED_TEST(Array, CanBeCreatedFromExistingData)
{
    gko::array<TypeParam> a{this->exec, 3, new TypeParam[3],
                            std::default_delete<TypeParam[]>{}};

    EXPECT_EQ(a.size(), 3);
}


TYPED_TEST(Array, CanBeCreatedFromDataOnExecutor)
{
    gko::array<TypeParam> a{this->exec, 3,
                            this->exec->template alloc<TypeParam>(3)};

    EXPECT_EQ(a.size(), 3);
}


TYPED_TEST(Array, CanBeCreatedFromRange)
{
    using std::begin;
    auto data = {1, 2, 3};

    gko::array<TypeParam> a{this->exec, begin(data), end(data)};

    EXPECT_EQ(a.const_data()[0], TypeParam{1});
    EXPECT_EQ(a.const_data()[1], TypeParam{2});
    ASSERT_EQ(a.const_data()[2], TypeParam{3});
}


TYPED_TEST(Array, CanBeCreatedFromInitializerList)
{
    gko::array<TypeParam> a{this->exec, {1, 2, 3}};

    EXPECT_EQ(a.const_data()[0], TypeParam{1});
    EXPECT_EQ(a.const_data()[1], TypeParam{2});
    ASSERT_EQ(a.const_data()[2], TypeParam{3});
}


TYPED_TEST(Array, KnowsItsSize) { ASSERT_EQ(this->x.size(), 2); }


TYPED_TEST(Array, ReturnsValidDataPtr)
{
    EXPECT_EQ(this->x.data()[0], TypeParam{5});
    EXPECT_EQ(this->x.data()[1], TypeParam{2});
}


TYPED_TEST(Array, ReturnsValidConstDataPtr)
{
    EXPECT_EQ(this->x.const_data()[0], TypeParam{5});
    EXPECT_EQ(this->x.const_data()[1], TypeParam{2});
}


TYPED_TEST(Array, KnowsItsExecutor)
{
    ASSERT_EQ(this->x.get_executor(), this->exec);
}


TYPED_TEST(Array, CanBeCopyConstructed)
{
    gko::array<TypeParam> a(this->x);
    this->x.data()[0] = 7;

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeMoveConstructed)
{
    gko::array<TypeParam> a(std::move(this->x));

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeCopyConstructedToADifferentExecutor)
{
    gko::array<TypeParam> a{this->exec, this->x};

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeMoveConstructedToADifferentExecutor)
{
    gko::array<TypeParam> a{this->exec, std::move(this->x)};

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, MoveConstructedFromArrayExecutorlessIsEmpty)
{
    gko::array<TypeParam> a{std::move(this->x)};

    a = std::move(this->x);

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, MoveConstructedFromArraySameExecutorIsEmpty)
{
    gko::array<TypeParam> a{this->exec, std::move(this->x)};

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, MoveConstructedFromArrayDifferentExecutorIsEmpty)
{
    gko::array<TypeParam> a{gko::ReferenceExecutor::create(),
                            std::move(this->x)};

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, CanBeCopied)
{
    auto omp = gko::OmpExecutor::create();
    gko::array<TypeParam> a(omp, 3);

    a = this->x;
    this->x.data()[0] = 7;

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeCopiedToExecutorlessArray)
{
    gko::array<TypeParam> a;

    a = this->x;

    ASSERT_EQ(a.get_executor(), this->x.get_executor());
    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeCopiedFromExecutorlessArray)
{
    gko::array<TypeParam> a;

    this->x = a;

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, CanBeMoved)
{
    auto omp = gko::OmpExecutor::create();
    gko::array<TypeParam> a(omp, 3);

    a = std::move(this->x);

    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeMovedToExecutorlessArray)
{
    gko::array<TypeParam> a;

    a = std::move(this->x);

    ASSERT_EQ(a.get_executor(), this->exec);
    this->assert_equal_to_original_x(a);
}


TYPED_TEST(Array, CanBeMovedFromExecutorlessArray)
{
    gko::array<TypeParam> a;

    this->x = std::move(a);

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, MovedFromArrayExecutorlessIsEmpty)
{
    gko::array<TypeParam> a;

    a = std::move(this->x);

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, MovedFromArraySameExecutorIsEmpty)
{
    gko::array<TypeParam> a{this->exec};

    a = std::move(this->x);

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, MovedFromArrayDifferentExecutorIsEmpty)
{
    gko::array<TypeParam> a{gko::ReferenceExecutor::create()};

    a = std::move(this->x);

    ASSERT_EQ(this->x.get_executor(), this->exec);
    ASSERT_EQ(this->x.size(), 0);
}


TYPED_TEST(Array, CanCreateTemporaryCloneOnSameExecutor)
{
    auto tmp_clone = make_temporary_clone(this->exec, &this->x);

    ASSERT_EQ(tmp_clone.get(), &this->x);
}


TYPED_TEST(Array, CanCreateTemporaryOutputCloneOnSameExecutor)
{
    auto tmp_clone = make_temporary_output_clone(this->exec, &this->x);

    ASSERT_EQ(tmp_clone.get(), &this->x);
}


// For tests between different memory, check cuda/test/base/array.cu
TYPED_TEST(Array, DoesNotCreateATemporaryCloneBetweenSameMemory)
{
    auto other = gko::ReferenceExecutor::create();

    auto tmp_clone = make_temporary_clone(other, &this->x);

    this->assert_equal_to_original_x(*tmp_clone.get());
    ASSERT_EQ(tmp_clone.get(), &this->x);
}


TYPED_TEST(Array, DoesNotCopyBackTemporaryCloneBetweenSameMemory)
{
    auto other = gko::ReferenceExecutor::create();

    {
        auto tmp_clone = make_temporary_clone(other, &this->x);
        // change x, and check that there is no copy-back to overwrite it again
        this->x.data()[0] = 0;
    }

    this->assert_equal_to_original_x(this->x, false);
    EXPECT_EQ(this->x.data()[0], TypeParam{0});
}


TYPED_TEST(Array, CanCreateTemporaryOutputCloneOnDifferentExecutors)
{
    auto other = gko::OmpExecutor::create();

    {
        auto tmp_clone = make_temporary_output_clone(other, &this->x);
        tmp_clone->data()[0] = 4;
        tmp_clone->data()[1] = 5;

        // there is no reliable way to check the memory is uninitialized
        ASSERT_EQ(tmp_clone->size(), this->x.size());
        ASSERT_EQ(tmp_clone->get_executor(), other);
        ASSERT_EQ(this->x.get_executor(), this->exec);
        ASSERT_EQ(this->x.data()[0], TypeParam{5});
        ASSERT_EQ(this->x.data()[1], TypeParam{2});
    }
    ASSERT_EQ(this->x.data()[0], TypeParam{4});
    ASSERT_EQ(this->x.data()[1], TypeParam{5});
}


TYPED_TEST(Array, CanBeCleared)
{
    this->x.clear();

    ASSERT_EQ(this->x.size(), 0);
    ASSERT_EQ(this->x.data(), nullptr);
    ASSERT_EQ(this->x.const_data(), nullptr);
}


TYPED_TEST(Array, CanBeResized)
{
    this->x.resize_and_reset(3);

    this->x.data()[0] = 1;
    this->x.data()[1] = 8;
    this->x.data()[2] = 7;

    EXPECT_EQ(this->x.const_data()[0], TypeParam{1});
    EXPECT_EQ(this->x.const_data()[1], TypeParam{8});
    EXPECT_EQ(this->x.const_data()[2], TypeParam{7});
}


TYPED_TEST(Array, ViewCannotBeResized)
{
    TypeParam data[] = {1, 2, 3};
    auto view = gko::make_array_view(this->exec, 3, data);

    EXPECT_THROW(view.resize_and_reset(1), gko::NotSupported);
    EXPECT_EQ(view.size(), 3);
    ASSERT_EQ(view.data()[0], TypeParam{1});
}


template <typename T>
class my_null_deleter {
public:
    using pointer = T*;

    void operator()(pointer) const noexcept {}
};

template <typename T>
class my_null_deleter<T[]> {
public:
    using pointer = T[];

    void operator()(pointer) const noexcept {}
};


TYPED_TEST(Array, CustomDeleterCannotBeResized)
{
    TypeParam data[] = {1, 2, 3};
    auto view_custom_deleter = gko::array<TypeParam>(
        this->exec, 3, data, my_null_deleter<TypeParam[]>{});

    EXPECT_THROW(view_custom_deleter.resize_and_reset(1), gko::NotSupported);
    EXPECT_EQ(view_custom_deleter.size(), 3);
    ASSERT_EQ(view_custom_deleter.data()[0], TypeParam{1});
}


TYPED_TEST(Array, CanBeAssignedAnExecutor)
{
    gko::array<TypeParam> a;

    a.set_executor(this->exec);

    ASSERT_EQ(a.get_executor(), this->exec);
}


TYPED_TEST(Array, ChangesExecutors)
{
    auto omp = gko::OmpExecutor::create();
    this->x.set_executor(omp);

    ASSERT_EQ(this->x.get_executor(), omp);
    this->assert_equal_to_original_x(this->x);
}


TYPED_TEST(Array, ViewModifiesOriginalData)
{
    TypeParam data[] = {1, 2, 3};
    auto view = gko::make_array_view(this->exec, 3, data);

    TypeParam new_data[] = {5, 4, 2};
    std::copy(new_data, new_data + 3, view.data());

    EXPECT_EQ(data[0], TypeParam{5});
    EXPECT_EQ(data[1], TypeParam{4});
    EXPECT_EQ(data[2], TypeParam{2});
    ASSERT_EQ(view.size(), 3);
}


TYPED_TEST(Array, CopyArrayToArray)
{
    gko::array<TypeParam> array(this->exec, {1, 2, 3});
    gko::array<TypeParam> array2(this->exec, {5, 4, 2, 1});

    array = array2;

    EXPECT_EQ(array.data()[0], TypeParam{5});
    EXPECT_EQ(array.data()[1], TypeParam{4});
    EXPECT_EQ(array.data()[2], TypeParam{2});
    EXPECT_EQ(array.data()[3], TypeParam{1});
    EXPECT_EQ(array.size(), 4);
    EXPECT_NE(array.data(), array2.data());
    ASSERT_EQ(array2.size(), 4);
}


TYPED_TEST(Array, CopyViewToView)
{
    TypeParam data[] = {1, 2, 3};
    auto view = gko::make_array_view(this->exec, 3, data);
    TypeParam data2[] = {5, 4, 2};
    auto view2 = gko::make_array_view(this->exec, 3, data2);
    TypeParam data_size4[] = {5, 4, 2, 1};
    auto view_size4 = gko::make_array_view(this->exec, 4, data_size4);

    view = view2;
    view2.data()[0] = 2;

    EXPECT_EQ(data[0], TypeParam{5});
    EXPECT_EQ(data[1], TypeParam{4});
    EXPECT_EQ(data[2], TypeParam{2});
    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view2.size(), 3);
    EXPECT_EQ(view2.data()[0], TypeParam{2});
    ASSERT_THROW(view2 = view_size4, gko::OutOfBoundsError);
}


TYPED_TEST(Array, CopyViewToArray)
{
    TypeParam data[] = {1, 2, 3, 4};
    auto view = gko::make_array_view(this->exec, 4, data);
    gko::array<TypeParam> array(this->exec, {5, 4, 2});

    array = view;
    view.data()[0] = 2;

    EXPECT_EQ(array.data()[0], TypeParam{1});
    EXPECT_EQ(array.data()[1], TypeParam{2});
    EXPECT_EQ(array.data()[2], TypeParam{3});
    EXPECT_EQ(array.data()[3], TypeParam{4});
    EXPECT_EQ(array.size(), 4);
    ASSERT_EQ(view.size(), 4);
}


TYPED_TEST(Array, CopyArrayToView)
{
    TypeParam data[] = {1, 2, 3};
    auto view = gko::make_array_view(this->exec, 3, data);
    gko::array<TypeParam> array_size2(this->exec, {5, 4});
    gko::array<TypeParam> array_size4(this->exec, {5, 4, 2, 1});

    view = array_size2;

    EXPECT_EQ(data[0], TypeParam{5});
    EXPECT_EQ(data[1], TypeParam{4});
    EXPECT_EQ(data[2], TypeParam{3});
    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(array_size2.size(), 2);
    ASSERT_THROW(view = array_size4, gko::OutOfBoundsError);
}


TYPED_TEST(Array, MoveArrayToArray)
{
    gko::array<TypeParam> array(this->exec, {1, 2, 3});
    gko::array<TypeParam> array2(this->exec, {5, 4, 2, 1});
    auto data2 = array2.data();

    array = std::move(array2);

    EXPECT_EQ(array.data(), data2);
    EXPECT_EQ(array.data()[0], TypeParam{5});
    EXPECT_EQ(array.data()[1], TypeParam{4});
    EXPECT_EQ(array.data()[2], TypeParam{2});
    EXPECT_EQ(array.data()[3], TypeParam{1});
    EXPECT_EQ(array.size(), 4);
    EXPECT_EQ(array2.data(), nullptr);
    ASSERT_EQ(array2.size(), 0);
}


TYPED_TEST(Array, MoveViewToView)
{
    TypeParam data[] = {1, 2, 3, 4};
    auto view = gko::make_array_view(this->exec, 4, data);
    TypeParam data2[] = {5, 4, 2};
    auto view2 = gko::make_array_view(this->exec, 3, data2);

    view = std::move(view2);

    EXPECT_EQ(view.data(), data2);
    EXPECT_EQ(view.data()[0], TypeParam{5});
    EXPECT_EQ(view.data()[1], TypeParam{4});
    EXPECT_EQ(view.data()[2], TypeParam{2});
    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view2.data(), nullptr);
    EXPECT_EQ(view2.size(), 0);
    EXPECT_NE(data, nullptr);
    EXPECT_EQ(data[0], TypeParam{1});
    EXPECT_EQ(data[1], TypeParam{2});
    EXPECT_EQ(data[2], TypeParam{3});
    ASSERT_EQ(data[3], TypeParam{4});
}


TYPED_TEST(Array, MoveViewToArray)
{
    TypeParam data[] = {1, 2, 3, 4};
    gko::array<TypeParam> array(this->exec, {5, 4, 2});
    auto view = gko::make_array_view(this->exec, 4, data);

    array = std::move(view);

    EXPECT_EQ(array.data(), data);
    EXPECT_EQ(array.data()[0], TypeParam{1});
    EXPECT_EQ(array.data()[1], TypeParam{2});
    EXPECT_EQ(array.data()[2], TypeParam{3});
    EXPECT_EQ(array.data()[3], TypeParam{4});
    EXPECT_EQ(array.size(), 4);
    EXPECT_EQ(data[0], TypeParam{1});
    EXPECT_EQ(data[1], TypeParam{2});
    EXPECT_EQ(data[2], TypeParam{3});
    EXPECT_EQ(data[3], TypeParam{4});
    EXPECT_EQ(view.data(), nullptr);
    ASSERT_EQ(view.size(), 0);
}


TYPED_TEST(Array, MoveArrayToView)
{
    TypeParam data[] = {1, 2, 3};
    auto view = gko::make_array_view(this->exec, 3, data);
    gko::array<TypeParam> array_size2(this->exec, {5, 4});
    gko::array<TypeParam> array_size4(this->exec, {5, 4, 2, 1});
    auto size2_ptr = array_size2.data();
    auto size4_ptr = array_size4.data();

    view = std::move(array_size2);

    EXPECT_EQ(view.data()[0], TypeParam{5});
    EXPECT_EQ(view.data()[1], TypeParam{4});
    EXPECT_EQ(view.size(), 2);
    EXPECT_NE(view.data(), data);
    EXPECT_EQ(view.data(), size2_ptr);
    EXPECT_NO_THROW(view = std::move(array_size4));
    EXPECT_EQ(view.data(), size4_ptr);
    EXPECT_EQ(array_size2.data(), nullptr);
    ASSERT_EQ(array_size2.size(), 0);
}


TYPED_TEST(Array, AsView)
{
    auto ptr = this->x.data();
    auto size = this->x.size();
    auto exec = this->x.get_executor();
    auto view = this->x.as_view();

    ASSERT_EQ(ptr, this->x.data());
    ASSERT_EQ(ptr, view.data());
    ASSERT_EQ(size, this->x.size());
    ASSERT_EQ(size, view.size());
    ASSERT_EQ(exec, this->x.get_executor());
    ASSERT_EQ(exec, view.get_executor());
    ASSERT_TRUE(this->x.is_owning());
    ASSERT_FALSE(view.is_owning());
}


TYPED_TEST(Array, AsConstView)
{
    auto ptr = this->x.data();
    auto size = this->x.size();
    auto exec = this->x.get_executor();
    auto view = this->x.as_const_view();

    ASSERT_EQ(ptr, this->x.data());
    ASSERT_EQ(ptr, view.const_data());
    ASSERT_EQ(size, this->x.size());
    ASSERT_EQ(size, view.size());
    ASSERT_EQ(exec, this->x.get_executor());
    ASSERT_EQ(exec, view.get_executor());
    ASSERT_TRUE(this->x.is_owning());
    ASSERT_FALSE(view.is_owning());
}


TYPED_TEST(Array, ArrayConstCastWorksOnView)
{
    auto ptr = this->x.data();
    auto size = this->x.size();
    auto exec = this->x.get_executor();
    auto const_view = this->x.as_const_view();
    auto view = gko::detail::array_const_cast(std::move(const_view));
    static_assert(std::is_same<decltype(view), decltype(this->x)>::value,
                  "wrong return type");

    ASSERT_EQ(nullptr, const_view.const_data());
    ASSERT_EQ(0, const_view.size());
    ASSERT_EQ(exec, const_view.get_executor());
    ASSERT_EQ(ptr, this->x.data());
    ASSERT_EQ(ptr, view.const_data());
    ASSERT_EQ(size, this->x.size());
    ASSERT_EQ(size, view.size());
    ASSERT_EQ(exec, this->x.get_executor());
    ASSERT_EQ(exec, view.get_executor());
    ASSERT_TRUE(this->x.is_owning());
    ASSERT_FALSE(view.is_owning());
}


}  // namespace
