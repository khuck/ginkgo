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

#include <ginkgo/core/base/executor.hpp>


#include <exception>
#include <memory>
#include <type_traits>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class DpcppExecutor : public ::testing::Test {
protected:
    DpcppExecutor()
        : ref(gko::ReferenceExecutor::create()), dpcpp(nullptr), dpcpp2(nullptr)
    {}

    void SetUp()
    {
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            dpcpp = gko::DpcppExecutor::create(0, ref, "gpu");
            if (gko::DpcppExecutor::get_num_devices("gpu") > 1) {
                dpcpp2 = gko::DpcppExecutor::create(1, ref, "gpu");
            }
        } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
            dpcpp = gko::DpcppExecutor::create(0, ref, "cpu");
            if (gko::DpcppExecutor::get_num_devices("cpu") > 1) {
                dpcpp2 = gko::DpcppExecutor::create(1, ref, "cpu");
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    void TearDown()
    {
        // ensure that previous calls finished and didn't throw an error
        ASSERT_NO_THROW(dpcpp->synchronize());
        if (dpcpp2 != nullptr) {
            ASSERT_NO_THROW(dpcpp2->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> ref{};
    std::shared_ptr<const gko::DpcppExecutor> dpcpp{};
    std::shared_ptr<const gko::DpcppExecutor> dpcpp2{};
};


TEST_F(DpcppExecutor, CanInstantiateTwoExecutorsOnOneDevice)
{
    auto dpcpp = gko::DpcppExecutor::create(0, ref);
    if (dpcpp2 != nullptr) {
        auto dpcpp2 = gko::DpcppExecutor::create(0, ref);
    }

    // We want automatic deinitialization to not create any error
}


TEST_F(DpcppExecutor, CanGetExecInfo)
{
    dpcpp = gko::DpcppExecutor::create(0, ref);

    ASSERT_TRUE(dpcpp->get_num_computing_units() > 0);
    ASSERT_TRUE(dpcpp->get_subgroup_sizes().size() > 0);
    ASSERT_TRUE(dpcpp->get_max_workitem_sizes().size() > 0);
    ASSERT_TRUE(dpcpp->get_max_workgroup_size() > 0);
    ASSERT_TRUE(dpcpp->get_max_subgroup_size() > 0);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeAll)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::all).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("all");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeCPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::cpu).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("cpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeGPU)
{
    auto count = sycl::device::get_devices(sycl::info::device_type::gpu).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("gpu");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, KnowsNumberOfDevicesOfTypeAccelerator)
{
    auto count =
        sycl::device::get_devices(sycl::info::device_type::accelerator).size();

    auto num_devices = gko::DpcppExecutor::get_num_devices("accelerator");

    ASSERT_EQ(count, num_devices);
}


TEST_F(DpcppExecutor, Synchronizes)
{
    // Todo design a proper unit test once we support streams
    ASSERT_NO_THROW(dpcpp->synchronize());
}


TEST_F(DpcppExecutor, FreeAfterKernel)
{
    size_t length = 10000;
    auto dpcpp =
        gko::DpcppExecutor::create(0, gko::ReferenceExecutor::create());
    {
        gko::Array<float> x(dpcpp, length);
        gko::Array<float> y(dpcpp, length);
        auto x_val = x.get_data();
        auto y_val = y.get_data();
        dpcpp->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>{length},
                             [=](sycl::id<1> i) { y_val[i] += x_val[i]; });
        });
    }
    // to ensure everything on queue is finished.
    dpcpp->synchronize();
}


}  // namespace
