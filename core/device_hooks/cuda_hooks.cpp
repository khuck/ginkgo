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

#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/version.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {


version version_info::get_cuda_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


std::shared_ptr<CudaExecutor> CudaExecutor::create(
    int device_id, std::shared_ptr<Executor> master, bool device_reset,
    allocation_mode alloc_mode, CUstream_st* stream)
{
    return std::shared_ptr<CudaExecutor>(new CudaExecutor(
        device_id, std::move(master), device_reset, alloc_mode, stream));
}


void CudaExecutor::populate_exec_info(const machine_topology* mach_topo)
{
    // This method is always called, so cannot throw when not compiled.
}


void OmpExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                              const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_free(void* ptr) const noexcept
{
    // Free must never fail, as it can be called in destructors.
    // If the nvidia module was not compiled, the library couldn't have
    // allocated the memory, so there is no need to deallocate it.
}


void* CudaExecutor::raw_alloc(size_type num_bytes) const GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const OmpExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const CudaExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const HipExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::raw_copy_to(const DpcppExecutor*, size_type num_bytes,
                               const void* src_ptr, void* dest_ptr) const
    GKO_NOT_COMPILED(cuda);


void CudaExecutor::synchronize() const GKO_NOT_COMPILED(cuda);


scoped_device_id_guard CudaExecutor::get_scoped_device_id_guard() const
    GKO_NOT_COMPILED(cuda);


std::string CudaError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CublasError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CurandError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CusparseError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


std::string CufftError::get_error(int64)
{
    return "ginkgo CUDA module is not compiled";
}


int CudaExecutor::get_num_devices() { return 0; }


void CudaExecutor::set_gpu_property() {}


void CudaExecutor::init_handles() {}


scoped_device_id_guard::scoped_device_id_guard(const CudaExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(cuda);


cuda_stream::cuda_stream(int device_id) GKO_NOT_COMPILED(cuda);


cuda_stream::~cuda_stream() {}


cuda_stream::cuda_stream(cuda_stream&&) GKO_NOT_COMPILED(cuda);


CUstream_st* cuda_stream::get() const GKO_NOT_COMPILED(cuda);


CudaTimer::CudaTimer(std::shared_ptr<const CudaExecutor> exec)
    GKO_NOT_COMPILED(cuda);


time_point CudaTimer::record() GKO_NOT_COMPILED(cuda);


int64 CudaTimer::difference(const time_point& start, const time_point& stop)
    GKO_NOT_COMPILED(cuda);


namespace kernels {
namespace cuda {


void reset_device(int device_id) GKO_NOT_COMPILED(cuda);


void destroy_event(CUevent_st* event) GKO_NOT_COMPILED(cuda);


}  // namespace cuda
}  // namespace kernels


namespace log {


void init_nvtx() GKO_NOT_COMPILED(cuda);


std::function<void(const char*, profile_event_category)> begin_nvtx_fn(
    uint32_t color_rgb) GKO_NOT_COMPILED(cuda);


void end_nvtx(const char*, profile_event_category) GKO_NOT_COMPILED(cuda);


}  // namespace log
}  // namespace gko


#define GKO_HOOK_MODULE cuda
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
