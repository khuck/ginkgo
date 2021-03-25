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

template <typename PrecType, typename StopType, typename MatrixType,
          typename ValueType>
__global__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> rel_tol,
    const gko::remove_complex<ValueType> relax_factor, const MatrixType mat,
    const gko::batch_dense::UniformBatch<const ValueType> b,
    const gko::batch_dense::UniformBatch<ValueType> x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = mat.num_batch;
    const auto num_rows = mat.num_rows;
    const auto num_rhs = b.num_rhs;
    constexpr int max_nrhs = batch_config<ValueType>::max_num_rhs;

    if (batch_config<ValueType>::max_num_rows < num_rows * num_rhs) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(" Static vector size not enough!\n");
        }
        return;
    }

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        __shared__ ValueType residual[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType delta_x[batch_config<ValueType>::max_num_rows];
        __shared__ ValueType prec_work[PrecType::work_size];
        uint32 converged = 0;

        const gko::batch_dense::BatchEntry<ValueType> r_b{residual, num_rhs,
                                                          num_rows, num_rhs};
        const gko::batch_dense::BatchEntry<ValueType> dx_b{delta_x, num_rhs,
                                                           num_rows, num_rhs};
        const typename MatrixType::entry_type a_b =
            gko::batch::batch_entry(mat, ibatch);
        const gko::batch_dense::BatchEntry<const ValueType> b_b =
            gko::batch::batch_entry(b, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> x_b =
            gko::batch::batch_entry(x, ibatch);

        // These need to be duplicated across threads
        const ValueType one[] = {1.0};
        const gko::batch_dense::BatchEntry<const ValueType> one_b{one, 1, 1, 1};

        __shared__ real_type norms[max_nrhs];
        for (int iz = threadIdx.x; iz < max_nrhs; iz += blockDim.x) {
            norms[iz] = 0.0;
        }
        const gko::batch_dense::BatchEntry<real_type> norms_b{norms, max_nrhs,
                                                              1, num_rhs};

        PrecType prec(a_b, prec_work);

        // initial residual
        for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
            const int i = iz / num_rhs;
            const int j = iz % num_rhs;
            r_b.values[i * r_b.stride + j] = b_b.values[i * b_b.stride + j];
        }
        batch_dense::compute_norm2<ValueType>(gko::batch_dense::to_const(r_b),
                                              norms_b);
        __shared__ real_type init_rel_res_norm[max_nrhs];
        for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
            init_rel_res_norm[iz] = sqrt(norms_b.values[iz]);
        }

        StopType stop(num_rhs, max_iter, rel_tol, converged, init_rel_res_norm);

        int iter = 0;
        while (1) {
            // r <- r - Adx  This causes instability!
            // batch_adv_spmv_single(static_cast<ValueType>(-1.0), a_b,
            //     gko::batch_dense::to_const(dx_b),
            //     static_cast<ValueType>(1.0), r_b);

            // r <- b - Ax
            for (int iz = threadIdx.x; iz < num_rows * num_rhs;
                 iz += blockDim.x) {
                const int i = iz / num_rhs;
                const int j = iz % num_rhs;
                r_b.values[i * r_b.stride + j] = b_b.values[i * b_b.stride + j];
            }
            __syncthreads();
            batch_adv_spmv_single(static_cast<ValueType>(-1.0), a_b,
                                  gko::batch_dense::to_const(x_b),
                                  static_cast<ValueType>(1.0), r_b);
            __syncthreads();

            batch_dense::compute_norm2<ValueType>(
                gko::batch_dense::to_const(r_b), norms_b);
            __syncthreads();
            for (int j = threadIdx.x; j < num_rhs; j += blockDim.x) {
                norms[j] = sqrt(norms[j]);
            }

            const bool all_converged =
                stop.check_converged(iter, norms, {NULL, 0, 0, 0}, converged);
            if (all_converged) {
                break;
            }

            prec.apply(gko::batch_dense::to_const(r_b), dx_b);
            __syncthreads();

            // zero out dx for rhs's which do not need to be updated,
            //  though this is unnecessary for this solver.
            for (int j = 0; j < num_rhs; j++) {
                const uint32 conv = converged & (1 << j);
                if (conv) {
                    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
                        dx_b.values[i * dx_b.stride + j] = 0.0;
                    }
                }
            }
            __syncthreads();

            batch_dense::add_scaled(one_b, gko::batch_dense::to_const(dx_b),
                                    x_b);
            __syncthreads();
            iter++;
        }
    }
}