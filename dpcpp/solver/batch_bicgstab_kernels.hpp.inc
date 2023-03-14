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


namespace {

template <typename BatchMatrixType_entry, typename ValueType>
__dpct_inline__ void initialize(
    const int num_rows, const BatchMatrixType_entry& a_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType& rho_old, ValueType& omega,
    ValueType& alpha, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, ValueType* const r_hat_shared_entry,
    ValueType* const p_shared_entry, ValueType* const v_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norm,
    typename gko::remove_complex<ValueType>& res_norm,
    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    const auto group = item_ct1.get_group();

    rho_old = one<ValueType>();
    omega = one<ValueType>();
    alpha = one<ValueType>();

    // copy x from global to shared memory
    // r = b
    for (int iz = tid; iz < num_rows; iz += group_size) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0) {
        compute_norm2_sg_kernel(num_rows, r_shared_entry, res_norm, item_ct1);
    } else if (sg_id == 1) {
        // Compute norms of rhs
        compute_norm2_sg_kernel(num_rows, b_global_entry, rhs_norm, item_ct1);
    }

    for (int iz = tid; iz < num_rows; iz += group_size) {
        r_hat_shared_entry[iz] = r_shared_entry[iz];
        p_shared_entry[iz] = zero<ValueType>();
        v_shared_entry[iz] = zero<ValueType>();
    }
}


template <typename ValueType>
__dpct_inline__ void update_p(const int num_rows, const ValueType& rho_new,
                              const ValueType& rho_old, const ValueType& alpha,
                              const ValueType& omega,
                              const ValueType* const r_shared_entry,
                              const ValueType* const v_shared_entry,
                              ValueType* const p_shared_entry,
                              sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        const ValueType beta = (rho_new / rho_old) * (alpha / omega);
        p_shared_entry[r] =
            r_shared_entry[r] +
            beta * (p_shared_entry[r] - omega * v_shared_entry[r]);
    }
}

template <typename ValueType>
__dpct_inline__ void compute_alpha(const int num_rows, const ValueType& rho_new,
                                   const ValueType* const r_hat_shared_entry,
                                   const ValueType* const v_shared_entry,
                                   ValueType& alpha, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group = item_ct1.get_group();

    if (sg_id == 0) {
        compute_dot_product_sg_kernel(num_rows, r_hat_shared_entry,
                                      v_shared_entry, alpha, item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        alpha = rho_new / alpha;
    }
}


template <typename ValueType>
__dpct_inline__ void update_s(const int num_rows,
                              const ValueType* const r_shared_entry,
                              const ValueType& alpha,
                              const ValueType* const v_shared_entry,
                              ValueType* const s_shared_entry,
                              sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        s_shared_entry[r] = r_shared_entry[r] - alpha * v_shared_entry[r];
    }
}


template <typename ValueType>
__dpct_inline__ void compute_omega(const int num_rows,
                                   const ValueType* const t_shared_entry,
                                   const ValueType* const s_shared_entry,
                                   ValueType& temp, ValueType& omega,
                                   sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group = item_ct1.get_group();

    if (sg_id == 0) {
        compute_dot_product_sg_kernel(num_rows, t_shared_entry, s_shared_entry,
                                      omega, item_ct1);
    } else if (sg_id == 1) {
        compute_dot_product_sg_kernel(num_rows, t_shared_entry, t_shared_entry,
                                      temp, item_ct1);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        omega /= temp;
    }
}

template <typename ValueType>
__dpct_inline__ void update_x_and_r(
    const int num_rows, const ValueType* const p_hat_shared_entry,
    const ValueType* const s_hat_shared_entry, const ValueType& alpha,
    const ValueType& omega, const ValueType* const s_shared_entry,
    const ValueType* const t_shared_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r] +
                            omega * s_hat_shared_entry[r];
        r_shared_entry[r] = s_shared_entry[r] - omega * t_shared_entry[r];
    }
}


template <typename ValueType>
__dpct_inline__ void update_x_middle(const int num_rows, const ValueType& alpha,
                                     const ValueType* const p_hat_shared_entry,
                                     ValueType* const x_shared_entry,
                                     sycl::nd_item<3> item_ct1)
{
    for (int r = item_ct1.get_local_linear_id(); r < num_rows;
         r += item_ct1.get_local_range().size()) {
        x_shared_entry[r] = x_shared_entry[r] + alpha * p_hat_shared_entry[r];
    }
}

}  // namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void apply_kernel(const gko::kernels::batch_bicgstab::StorageConfig sconf,
                  const int max_iter, const gko::remove_complex<ValueType> tol,
                  LogType logger, PrecType prec_shared,
                  const BatchMatrixType A_global_entry,
                  const ValueType* const __restrict__ b_global_entry,
                  ValueType* const __restrict__ x_global_entry,
                  const size_type nrows, const size_type nnz,
                  ValueType* slm_values,
                  gko::remove_complex<ValueType>* slm_reals,
                  sycl::nd_item<3> item_ct1,
                  ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();
    const int tid = item_ct1.get_local_linear_id();
    auto group = item_ct1.get_group();
    const int group_size = item_ct1.get_local_range().size();

    const auto ibatch = item_ct1.get_group_linear_id();

    ValueType* rho_old_sh = &slm_values[0];
    ValueType* rho_new_sh = &slm_values[1];
    ValueType* alpha_sh = &slm_values[2];
    ValueType* omega_sh = &slm_values[3];
    ValueType* temp_sh = &slm_values[4];
    real_type* norms_rhs_sh = &slm_reals[0];
    real_type* norms_res_sh = &slm_reals[1];

    const int gmem_offset =
        ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* p_hat_sh;
    ValueType* s_hat_sh;
    ValueType* s_sh;
    ValueType* p_sh;
    ValueType* r_sh;
    ValueType* r_hat_sh;
    ValueType* v_sh;
    ValueType* t_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;
    if (sconf.n_shared >= 1) {
        p_hat_sh = &slm_values[5];
    } else {
        p_hat_sh = workspace + gmem_offset;
    }
    if (sconf.n_shared == 1) {
        s_hat_sh = workspace + gmem_offset;
    } else {
        s_hat_sh = p_hat_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 2) {
        v_sh = workspace + gmem_offset;
    } else {
        v_sh = s_hat_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 3) {
        t_sh = workspace + gmem_offset;
    } else {
        t_sh = v_sh + sconf.padded_vec_len;
    }
    if (!sconf.prec_shared && sconf.n_shared == 4) {
        prec_work_sh = workspace + gmem_offset;
    } else {
        prec_work_sh = t_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 4 && sconf.prec_shared) {
        p_sh = workspace + gmem_offset;
    } else {
        p_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
    }
    if (sconf.n_shared == 5) {
        s_sh = workspace + gmem_offset;
    } else {
        s_sh = p_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 6) {
        r_sh = workspace + gmem_offset;
    } else {
        r_sh = s_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 7) {
        r_hat_sh = workspace + gmem_offset;
    } else {
        r_hat_sh = r_sh + sconf.padded_vec_len;
    }
    if (sconf.n_shared == 8) {
        x_sh = workspace + gmem_offset;
    } else {
        x_sh = r_hat_sh + sconf.padded_vec_len;
    }
    // generate preconditioner
    prec_shared.generate(ibatch, A_global_entry, prec_work_sh, item_ct1);

    // initialization
    // rho_old = 1, omega = 1, alpha = 1
    // compute b norms
    // copy x from global to shared memory
    // r = b - A*x
    // compute residual norms
    // r_hat = r
    // p = 0
    // v = 0
    initialize(nrows, A_global_entry, b_global_entry, x_global_entry,
               rho_old_sh[0], omega_sh[0], alpha_sh[0], x_sh, r_sh, r_hat_sh,
               p_sh, v_sh, norms_rhs_sh[0], norms_res_sh[0], item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        if (stop.check_converged(norms_res_sh)) {
            break;
        }


        // rho_new =  < r_hat , r > = (r_hat)' * (r)
        if (sg_id == 0) {
            compute_dot_product_sg_kernel(nrows, r_hat_sh, r_sh, rho_new_sh[0],
                                          item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // beta = (rho_new / rho_old)*(alpha / omega)
        // p = r + beta*(p - omega * v)
        update_p(nrows, rho_new_sh[0], rho_old_sh[0], alpha_sh[0], omega_sh[0],
                 r_sh, v_sh, p_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // p_hat = precond * p
        prec_shared.apply(nrows, p_sh, p_hat_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // v = A * p_hat
        single_matvec_kernel(A_global_entry, p_hat_sh, v_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // alpha = rho_new / < r_hat , v>
        compute_alpha(nrows, rho_new_sh[0], r_hat_sh, v_sh, alpha_sh[0],
                      item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // s = r - alpha*v
        update_s(nrows, r_sh, alpha_sh[0], v_sh, s_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // an estimate of residual norms
        if (sg_id == 0) {
            compute_norm2_sg_kernel(nrows, s_sh, norms_res_sh[0], item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // if (norms_res_sh[0] / norms_rhs_sh[0] < tol) {
        if (stop.check_converged(norms_res_sh)) {
            update_x_middle(nrows, alpha_sh[0], p_hat_sh, x_sh, item_ct1);
            break;
        }

        // s_hat = precond * s
        prec_shared.apply(nrows, s_sh, s_hat_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // t = A * s_hat
        single_matvec_kernel(A_global_entry, s_hat_sh, t_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // omega = <t,s> / <t,t>
        compute_omega(nrows, t_sh, s_sh, temp_sh[0], omega_sh[0], item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // x = x + alpha*p_hat + omega *s_hat
        // r = s - omega * t
        update_x_and_r(nrows, p_hat_sh, s_hat_sh, alpha_sh[0], omega_sh[0],
                       s_sh, t_sh, x_sh, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (sg_id == 0) {
            compute_norm2_sg_kernel(nrows, r_sh, norms_res_sh[0], item_ct1);
        }

        if (tid == group_size - 1) {
            rho_old_sh[0] = rho_new_sh[0];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    logger.log_iteration(ibatch, iter, norms_res_sh[0]);

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
}