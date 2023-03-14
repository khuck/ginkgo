#ifndef GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP

#include <ginkgo/ginkgo.hpp>
#include "core/base/iterator_factory.hpp"

#include "types.hpp"

namespace gko {

/**
 * Struct to hold all necessary information for an all-to-all communication
 * on vectors with shared DOFs.
 */
struct comm_info_t {
    comm_info_t() = default;

    /**
     * Extracts communication pattern from a list of shared DOFs.
     *
     * @param comm
     * @param shared_idxs
     */
    comm_info_t(experimental::mpi::communicator comm,
                const array<shared_idx_t>& shared_idxs)
        : send_sizes(comm.size()),
          send_offsets(comm.size() + 1),
          recv_sizes(comm.size()),
          recv_offsets(comm.size() + 1),
          recv_idxs(shared_idxs.get_executor(), shared_idxs.get_num_elems())
    {
        auto exec = shared_idxs.get_executor()->get_master();
        std::vector<int> remote_idxs(shared_idxs.get_num_elems());
        std::vector<int> non_owned_idxs_temp;
        std::vector<int> recv_ranks(shared_idxs.get_num_elems());

        // this is basically AOS->SOA but with just counting the remote ranks
        for (int i = 0; i < shared_idxs.get_num_elems(); ++i) {
            auto& shared_idx = shared_idxs.get_const_data()[i];
            recv_sizes[shared_idx.remote_rank]++;

            recv_ranks[i] = shared_idx.remote_rank;
            recv_idxs.get_data()[i] = shared_idx.local_idx;

            remote_idxs[i] = shared_idx.remote_idx;

            if (shared_idx.owning_rank != comm.rank()) {
                non_owned_idxs_temp.push_back(shared_idx.local_idx);
            }
        }
        // sort by rank
        auto sort_it = detail::make_zip_iterator(
            recv_ranks.data(), recv_idxs.get_data(), remote_idxs.data());
        std::sort(sort_it, sort_it + shared_idxs.get_num_elems(),
                  [](const auto a, const auto b) {
                      return std::get<0>(a) < std::get<0>(b);
                  });

        std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                         recv_offsets.begin() + 1);

        // exchange recv_idxs to get which indices this rank has to sent to
        // every other rank
        comm.all_to_all(exec, recv_sizes.data(), 1, send_sizes.data(), 1);
        std::partial_sum(send_sizes.begin(), send_sizes.end(),
                         send_offsets.begin() + 1);

        comm.all_to_all_v(exec, remote_idxs.data(), recv_sizes.data(),
                          recv_offsets.data(), send_idxs.get_data(),
                          send_sizes.data(), send_offsets.data());

        non_owned_idxs = gko::array<LocalIndexType>{shared_idxs.get_executor(),
                                                    non_owned_idxs_temp.begin(),
                                                    non_owned_idxs_temp.end()};
    }

    // default variable all-to-all data
    std::vector<int> send_sizes;
    std::vector<int> send_offsets;
    std::vector<int> recv_sizes;
    std::vector<int> recv_offsets;

    // DOFs to send. These are dofs that are shared with other ranks,
    // but this rank owns them.
    gko::array<LocalIndexType> send_idxs;
    // DOFs to send. These are dofs that are shared with other ranks,
    // but other ranks own them. May overlap with send_idxs.
    gko::array<LocalIndexType> recv_idxs;
    // Keep track of which shared DOFs are not owned by this rank,
    // i.e. recv_idxs/send_idxs
    gko::array<LocalIndexType> non_owned_idxs;
};


struct overlapping_vec : public EnableLinOp<overlapping_vec, vec> {
    overlapping_vec(std::shared_ptr<const Executor> exec,
                    experimental::mpi::communicator comm = {MPI_COMM_NULL},
                    std::shared_ptr<vec> local_vec = nullptr,
                    comm_info_t comm_info = {})
        : EnableLinOp<overlapping_vec, vec>(local_vec->get_executor(),
                                            local_vec->get_size()),
          local_flag(local_vec->get_executor(), local_vec->get_size()[0]),
          num_ovlp(comm_info.non_owned_idxs.get_num_elems()),
          comm(comm),
          comm_info(comm_info)
    {
        static_cast<vec&>(*this) = *local_vec;
        local_flag.fill(true);
        for (int i = 0; i < comm_info.non_owned_idxs.get_num_elems(); ++i) {
            local_flag.get_data()[comm_info.non_owned_idxs.get_data()[i]] =
                false;
        }
    }

    /**
     * Creates a Dense matrix with the same size and stride as the callers
     * matrix.
     *
     * @returns a Dense matrix with the same size and stride as the caller.
     */
    std::unique_ptr<vec> create_with_same_config() const override
    {
        return std::make_unique<overlapping_vec>(
            this->get_executor(), comm,
            gko::clone(static_cast<const vec*>(this)), comm_info);
    }


    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    enum class operation { copy, add, average };

    /**
     * Updates non-local dofs from their respective ranks, using
     * the update scheme (copy, add) specified for each dof.
     *
     * Considering a distributed operator A = sum_i R_i^T D_i A_i R_i, this
     * will peform u_i = R_i R_i^T D_i u_i, eliminating the need for a
     * fully global vector
     *
     * The update operation corresponds to the entries of D_i for the shared
     * DOFs. Currently, copy results in D_i = 0 for non_owned_idxs and D_i =
     * 1/#shared_ranks for owned, but shared DOFS. The D_i = 1/#shared_ranks is
     * implicit, because it is assumed that the vector contains the same value
     * on these indices for all sharing ranks. This is the case in overlapping
     * FEM applications, because the local stiffness matrices will have the same
     * values for rows with DOFs that are shared between ranks and in the
     * interior of all overlapping domains.
     * Other D_i, for example D_i = 1/#shared_ranks for all shared DOFs (even
     * locally owned), would require more information than currently available.
     * This would require specifying DOFs that are locally owned, but shared
     * with other ranks explicitly as shared_dofs during the creation of
     * comm_info, as well as combining the sets recv_idxs and send_idxs.
     *
     * The other operations are not implemented yet.
     *
     * Note that all point-wise vector operations (adding, scaling, etc) keep
     * the consistency of the vector. Only operations where the update for an
     * entry i depends on other entries than i can make a vector non-consistent.
     * This usually applies for all operator applications.
     *
     * Perhaps something more generic to allow for user-defined transformations?
     */
    void make_consistent(operation op)
    {
        auto exec = this->get_executor();
        auto send_buffer =
            vec::create(exec, dim<2>(comm_info.send_offsets.back(), 1));
        auto recv_buffer =
            vec::create(exec, dim<2>(comm_info.recv_offsets.back(), 1));

        if (op == operation::copy) {
            if (comm_info.send_offsets.back() > 0) {
                // can't handle row gather with empty idxs??
                this->row_gather(&comm_info.send_idxs, send_buffer.get());
                comm.all_to_all_v(
                    exec, send_buffer->get_values(),
                    comm_info.send_sizes.data(), comm_info.send_offsets.data(),
                    recv_buffer->get_values(), comm_info.recv_sizes.data(),
                    comm_info.recv_offsets.data());
                // inverse row_gather
                // unnecessary if shared_idxs would be stored separately
                for (int i = 0; i < comm_info.recv_idxs.get_num_elems(); ++i) {
                    this->at(comm_info.recv_idxs.get_data()[i]) =
                        recv_buffer->at(i);
                }
            }
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    /**
     * Constraints the overlapping vector to the DOFs owned by this rank. This
     * can still contain DOFs shared with other ranks, but not exclusively owned
     * by them.
     */
    std::unique_ptr<vec> extract_local() const
    {
        auto exec = this->get_executor();
        auto no_ovlp_local =
            vec::create(exec, dim<2>{this->get_size()[0] - num_ovlp, 1});

        // copy-if, but in stupid
        // could be row_gather(complement(idx_set))
        int i = 0;
        for (int j = 0; j < this->get_size()[0]; ++j) {
            if (local_flag.get_const_data()[j]) {
                no_ovlp_local->at(i) = this->at(j);
                i++;
            }
        }

        return no_ovlp_local;
    }

    void compute_dot_impl(const LinOp* b, LinOp* result) const override
    {
        auto ovlp_b = dynamic_cast<const overlapping_vec*>(b);
        auto no_ovlp_b = ovlp_b->extract_local();
        auto no_ovlp_local = this->extract_local();

        auto dist_b = dist_vec::create(
            no_ovlp_b->get_executor(),
            as<const experimental::distributed::DistributedBase>(b)
                ->get_communicator(),
            no_ovlp_b.get());
        auto dist_local = dist_vec::create(no_ovlp_local->get_executor(), comm,
                                           no_ovlp_local.get());

        dist_local->compute_dot(dist_b.get(), result);
    }

    void compute_norm2_impl(LinOp* result) const override
    {
        auto no_ovlp_local = extract_local();

        dist_vec::create(no_ovlp_local->get_executor(), comm,
                         no_ovlp_local.get())
            ->compute_norm2(result);
    }

    array<int> local_flag;
    size_type num_ovlp;
    experimental::mpi::communicator comm;
    comm_info_t comm_info;
};


struct overlapping_operator
    : public experimental::distributed::DistributedBase,
      public experimental::EnableDistributedLinOp<overlapping_operator> {
    overlapping_operator(std::shared_ptr<const Executor> exec,
                         experimental::mpi::communicator comm,
                         std::shared_ptr<LinOp> local_op = nullptr,
                         comm_info_t comm_info = {})
        : experimental::distributed::DistributedBase(comm),
          experimental::EnableDistributedLinOp<overlapping_operator>{
              exec, local_op->get_size()},
          local_op(local_op),
          comm_info(comm_info)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        local_op->apply(b, x);
        // exchange data
        as<overlapping_vec>(x)->make_consistent(
            overlapping_vec::operation::copy);
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        auto copy_x = x->clone();
        apply_impl(b, x);
        as<vec>(x)->scale(alpha);
        as<vec>(x)->add_scaled(beta, copy_x);
    }

    std::shared_ptr<LinOp> local_op;
    comm_info_t comm_info;
};


}  // namespace gko


#endif  // GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_OVERLAP_HPP
