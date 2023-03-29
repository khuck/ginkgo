#ifndef GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP

#include <ginkgo/ginkgo.hpp>

#include "types.hpp"

template <typename T>
constexpr std::array<std::array<T, 3>, 3> A_loc{
    {{1.0, -0.5, -0.5}, {-0.5, 0.5, 0.0}, {-0.5, 0.0, 0.5}}};


/* divide each quadratic element in to two triangles
 * 0        1
 * |‾‾‾‾‾‾‾/|
 * | utr /  |
 * |  / ltr |
 * |/_______|
 * 2        0
 * The number denote the local index of the vertices.
 * The following two functions create mappings for a specific triangle (x,
 * y) from the local indexing to the global indexing.
 */
template <typename IndexType>
auto create_ltr_map(gko::size_type num_vertices_y,
                    gko::size_type num_vertices_x)
{
    return [=](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{y * num_vertices_x + x + 1,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return [=](const auto i) { return static_cast<IndexType>(map[i]); };
    };
}
template <typename IndexType>
auto create_utr_map(gko::size_type num_vertices_y,
                    gko::size_type num_vertices_x)
{
    return [=](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{(y + 1) * num_vertices_x + x,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return [=](const auto i) { return static_cast<IndexType>(map[i]); };
    };
}


template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> assemble(
    gko::size_type num_elements_y, gko::size_type num_elements_x,
    gko::size_type num_vertices_y, gko::size_type num_vertices_x,
    bool left_dirichlet_brdy, bool right_dirichlet_bdry,
    bool bottom_dirichlet_bdry, bool top_dirichlet_bdry)
{
    auto utr_map = create_utr_map<IndexType>(num_vertices_y, num_vertices_x);
    auto ltr_map = create_ltr_map<IndexType>(num_vertices_y, num_vertices_x);

    auto process_element = [](auto&& map, auto& data) {
        for (int jy = 0; jy < A_loc<ValueType>.size(); ++jy) {
            for (int jx = 0; jx < A_loc<ValueType>.size(); ++jx) {
                data.add_value(map(jy), map(jx), A_loc<ValueType>[jy][jx]);
            }
        }
    };

    auto process_boundary = [&](const std::vector<int>& local_bdry_idxs,
                                auto&& map, auto& data) {
        for (int i : local_bdry_idxs) {
            auto global_idx = map(i);
            auto global_idx_x = global_idx % num_vertices_x;
            auto global_idx_y = global_idx / num_vertices_x;

            if (global_idx_x != 0) {
                data.set_value(map(i), global_idx - 1, 0.0);
            }
            if (global_idx_x != num_vertices_x - 1) {
                data.set_value(map(i), global_idx + 1, 0.0);
            }
            if (global_idx_y != 0) {
                data.set_value(map(i), global_idx - num_vertices_x, 0.0);
            }
            if (global_idx_y != num_vertices_y - 1) {
                data.set_value(map(i), global_idx + num_vertices_x, 0.0);
            }

            data.set_value(map(i), map(i), 1.0);
        }
    };

    auto size = num_vertices_x * num_vertices_y;
    gko::matrix_assembly_data<ValueType, IndexType> data{
        gko::dim<2>{size, size}};

    for (int iy = 0; iy < num_elements_y; iy++) {
        for (int ix = 0; ix < num_elements_x; ix++) {
            // handle upper triangle
            process_element(utr_map(iy, ix), data);

            // handle lower triangle
            process_element(ltr_map(iy, ix), data);
        }
    }
    for (int iy = 0; iy < num_elements_y; iy++) {
        for (int ix = 0; ix < num_elements_x; ix++) {
            // handle boundary
            if (ix == 0 && left_dirichlet_brdy) {
                process_boundary({0, 2}, utr_map(iy, ix), data);
            }
            if (ix == num_elements_x - 1 && right_dirichlet_bdry) {
                process_boundary({0, 1}, ltr_map(iy, ix), data);
            }
            if (iy == 0 && bottom_dirichlet_bdry) {
                process_boundary({0, 2}, ltr_map(iy, ix), data);
            }
            if (iy == num_elements_y - 1 && top_dirichlet_bdry) {
                process_boundary({0, 1}, utr_map(iy, ix), data);
            }
        }
    }

    return data.get_ordered_data();
}


// u(0) = u(1) = 1
// values in the interior will be overwritten during the communication
// also set initial guess to dirichlet condition
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> assemble_rhs(
    gko::size_type num_vertices_y, gko::size_type num_vertices_x,
    bool left_brdy, bool right_brdy, bool bottom_bdry, bool top_bdry)
{
    auto f_one = [&](const auto iy, const auto ix) { return 1.0; };
    auto f_linear = [&](const auto iy, const auto ix) {
        return 0.5 * (ix / (num_vertices_x - 1) + iy / (num_vertices_y - 1));
    };

    auto size = num_vertices_x * num_vertices_y;
    gko::matrix_assembly_data<ValueType, IndexType> data{gko::dim<2>{size, 1}};

    // vertical boundaries
    for (int i = 0; i < num_vertices_y; i++) {
        if (left_brdy) {
            auto idx = i * num_vertices_x;
            data.set_value(idx, 0, f_one(i, 0));
        }
        if (right_brdy) {
            auto idx = (i + 1) * num_vertices_x - 1;
            data.set_value(idx, 0, f_one(i, num_vertices_x - 1));
        }
    }
    // horizontal boundaries
    for (int i = 0; i < num_vertices_x; i++) {
        if (top_bdry) {
            auto idx = i;
            data.set_value(idx, 0, f_one(0, i));
        }
        if (bottom_bdry) {
            auto idx = i + (num_vertices_y - 1) * num_vertices_x;
            data.set_value(idx, 0, f_one(num_vertices_y - 1, i));
        }
    }
    return data.get_ordered_data();
}


std::array<gko::size_type, 2> add_overlap(const std::array<int, 2>& dims,
                                          const std::array<int, 2>& coords,
                                          gko::size_type num_interior_elements,
                                          gko::size_type overlap)
{
    std::array<std::array<bool, 2>, 2> on_bdry = {
        {{coords[0] == 0, coords[0] == dims[0] - 1},
         {coords[1] == 0, coords[1] == dims[1] - 1}}};

    std::array<int, 2> num_boundary_intersections = {
        on_bdry[0][0] + on_bdry[1][0],
        on_bdry[0][1] + on_bdry[1][1],
    };

    return {
        num_interior_elements + overlap * (2 - num_boundary_intersections[0]),
        num_interior_elements + overlap * (2 - num_boundary_intersections[1])};
}


std::vector<shared_idx_t> setup_shared_idxs(
    const gko::experimental::mpi::communicator& comm,
    const std::array<int, 2> dims, const std::array<int, 2> coords,
    gko::size_type num_interior_elements, gko::size_type num_elements_y,
    gko::size_type num_elements_x, int overlap_size)
{
    auto exec = gko::ReferenceExecutor::create();

    auto this_rank = comm.rank();

    std::array<std::array<bool, 2>, 2> on_bdry = {
        {{coords[0] == 0, coords[0] == dims[0] - 1},
         {coords[1] == 0, coords[1] == dims[1] - 1}}};


    auto target_rank = [=](int dx, int dy) {
        return coords[0] + dx + dims[0] * (coords[1] + dy);
    };

    auto relative_id = [=](int dx, int dy) { return dx + 3 * dy + 4; };

    std::array<std::array<gko::size_type, 2>, 9> neighbor_num_elements{};
    std::array<gko::size_type, 9> neighbor_num_elements_y{};

    std::array<gko::size_type, 2> num_elements{num_elements_x, num_elements_y};


    std::vector<gko::experimental::mpi::request> reqs;
    if (!on_bdry[0][0]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(-1, 0),
                    relative_id(1, 0));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(-1, 0)].data(),
                        2, target_rank(-1, 0), relative_id(-1, 0)));
    }
    if (!on_bdry[0][1]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(1, 0),
                    relative_id(-1, 0));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(1, 0)].data(),
                        2, target_rank(1, 0), relative_id(1, 0)));
    }
    if (!on_bdry[1][0]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(0, -1),
                    relative_id(0, 1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(0, -1)].data(),
                        2, target_rank(0, -1), relative_id(0, -1)));
    }
    if (!on_bdry[1][1]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(0, 1),
                    relative_id(0, -1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(0, 1)].data(),
                        2, target_rank(0, 1), relative_id(0, 1)));
    }
    // diagonal neighbors
    if (!on_bdry[0][0] && !on_bdry[1][0]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(-1, -1),
                    relative_id(1, 1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(-1, -1)].data(),
                        2, target_rank(-1, -1), relative_id(-1, -1)));
    }
    if (!on_bdry[0][1] && !on_bdry[1][0]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(1, -1),
                    relative_id(-1, 1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(1, -1)].data(),
                        2, target_rank(-1, 0), relative_id(1, -1)));
    }
    if (!on_bdry[0][0] && !on_bdry[1][1]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(-1, 1),
                    relative_id(1, -1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(-1, 1)].data(),
                        2, target_rank(-1, 1), relative_id(-1, 1)));
    }
    if (!on_bdry[0][1] && !on_bdry[1][1]) {
        comm.i_send(exec, num_elements.data(), 2, target_rank(1, 1),
                    relative_id(-1, -1));
        reqs.emplace_back(
            comm.i_recv(exec, neighbor_num_elements[relative_id(1, 1)].data(),
                        2, target_rank(1, 1), relative_id(1, 1)));
    }


    std::vector<shared_idx_t> shared_idxs;
    auto utr_map = create_utr_map<int>(num_elements_y + 1, num_elements_x + 1);
    auto ltr_map = create_ltr_map<int>(num_elements_y + 1, num_elements_x + 1);
    // TODO: should remove physical boundary idxs
    auto fixed_x_map = [&](const auto x, const auto offset, auto&& map) {
        return [=](const auto y) { return map(y + offset, x); };
    };
    auto fixed_y_map = [&](const auto y, const auto offset, auto&& map) {
        return [=](const auto x) { return map(y, x + offset); };
    };
    auto setup_idxs = [num_interior_elements](
                          auto&& partial_map_local, auto&& partial_map_remote,
                          int remote_rank,
                          const std::vector<int> element_local_bdry_idx,
                          std::vector<shared_idx_t>& idxs) {
        for (int iy = 0; iy < num_interior_elements; ++iy) {
            auto local_map = partial_map_local(iy);
            auto remote_map = partial_map_remote(iy);
            if (iy == 0) {
                idxs.push_back({local_map(element_local_bdry_idx[0]),
                                remote_map(element_local_bdry_idx[0]),
                                remote_rank, remote_rank});
            }
            idxs.push_back({local_map(element_local_bdry_idx[1]),
                            remote_map(element_local_bdry_idx[1]), remote_rank,
                            remote_rank});
        }
    };

    if (!on_bdry[0][0]) {
        auto neighbor_ne = neighbor_num_elements[relative_id(-1, 0)];
        auto neighbor_utr_map =
            create_utr_map<int>(neighbor_ne[1] + 1, neighbor_ne[0] + 1);
        for (int overlap_idx = 0; overlap_idx < overlap_size; ++overlap_idx) {
            setup_idxs(
                fixed_x_map(overlap_idx, on_bdry[1][0], utr_map),
                fixed_x_map(neighbor_ne[0] - 2 * overlap_size + overlap_idx,
                            on_bdry[1][0], neighbor_utr_map),
                this_rank - 1, {2, 0}, shared_idxs);
        }
    }
    if (!on_bdry[0][1]) {
        auto neighbor_ne = neighbor_num_elements[relative_id(-1, 0)];
        auto neighbor_ltr_map =
            create_ltr_map<int>(neighbor_ne[1] + 1, neighbor_ne[0] + 1);

        // one additional layer to create a partition of unity, currently uses
        // unique ownership
        for (int overlap_idx = 0; overlap_idx < overlap_size + 1;
             ++overlap_idx) {
            setup_idxs(fixed_x_map(num_elements_x - 1 - overlap_idx,
                                   on_bdry[1][0], ltr_map),
                       fixed_x_map(2 * overlap_size - 1 - overlap_idx,
                                   on_bdry[1][0], neighbor_ltr_map),
                       this_rank + 1, {0, 1}, shared_idxs);
        }
    }
    if (!on_bdry[1][0]) {
        auto neighbor_ne = neighbor_num_elements[relative_id(0, -1)];
        auto neighbor_ltr_map =
            create_ltr_map<int>(neighbor_ne[1] + 1, neighbor_ne[0] + 1);
        for (int overlap_idx = 0; overlap_idx < overlap_size; ++overlap_idx) {
            setup_idxs(
                fixed_y_map(overlap_idx, on_bdry[0][1], ltr_map),
                fixed_y_map(neighbor_ne[0] - 2 * overlap_size + overlap_idx,
                            on_bdry[0][1], neighbor_ltr_map),
                this_rank - 1, {2, 0}, shared_idxs);
        }
    }
    return shared_idxs;
}


std::vector<shared_idx_t> setup_non_ovlp_shared_idxs(
    const gko::experimental::mpi::communicator& comm,
    gko::size_type num_elements_y, gko::size_type num_elements_x)
{
    auto exec = gko::ReferenceExecutor::create();

    auto this_rank = comm.rank();
    auto share_left_bdry = this_rank > 0;
    auto share_right_bdry = this_rank < comm.size() - 1;

    gko::experimental::mpi::request req_l;
    gko::experimental::mpi::request req_r;
    gko::size_type left_neighbor_nex;
    gko::size_type right_neighbor_nex;
    if (share_left_bdry) {
        comm.i_send(exec, &num_elements_x, 1, this_rank - 1, this_rank - 1);
        req_l =
            comm.i_recv(exec, &left_neighbor_nex, 1, this_rank - 1, this_rank);
    }
    if (share_right_bdry) {
        comm.i_send(exec, &num_elements_x, 1, this_rank + 1, this_rank + 1);
        req_r =
            comm.i_recv(exec, &right_neighbor_nex, 1, this_rank + 1, this_rank);
    }
    req_l.wait();
    req_r.wait();

    std::vector<shared_idx_t> shared_idxs;
    auto utr_map = create_utr_map<int>(num_elements_y + 1, num_elements_x + 1);
    auto ltr_map = create_ltr_map<int>(num_elements_y + 1, num_elements_x + 1);
    // TODO: should remove physical boundary idxs
    auto fixed_x_map = [&](const auto x, auto&& map) {
        return [=](const auto y) { return map(y, x); };
    };
    auto setup_idxs = [num_elements_y, this_rank](
                          auto&& partial_map_local, auto&& partial_map_remote,
                          int remote_rank,
                          const std::vector<int> element_local_bdry_idx,
                          const std::vector<int> element_neighbor_bdry_idx,
                          std::vector<shared_idx_t>& idxs) {
        for (int iy = 1; iy < num_elements_y - 1; ++iy) {
            auto local_map = partial_map_local(iy);
            auto remote_map = partial_map_remote(iy);
            idxs.push_back({local_map(element_local_bdry_idx[0]),
                            remote_map(element_neighbor_bdry_idx[0]),
                            remote_rank, this_rank});
            if (iy == num_elements_y - 2) {
                idxs.push_back({local_map(element_local_bdry_idx[1]),
                                remote_map(element_neighbor_bdry_idx[1]),
                                remote_rank, this_rank});
            }
        }
    };

    if (share_left_bdry) {
        auto neighbor_ltr_map =
            create_ltr_map<int>(num_elements_y + 1, left_neighbor_nex + 1);
        setup_idxs(fixed_x_map(0, utr_map),
                   fixed_x_map(left_neighbor_nex - 1, neighbor_ltr_map),
                   this_rank - 1, {2, 0}, {0, 1}, shared_idxs);
    }
    if (share_right_bdry) {
        auto neighbor_utr_map =
            create_utr_map<int>(num_elements_y + 1, right_neighbor_nex + 1);

        setup_idxs(fixed_x_map(num_elements_x - 1, ltr_map),
                   fixed_x_map(0, neighbor_utr_map), this_rank + 1, {0, 1},
                   {2, 0}, shared_idxs);
    }
    return shared_idxs;
}


#endif  // GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP
