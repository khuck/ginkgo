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

#ifndef GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_CG_HPP_
#define GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_CG_HPP_


#include <ginkgo/core/solver/cg.hpp>


#include "resource_manager/base/generic_constructor.hpp"
#include "resource_manager/base/helper.hpp"
#include "resource_manager/base/macro_helper.hpp"
#include "resource_manager/base/rapidjson_helper.hpp"
#include "resource_manager/base/type_default.hpp"
#include "resource_manager/base/type_pack.hpp"
#include "resource_manager/base/type_resolving.hpp"
#include "resource_manager/base/type_string.hpp"
#include "resource_manager/base/types.hpp"


namespace gko {
namespace extension {
namespace resource_manager {


template <typename ValueType>
struct Generic<typename gko::solver::Cg<ValueType>::Factory,
               gko::solver::Cg<ValueType>> {
    using type = std::shared_ptr<typename gko::solver::Cg<ValueType>::Factory>;
    static type build(rapidjson::Value& item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager* manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY(gko::solver::Cg<ValueType>, manager, item, exec,
                          linop);
            SET_POINTER_VECTOR(const stop::CriterionFactory, criteria);
            SET_POINTER(const LinOpFactory, preconditioner);
            SET_POINTER(const LinOp, generated_preconditioner);
            SET_EXECUTOR;
        }();
        return std::move(ptr);
    }
};


SIMPLE_LINOP_WITH_FACTORY_IMPL(gko::solver::Cg, typename ValueType, ValueType);


ENABLE_SELECTION(cg_factory_select, call, std::shared_ptr<gko::LinOpFactory>,
                 get_actual_factory_type);
ENABLE_SELECTION(cg_select, call, std::shared_ptr<gko::LinOp>, get_actual_type);


constexpr auto cg_list =
    typename span_list<tt_list_g_t<handle_type::ValueType>>::type();


template <>
std::shared_ptr<gko::LinOpFactory> create_from_config<
    RM_LinOpFactory, RM_LinOpFactory::CgFactory, gko::LinOpFactory>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()));
    auto ptr = cg_factory_select<gko::solver::Cg>(
        cg_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}

template <>
std::shared_ptr<gko::LinOp>
create_from_config<RM_LinOp, RM_LinOp::Cg, gko::LinOp>(
    rapidjson::Value& item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    // go though the type
    auto type_string = create_type_name(  // trick for clang-format
        get_value_with_default(item, "ValueType",
                               get_default_string<handle_type::ValueType>()));
    auto ptr = cg_select<gko::solver::Cg>(
        cg_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}


}  // namespace resource_manager
}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_RESOURCE_MANAGER_SOLVER_CG_HPP_