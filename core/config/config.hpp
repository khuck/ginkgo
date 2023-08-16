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

#ifndef GKO_CORE_CONFIG_CONFIG_HPP_
#define GKO_CORE_CONFIG_CONFIG_HPP_


#include <string>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {

/**
 * This function is to update the default type setting from current config.
 *
 * @note It might update the unused type for the current class.
 */
inline TypeDescriptor update_type(const Config& config,
                                  const TypeDescriptor& td)
{
    TypeDescriptor updated = td;

    if (config.contains("ValueType")) {
        updated.first = config.at("ValueType").get_data<std::string>();
    }
    if (config.contains("IndexType")) {
        updated.second = config.at("IndexType").get_data<std::string>();
    }
    return updated;
}


using item = pnode;

template <typename T>
inline std::shared_ptr<T> get_pointer(const item& item, const registry& context,
                                      std::shared_ptr<const Executor> exec,
                                      TypeDescriptor td)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    ptr = context.search_data<T_non_const>(item.get_data<std::string>());
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
inline std::shared_ptr<const LinOpFactory> get_pointer<const LinOpFactory>(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    std::shared_ptr<const LinOpFactory> ptr;
    if (item.is(pnode::status_t::object)) {
        ptr = context.search_data<LinOpFactory>(item.get_data<std::string>());
    } else if (item.is(pnode::status_t::list)) {
        ptr = build_from_config(item, context, exec, td);
    }
    // handle object is item
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
std::shared_ptr<const stop::CriterionFactory>
get_pointer<const stop::CriterionFactory>(const item& item,
                                          const registry& context,
                                          std::shared_ptr<const Executor> exec,
                                          TypeDescriptor td);


template <typename T>
inline std::vector<std::shared_ptr<T>> get_pointer_vector(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td)
{
    std::vector<std::shared_ptr<T>> res;
    // for loop in item
    if (item.is(pnode::status_t::array)) {
        for (const auto& it : item.get_array()) {
            res.push_back(get_pointer<T>(it, context, exec, td));
        }
    } else {
        // only one item can be passed without array
        res.push_back(get_pointer<T>(item, context, exec, td));
    }

    return res;
}

template <>
std::vector<std::shared_ptr<const stop::CriterionFactory>>
get_pointer_vector<const stop::CriterionFactory>(
    const item& item, const registry& context,
    std::shared_ptr<const Executor> exec, TypeDescriptor td);


template <typename IndexType, typename = typename std::enable_if<
                                  std::is_integral<IndexType>::value>::type>
inline IndexType get_value(const item& item)
{
    auto val = item.get_data<long long int>();
    assert(val <= std::numeric_limits<IndexType>::max() &&
           val >= std::numeric_limits<IndexType>::min());
    return static_cast<IndexType>(val);
}


#define SET_POINTER(_factory, _param_type, _param_name, _config, _context,     \
                    _exec, _td)                                                \
    {                                                                          \
        if (_config.contains(#_param_name)) {                                  \
            _factory.with_##_param_name(gko::config::get_pointer<_param_type>( \
                _config.at(#_param_name), _context, _exec, _td));              \
        }                                                                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


#define SET_POINTER_VECTOR(_factory, _param_type, _param_name, _config,      \
                           _context, _exec, _td)                             \
    {                                                                        \
        if (_config.contains(#_param_name)) {                                \
            _factory.with_##_param_name(                                     \
                gko::config::get_pointer_vector<_param_type>(                \
                    _config.at(#_param_name), _context, _exec, _td));        \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define SET_VALUE(_factory, _param_type, _param_name, _config)               \
    {                                                                        \
        if (_config.contains(#_param_name)) {                                \
            _factory.with_##_param_name(gko::config::get_value<_param_type>( \
                _config.at(#_param_name)));                                  \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


// If we do not put the build_from_config in the class directly, the following
// can also be in internal header.
template <typename T>
struct type_string {
    static std::string str() { return "N"; };
};

#define TYPE_STRING_OVERLOAD(_type, _str)         \
    template <>                                   \
    struct type_string<_type> {                   \
        static std::string str() { return _str; } \
    }

TYPE_STRING_OVERLOAD(double, "double");
TYPE_STRING_OVERLOAD(float, "float");
TYPE_STRING_OVERLOAD(std::complex<double>, "complex<double>");
TYPE_STRING_OVERLOAD(std::complex<float>, "complex<float>");


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_CONFIG_HPP_
