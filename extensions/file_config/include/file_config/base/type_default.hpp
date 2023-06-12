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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_DEFAULT_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_DEFAULT_HPP_


#include <string>  // for string


#include "file_config/base/type_pack.hpp"    // for tt_list
#include "file_config/base/type_string.hpp"  // for get_string
#include "ginkgo/core/base/types.hpp"        // for int32, int64


namespace gko {
namespace extensions {
namespace file_config {


// denote the type we have the default supported list
enum class handle_type { ValueType, IndexType };


// tt_list_g::type will give the tt_list<supported types>
template <handle_type enum_item>
struct tt_list_g;

#define TT_LIST_G_PARTIAL(_enum, ...)      \
    template <>                            \
    struct tt_list_g<handle_type::_enum> { \
        using type = tt_list<__VA_ARGS__>; \
    }

TT_LIST_G_PARTIAL(ValueType, double, float, std::complex<double>,
                  std::complex<float>);
TT_LIST_G_PARTIAL(IndexType, int32, int64);


template <handle_type T>
using tt_list_g_t = typename tt_list_g<T>::type;


// return the default template
template <handle_type enum_item>
inline std::string get_default_string();

#define GET_DEFAULT_STRING_PARTIAL(_enum, _type)                             \
    template <>                                                              \
    inline std::string get_default_string<handle_type::_enum>()              \
    {                                                                        \
        return get_string<_type>();                                          \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GET_DEFAULT_STRING_PARTIAL(ValueType, double);
GET_DEFAULT_STRING_PARTIAL(IndexType, int);


}  // namespace file_config
}  // namespace extensions
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_TYPE_DEFAULT_HPP_
