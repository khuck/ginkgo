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

#include <ginkgo/core/config/config.hpp>


#include <map>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace config {


buildfromconfig_map generate_config_map()
{
    return {{"Cg", build_from_config<LinOpFactoryType::Cg>},
            {"Bicg", build_from_config<LinOpFactoryType::Bicg>},
            {"Bicgstab", build_from_config<LinOpFactoryType::Bicgstab>},
            {"Cgs", build_from_config<LinOpFactoryType::Cgs>},
            {"Fcg", build_from_config<LinOpFactoryType::Fcg>}};
}


std::unique_ptr<gko::LinOpFactory> build_from_config(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, type_descriptor td)
{
    if (config.contains("Type")) {
        auto func = context.get_build_map().at(
            config.at("Type").get_data<std::string>());
        return func(config, context, exec, td);
    }
    GKO_INVALID_STATE("Should contain Type property");
}


}  // namespace config
}  // namespace gko
