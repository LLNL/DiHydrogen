////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/backend.hpp"
#include "distconv/dnn_backend/backend_dace.hpp"

#include <dlfcn.h>

#include <sstream>
#include <string>

namespace distconv
{
// DaCe-generated function types
typedef dacehandle_t (*initfunc_t)();
typedef void (*exitfunc_t)(dacehandle_t handle);
typedef bool (*setstream_t)(dacehandle_t handle, backend::Stream_t stream);

template <class C, class T>
std::basic_ostream<C, T>& write_s5d(std::basic_ostream<C, T>& os, const s5d& s)
{
    return os << std::get<0>(s) << '_' << std::get<1>(s) << '_'
              << std::get<2>(s) << '_' << std::get<3>(s) << '_'
              << std::get<4>(s);
}

std::string ConvDescriptor::hash() const
{
    std::stringstream stream;
    const char* ctype =
        (this->type == FORWARD
             ? "fwd"
             : (this->type == BACKWARD_FILTER ? "bwdfilt" : "bwddata"));
    stream << "conv" << this->get_dimensionality() << "d_";
    write_s5d(stream, this->x_shape);
    write_s5d(stream, this->x_strides);
    write_s5d(stream, this->w_shape);
    write_s5d(stream, this->y_shape);
    write_s5d(stream, this->y_strides);
    stream << ctype;
    return stream.str();
}

bool ConvParams::operator<(const ConvParams& other) const
{
    for (int i = 0; i < 3; ++i)
    {
        if (pads[i] < other.pads[i])
            return true;
        if (pads[i] > other.pads[i])
            return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (strides[i] < other.strides[i])
            return true;
        if (strides[i] > other.strides[i])
            return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (dilation[i] < other.dilation[i])
            return true;
        if (dilation[i] > other.dilation[i])
            return false;
    }
    return groups < other.groups;
}

bool ConvParams::operator==(const ConvParams& other) const
{
    for (int i = 0; i < 3; ++i)
        if (pads[i] != other.pads[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (strides[i] != other.strides[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (dilation[i] != other.dilation[i])
            return false;
    return groups == other.groups;
}

bool ConvDescriptor::operator<(const ConvDescriptor& other) const
{
    if (type < other.type)
        return true;
    if (type == other.type)
    {
        if (params < other.params)
            return true;
        if (params == other.params)
        {
            if (x_shape < other.x_shape)
                return true;
            if (x_shape == other.x_shape)
            {
                if (x_strides < other.x_strides)
                    return true;
                if (x_strides == other.x_strides)
                {
                    if (w_shape < other.w_shape)
                        return true;
                    if (w_shape == other.w_shape)
                    {
                        if (y_shape < other.y_shape)
                            return true;
                        if (y_shape == other.y_shape)
                            return y_strides < other.y_strides;
                    }
                }
            }
        }
    }
    return false;
}

dace_state BackendDaCe::try_load(const std::string& hash)
{
    dace_state result{nullptr, nullptr, nullptr};
    const std::string& path = m_daceopts.m_cachepath;
    if (path.size() == 0)
    {
        util::MPIPrintStreamError()
            << "ERROR finding build path. Please "
            << "set the DACEDCONV_CACHEPATH environment variable.";
        std::abort();
    }

    std::stringstream fname, initname, funcname;
    fname << path << "/.dacecache/" << hash << "/build/lib" << hash << ".so";
    initname << "__dace_init_" << hash;
    funcname << "__program_" << hash;
    void* handle = dlopen(fname.str().c_str(), RTLD_LAZY);
    if (handle)
    {
        initfunc_t initfunc =
            (initfunc_t) dlsym(handle, initname.str().c_str());
        result.library = handle;
        if (!initfunc)
        { // No initializer
            util::MPIPrintStreamError() << "Initializer not found.";
            std::abort();
        }
        dacehandle_t dacehandle = initfunc();
        result.handle = dacehandle;

        // Set the current stream
        setstream_t setstreamfunc =
            (setstream_t) dlsym(handle, "__dace_gpu_set_all_streams");
        if (!setstreamfunc)
        { // No external stream setter
            util::MPIPrintStreamError()
                << "Set stream function not found, please regenerate the code.";
            std::abort();
        }
        if (m_curstream)
            setstreamfunc(dacehandle, m_curstream);
        else
            setstreamfunc(dacehandle, m_stream);

        result.func = (daceprogram_t) dlsym(handle, funcname.str().c_str());
    }
    return result;
}

bool BackendDaCe::unload(const std::string& hash, dace_state library)
{
    std::stringstream exitname;
    exitname << "__dace_exit_" << hash;
    exitfunc_t exitfunc =
        (exitfunc_t) dlsym(library.library, exitname.str().c_str());
    // Exit function not found
    if (!exitfunc)
    {
        dlclose(library.library);
        return false;
    }
    exitfunc(library.handle);
    if (dlclose(library.library))
        return false;
    return true;
}

void BackendDaCe::set_stream(backend::Handle_t handle, backend::Stream_t stream)
{
    this->m_curstream = stream;
}

dace_state BackendDaCe::compile(const ConvDescriptor& desc, const std::string& hash)
{
    dace_state result{nullptr, nullptr, nullptr};
    std::string script =
        std::string("python3 -m lbann.jit.generate_conv ") + hash;
    int res = system(script.c_str());
    if (res)
    {
        util::MPIPrintStreamError()
            << "Error running JIT compiler script: " << res;
        // Do not rerun
        return result;
    }

    // Try loading library after compiling
    result = try_load(hash);
    if (!result.func)
    {
        util::MPIPrintStreamError()
            << "Could not load JIT-compiled library after compilation: "
            << dlerror();
        std::abort();
    }
    return result;
}

bool BackendDaCe::invoke(const ConvDescriptor& desc,
                         void const* x,
                         void const* w,
                         void const* y,
                         float alpha,
                         float beta,
                         void* workspace)
{
    dace_state library;
    auto iter = m_dace_libraries.find(desc);

    // Need to JIT compile/load
    if (iter == m_dace_libraries.end())
    {
        std::string hash = desc.hash();
        library = try_load(hash);
        // Library does not already exist, compile and reload
        if (!library.library)
            library = compile(desc, hash);
        m_dace_libraries[desc] = library;
    }
    else
    {
        library = iter->second;
    }

    if (!library.library)
        return false; // Fall back to original implementation
    library.func(library.handle, w, x, y, alpha, beta);
    return true;
}

} // namespace distconv
