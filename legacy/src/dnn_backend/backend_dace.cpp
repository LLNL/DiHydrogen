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
typedef void* (*initfunc_t)();
typedef void (*exitfunc_t)(void* handle);
typedef bool (*setstream_t)(void* handle, backend::Stream_t stream);
typedef void (*daceprogram_fwd_t)(void* handle, float* w, float* x, float* y);
typedef void (*daceprogram_bwdfilt_t)(void* handle,
                                      float* dw,
                                      float* dy,
                                      float* x);
typedef void (*daceprogram_bwddata_t)(void* handle,
                                      float* dx,
                                      float* dy,
                                      float* w);

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

state_and_func BackendDaCe::try_load(const std::string& hash)
{
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
        if (!initfunc)
        { // No initializer
            util::MPIPrintStreamError() << "Initializer not found.";
            std::abort();
        }
        void* dacehandle = initfunc();

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

        void* func = dlsym(handle, funcname.str().c_str());
        if (func)
            return std::make_tuple(dacehandle, func);
    }
    return std::make_tuple(nullptr, nullptr);
}

bool BackendDaCe::unload(const std::string& hash, state_and_func library)
{
    std::stringstream exitname;
    exitname << "__dace_exit_" << hash;
    exitfunc_t exitfunc =
        (exitfunc_t) dlsym(std::get<0>(library), exitname.str().c_str());
    if (!exitfunc) // Exit function not found
        return false;
    exitfunc(std::get<1>(library));
    return true;
}

void BackendDaCe::set_stream(backend::Handle_t handle, backend::Stream_t stream)
{
    this->m_curstream = stream;
}

} // namespace distconv
