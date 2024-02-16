////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_session.hpp>

#include <El.hpp>
#include <h2/gpu/runtime.hpp>

struct GPUEnvironment
{
    GPUEnvironment()
    {
        El::gpu::Initialize();
        h2::gpu::init_runtime();
    }

    ~GPUEnvironment()
    {
        h2::gpu::finalize_runtime();
        El::gpu::Finalize();
    }
};

int main(int argc, char** argv)
{
    // Initialize and parse the Catch2 command line before any other
    // initialization.
    Catch::Session session;
    {
        const int return_code = session.applyCommandLine(argc, argv);
        if (return_code != 0 || session.configData().showHelp)
        {
            return return_code;
        }
    }

    // Initialize the GPU environment.
    GPUEnvironment env;

    return session.run();
}
