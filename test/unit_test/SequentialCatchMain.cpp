////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_session.hpp>

#include <El.hpp>


int main(int argc, char** argv)
{
#ifdef HYDROGEN_HAVE_GPU
  El::gpu::Initialize();
#endif
  int result = Catch::Session().run(argc, argv);
#ifdef HYDROGEN_HAVE_GPU
  El::gpu::Finalize();
#endif
  return result;
}
