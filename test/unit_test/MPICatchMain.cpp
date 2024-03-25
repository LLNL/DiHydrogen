////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_session.hpp>
#include <catch2/internal/catch_clara.hpp>
using Catch::Clara::Opt;

#include <unistd.h>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <El.hpp>
#include <h2_config.hpp>

#ifdef H2_HAS_GPU
#include <h2/gpu/runtime.hpp>
#endif

#include "mpi_utils.hpp"


/**
 * Replace a filename with a modified per-MPI rank version.
 *
 * Replaces "foo" with "foo.mpirank.mpisize" and replaces "foo.ext"
 * with "foo.mpirank.mpisize.ext". Anything after the last "." is
 * considered to be the extension. If mpisize is 1 or filename is
 * empty, the original string is returned.
 */
std::string edit_output_filename(std::string_view filename,
                                 int mpirank,
                                 int mpisize)
{
  if (mpisize == 0 || filename.size() == 1UL)
  {
    return std::string{filename};
  }

  const auto split = filename.find_last_of('.');
  std::ostringstream oss;
  oss << filename.substr(0, split) << '.' << mpirank << '.' << mpisize;
  if (split != std::string_view::npos)
  {
    oss << '.' << filename.substr(split+1, std::string_view::npos);
  }
  return oss.str();
}

// Note: We must initialize GPU support if we were built with it.
// Aluminum will be expecting it.

struct TestEnvironment
{
  TestEnvironment(int& argc, char**& argv)
  {
#ifdef H2_HAS_GPU
    El::gpu::Initialize();
    h2::gpu::init_runtime();
#endif
    El::mpi::InitializeThread(argc, argv, El::mpi::THREAD_MULTIPLE);
  }

  ~TestEnvironment()
  {
    El::mpi::Finalize();
#ifdef H2_HAS_GPU
    h2::gpu::finalize_runtime();
    El::gpu::Finalize();
#endif
  }
};

namespace
{
CommManager* comm_manager = nullptr;
}

h2::Comm& get_comm(int size)
{
  H2_ASSERT_ALWAYS(comm_manager != nullptr, "CommManager not initialized");
  return comm_manager->get_comm(size);
}

h2::Comm& get_comm_or_skip(int size)
{
  H2_ASSERT_ALWAYS(comm_manager != nullptr, "CommManager not initialized");
  try
  {
    return comm_manager->get_comm(size);
  }
  catch (const internal::NotParticipatingException&)
  {
    SKIP();
    throw;
  }
}

int main(int argc, char** argv)
{
  // Initialize Catch2.
  // This is done first to avoid any initialization if the user asks
  // for help.
  Catch::Session session;

  int hang_rank = -1;
  auto cli =
    session.cli() | Opt(hang_rank, "Rank to hang")["--hang-rank"](
                      "Hang this rank to attach a debugger.");
  session.cli(cli);

  // Parse the command line. Also exit if the user asks for help.
  {
    const int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0 || session.configData().showHelp)
    {
      return return_code;
    }
  }

  // Set up the basic test environment.
  TestEnvironment env(argc, argv);
  comm_manager = new CommManager();

  int rank = El::mpi::COMM_WORLD.Rank();
  int size = El::mpi::COMM_WORLD.Size();

  // Handle a debugger hang.
  if (rank == hang_rank)
  {
    char hostname[1024];
    gethostname(hostname, 1024);
    std::cerr << "[hang]: (hostname: " << hostname << ", pid: " << getpid()
              << ")" << std::endl;
    int volatile wait = 1;
    while (wait) {}
  }
  MPI_Barrier(MPI_COMM_WORLD);  // This should hang the other ranks.

  // Manipulate output file(s) if needed.
  auto& output_file = session.configData().defaultOutputFilename;
  if (output_file.size() > 0)
  {
    output_file = edit_output_filename(output_file, rank, size);
  }

  // Handle reporter-specific output files.
  for (auto& spec : session.configData().reporterSpecifications)
  {
    const auto& outfile = spec.outputFile();
    if (outfile.some())
    {
      spec = Catch::ReporterSpec(spec.name(),
                                 edit_output_filename(*outfile, rank, size),
                                 spec.colourMode(),
                                 spec.customOptions());
    }
  }

  // Run the Catch tests, outputting to the given file.
  const int num_failed = session.run();

  // Clean up.
  delete comm_manager;
  comm_manager = nullptr;
  return num_failed;
}
