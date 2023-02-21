////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2/utils/Logger.hpp>

#include <iostream>
#include <cstdlib>

void check_should_log(h2::Logger* logger);

struct NonExistentLoggerPolicy
{
  template <typename T>
  void handle(T val)
  {
    std::cout << "Logger does not exist" << std::endl;
  }
};

template <class T>
class Container
{
  void do_stuff()
  {
    std::cout << "Container does stuff" << std::endl;
  }
};

#define IGNORE_EXCEPTION(cmd, excpt) \
  do { \
    try { (cmd); } \
    catch (excpt) {} \
  } while(0)

int main(int, char*[])
{
  h2::Logger NotSetup("logger");
    auto not_log = NotSetup.get();
    not_log.trace("testing not setup logger");

    putenv("TEST_LOG_MASK=trace|debug|info|warn|error|critical, training=critical|error, special_logger=debug|info|warn");
    putenv("TEST_LOG_LEVEL=critical, io=debug");
    NonExistentLoggerPolicy policy;
    Container<h2::Logger> container;

    h2::Logger global_logger("global_logger");
    h2::Logger io_logger("io");
    h2::Logger training_logger("training");
    h2::Logger special_logger("special_logger");
    std::vector<h2::Logger*> loggers;
    loggers.push_back(&global_logger);
    loggers.push_back(&io_logger);
    loggers.push_back(&training_logger);
    loggers.push_back(&special_logger);

    const char* TEST_LOG_LEVEL = "TEST_LOG_LEVEL";
    const char* TEST_LOG_MASK = "TEST_LOG_MASK";

    h2::setup_levels(loggers, TEST_LOG_LEVEL);
/*
    try
    {
      h2::setup_levels_and_masks(loggers, TEST_LOG_LEVEL, TEST_LOG_MASK);
    }
    catch( std::string name )
    {
      std::cout << "Caught " << name << std::endl;
    }
*/
    //IGNORE_EXCEPTION(h2::setup_levels_and_masks(loggers, TEST_LOG_LEVEL, TEST_LOG_MASK), std::string unknown);

  for( auto& logger : loggers )
  {
    //logger.load_levels("TEST_LOG_LEVEL");
    check_should_log(logger);
    auto log = logger->get();
    log.trace("logged trace message");
    log.debug("logged debug message");
  }

  /*
    check_should_log(io_logger);
    log = io_logger.get();
    log.trace("logged trace message");
    log.debug("logged debug message");

    check_should_log(training_logger);
    log = training_logger.get();
    log.trace("logged trace message");
    log.debug("logged debug message");

    check_should_log(special_logger);
    log = special_logger.get();
    log.trace("logged trace message");
    log.debug("logged debug message");
  */
    /*
    H2_TRACE("AAAALLLLL the info");

    H2_DEBUG("Don't show this one to users");

    H2_INFO("testing spdlog v{}.{}", 0, 1);

    H2_ERROR("You've encountered error {}", 32);

    H2_WARN("Easy padding in numbers like {:08d}", 12);

    H2_CRITICAL("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}",
                42);
    */
    return 0;
}

void check_should_log(h2::Logger* logger)
{
    if (logger->should_log(h2::Logger::LogLevelType::TRACE))
      std::cout << logger->name() << " Log level trace" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::DEBUG))
      std::cout << logger->name() << " Log level debug" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::INFO))
      std::cout << logger->name() << " Log level info" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::WARN))
      std::cout << logger->name() << " Log level warn" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::ERROR))
      std::cout << logger->name() << " Log level error" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::CRITICAL))
      std::cout << logger->name() << " Log level critical" << std::endl;
    if (logger->should_log(h2::Logger::LogLevelType::OFF))
      std::cout << logger->name() << " Log level off" << std::endl;
}
