.. role:: bash(code)
          :language: bash

DiHydrogen CI
========================

DiHydrogen CI is a work in progress. This documentation will evolve
over time as a CI format is developed.


Running Tests
------------------------------

After building DiHydrogen, run the test suite using the
:code:`SeqCatchTests` executable::

  ./test/unit_test/SeqCatchTests

To run a subset of tests, run with the corresponding tag::

  ./test/unit_test/SeqCatchTests "[tag_name]"


Writing New Tests
-------------------------------

Unit Tests
~~~~~~~~~~

Add unit tests to DiHydrogen using the Catch2 framework. Documentation for
Catch2 testing can be found `here
<https://github.com/catchorg/Catch2/tree/v2.x>`_.

   1. Select the appropriate directory in :code:`tests/unit_tests/` or create a
      new directory for a new test category.

   2. Create a new test file in the selected test directory using the
      format :code:`unit_test_<test_name>.cpp`
      (i.e. :code:`unit_test_logging.cpp`).

   3. Modify :code:`CmakeLists.txt`:

      * Add any new directories to
        :code:`test/unit_tests/CMakeLists.txt` using
        :code:`add_subdirectory()`

      * Create or modify :code:`CMakeLists.txt` in the test directory
        to include::

          target_sources(SeqCatchTests PRIVATE
          unit_test_<test_name>.cpp
          )

        This will add the tests to SeqCatchTests. See `CMakeLists
        <https://github.com/LLNL/DiHydrogen/blob/develop/test/unit_test/patterns/factory/CMakeLists.txt>`_
        for an example.
