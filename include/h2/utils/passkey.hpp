////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Provides an implementation of the "passkey" idiom.
 */

namespace h2
{

/**
 * A generic passkey.
 *
 * The "passkey" idiom allows a class to restrict access to methods at
 * a finer-grained level than friendship by requiring the caller to
 * pass an object (the passkey) that only certain classes can create.
 *
 * This provides a copyable passkey.
 *
 * This is especially useful for things like `std::make_unique` and
 * private constructors.
 *
 * This implementation is derived from the classic one provided here:
 * https://stackoverflow.com/questions/3217390/clean-c-granular-friend-equivalent-answer-attorney-client-idiom/3218920#3218920
 */
template <typename T>
class Passkey
{
private:
  friend T;

  Passkey() = default;
};

/**
 * Version of Passkey that supports two types.
 *
 * Whenever we switch to C++26, we can use variadic friends (P2893R0)
 * to eliminate this.
 */
template <typename T, typename U>
class Passkey2
{
private:
  friend T;
  friend U;

  Passkey2() = default;

public:
  Passkey2(Passkey<T>) {}
  Passkey2(Passkey<U>) {}
};

/** A non-copyable passkey. */
template <typename T>
class NonCopyablePasskey
{
private:
  friend T;

  NonCopyablePasskey() = default;
  NonCopyablePasskey(const NonCopyablePasskey&) = delete;
  NonCopyablePasskey& operator=(const NonCopyablePasskey&) = delete;
};

} // namespace h2
