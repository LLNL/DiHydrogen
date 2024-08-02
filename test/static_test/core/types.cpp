////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/core/types.hpp"


using namespace h2;


// Storage types:

static_assert(IsH2StorageType_v<int>);
static_assert(IsH2StorageType_v<float>);
static_assert(IsH2StorageType_v<bool>);
static_assert(!IsH2StorageType_v<int*>);

struct Storable { int foo; };
static_assert(IsH2StorageType_v<Storable>);
static_assert(!IsH2StorageType_v<Storable*>);

struct NotStorable { virtual void foo(); };
static_assert(!IsH2StorageType_v<NotStorable>);
static_assert(!IsH2StorageType_v<NotStorable*>);

// Compute types:

static_assert(IsH2ComputeType_v<float>);
static_assert(IsH2ComputeType_v<double>);
static_assert(IsH2ComputeType_v<std::int32_t>);
static_assert(IsH2ComputeType_v<std::uint32_t>);
static_assert(!IsH2ComputeType_v<Storable>);
static_assert(!IsH2ComputeType_v<NotStorable>);
