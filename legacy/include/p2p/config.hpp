#pragma once

namespace p2p
{
namespace internal
{

struct Config
{
  bool insert_nvtx_mark = false;
};

extern Config cfg;

}  // namespace internal
}  // namespace p2p
