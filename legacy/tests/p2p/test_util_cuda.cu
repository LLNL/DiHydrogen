namespace
{
__global__ void kernel(long long int sleep_cycles)
{
  auto start = clock64();
  while (true)
  {
    auto now = clock64();
    if (now - start > sleep_cycles)
    {
      break;
    }
  }
  return;
}
}  // namespace

void spin_device(cudaStream_t st, int giga_cycles)
{
  kernel<<<1, 1, 0, st>>>((long long int) (1) * 1000 * 1000 * 1000
                          * giga_cycles);
}
