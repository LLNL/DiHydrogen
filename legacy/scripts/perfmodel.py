from math import sqrt, log2, log
import pickle
import subprocess
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Inter-node performance.
inter_perf = {
  'alpha': 1.2e-6,  # Latency, s
  'beta': 3.893e-11,  # Inverse bandwidth, B/s
  'gamma': 1.075e-13  # Computation time, single-precision FLOPS
}
# Intra-node performance.
intra_perf = {
  'alpha': 8e-6,  # Latency, s
  'beta': 2.661e-11,  # Inverse bandwidth, B/s
  'gamma': 1.075e-13  # Computation time, single-precision FLOPS
}
large_message_thresh = 2**9  # When to switch to large-message algorithms
word_size = 4  # Size of a word in bytes
ppn = 2  # Number of processes per node (used for intranode performance)
cudnn_benchmark = './cudnn_benchmark'

# Hack to be able to pickle conv_data.
def dd_d(): return defaultdict(dict)
def dd1(): return defaultdict(dd_d)
def dd2(): return defaultdict(dd1)
def dd3(): return defaultdict(dd2)
def dd4(): return defaultdict(dd3)
conv_data = defaultdict(dd4)

def run_configuration(n, c, h, w, f, k):
  r = subprocess.run([cudnn_benchmark, '-n', str(int(n)), '-c', str(int(c)),
                      '-h', str(int(h)), '-w', str(int(w)), '-m', str(int(f)),
                      '-s', str(int(k)), '-t', str(int(k))],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,
                     check=True)
  # Assume five runs.
  results = r.stdout.split(b'\n')[-7:-2]
  fwds, bwdds, bwdfs = [], [], []
  for line in results:
    # Convert from milliseconds.
    fwd, bwdd, bwdf = [float(x) / 1000 for x in line.split()[-3:]]
    fwds.append(fwd)
    bwdds.append(bwdd)
    bwdfs.append(bwdf)
  conv_data[n][c][h][w][f][k]['fwd'] = np.mean(fwds)
  conv_data[n][c][h][w][f][k]['bwdd'] = np.mean(bwdds)
  conv_data[n][c][h][w][f][k]['bwdf'] = np.mean(bwdfs)

def load_conv_data():
  global conv_data
  try:
    with open('conv_perf_cache', 'rb') as f:
      conv_data = pickle.load(f)
  except: pass

def save_conv_data():
  with open('conv_perf_cache', 'wb') as f:
    pickle.dump(conv_data, f)

def conv_forward_time(n, c, h, w, f, k):
  try:
    return conv_data[n][c][h][w][f][k]['fwd']
  except KeyError:
    run_configuration(n, c, h, w, f, k)
    return conv_data[n][c][h][w][f][k]['fwd']

def conv_backward_filter_time(n, c, h, w, f, k):
  try:
    return conv_data[n][c][h][w][f][k]['bwdf']
  except KeyError:
    run_configuration(n, c, h, w, f, k)
    return conv_data[n][c][h][w][f][k]['bwdf']

def conv_backward_data_time(n, c, h, w, f, k):
  try:
    return conv_data[n][c][h][w][f][k]['bwdd']
  except KeyError:
    run_configuration(n, c, h, w, f, k)
    return conv_data[n][c][h][w][f][k]['bwdd']

def allreduce_time(p, n, alpha, beta, gamma):
  if p == 1:
    return 0.0
  if n <= large_message_thresh:
    return log2(p)*alpha + n*word_size*log2(p)*beta + n*log2(p)*gamma
  else:
    return 2*log2(p)*alpha + 2*((p-1)/p)*n*word_size*beta + ((p-1)/p)*n*gamma

def allgather_time(p, n, alpha, beta, gamma):
  if p == 1:
    return 0.0
  return log2(p)*alpha + ((p-1)/p)*n*word_size*beta

def send_recv_time(n, alpha, beta, gamma):
  return alpha + n*word_size*beta

def sample_parallel_forward_time(p, n, c, h, w, f, k):
  p = min(p, n)
  return conv_forward_time(int(n / p), c, h, w, f, k)

def sample_parallel_backward_time(p, n, c, h, w, f, k):
  p = min(p, n)
  return (conv_backward_filter_time(int(n / p), c, h, w, f, k) +
          conv_backward_data_time(int(n / p), c, h, w, f, k))

def sample_parallel_opt_time(p, n, c, h, w, f, k, perf):
  return allreduce_time(p, c*f*k*k, **perf)

def channel_parallel_forward_time(p, n, c, h, w, f, k, perf):
  p = min(p, c)
  return (conv_forward_time(n, int(c / p), h, w, f, k) +
          allreduce_time(p, n*f*h*w, **perf))

def channel_parallel_backward_time(p, n, c, h, w, f, k, perf, avoid_ag=False):
  p = min(p, c)
  return (conv_backward_filter_time(n, int(c / p), h, w, f, k) +
          conv_backward_data_time(n, int(c / p), h, w, f, k) +
          (0 if avoid_ag else allgather_time(p, n*c*h*w, **perf)))

def channel_parallel_opt_time(p, n, c, h, w, f, k):
  return 0.0

def filter_parallel_forward_time(p, n, c, h, w, f, k, perf, avoid_ag=False):
  p = min(p, f)
  return (conv_forward_time(n, c, h, w, int(f / p), k) +
          (0 if avoid_ag else allgather_time(p, n*f*h*w, **perf)))

def filter_parallel_backward_time(p, n, c, h, w, f, k, perf):
  p = min(p, f)
  return (conv_backward_filter_time(n, c, h, w, int(f / p), k) +
          conv_backward_data_time(n, c, h, w, int(f / p), k) +
          allreduce_time(p, n*c*h*w, **perf))

def filter_parallel_opt_time(p, n, c, h, w, f, k):
  return 0.0

def spatial_parallel_forward_time(p, n, c, h, w, f, k, perf):
  h_ = max(1, h / int(sqrt(p)))
  w_ = max(1, w / int(sqrt(p)))
  return (conv_forward_time(n, c, h_, w_, f, k) +
          (0.0 if p == 1 else (
            send_recv_time(c*(k - 1)*h_, **perf) +
            send_recv_time(c*(k - 1)*w_, **perf) +
            send_recv_time(c*(k - 1)**2, **perf))))

def spatial_parallel_backward_time(p, n, c, h, w, f, k, perf):
  h_ = max(1, h / int(sqrt(p)))
  w_ = max(1, w / int(sqrt(p)))
  return (conv_backward_filter_time(n, c, h_, w_, f, k) +
          conv_backward_data_time(n, c, h_, w_, f, k) +
          (0.0 if p == 1 else (
            send_recv_time(c*(k - 1)*h_, **perf) +
            send_recv_time(c*(k - 1)*w_, **perf) +
            send_recv_time(c*(k - 1)**2, **perf))))

def spatial_parallel_opt_time(p, n, c, h, w, f, k, perf):
  return allreduce_time(p, c*f*k*k, **perf)

def plot_config_perf(max_procs, n, c, h, w, f, k):
  procs_by_2 = [2**x for x in range(int(log2(max_procs)) + 1)]
  # Need to be able to split into rectangles.
  procs_by_4 = [4**x for x in range(int(log(max_procs, 4)) + 1)]
  sample_fwd_times, channel_fwd_times, filter_fwd_times, spatial_fwd_times = [], [], [], []
  sample_bwd_times, channel_bwd_times, filter_bwd_times, spatial_bwd_times = [], [], [], []
  sample_opt_times, channel_opt_times, filter_opt_times, spatial_opt_times = [], [], [], []
  filter_fwd_noag_times, channel_bwd_noag_times = [], []
  for p in procs_by_2:
    if p <= ppn:
      perf = intra_perf
    else:
      perf = inter_perf
    sample_fwd_times.append(sample_parallel_forward_time(p, n, c, h, w, f, k))
    channel_fwd_times.append(channel_parallel_forward_time(p, n, c, h, w, f, k, perf))
    filter_fwd_times.append(filter_parallel_forward_time(p, n, c, h, w, f, k, perf))
    filter_fwd_noag_times.append(filter_parallel_forward_time(p, n, c, h, w, f, k, perf, avoid_ag=True))
    sample_bwd_times.append(sample_parallel_backward_time(p, n, c, h, w, f, k))
    channel_bwd_times.append(channel_parallel_backward_time(p, n, c, h, w, f, k, perf))
    filter_bwd_times.append(filter_parallel_backward_time(p, n, c, h, w, f, k, perf))
    channel_bwd_noag_times.append(channel_parallel_backward_time(p, n, c, h, w, f, k, perf, avoid_ag=True))
    sample_opt_times.append(sample_parallel_opt_time(p, n, c, h, w, f, k, perf))
    channel_opt_times.append(channel_parallel_opt_time(p, n, c, h, w, f, k))
    filter_opt_times.append(filter_parallel_opt_time(p, n, c, h, w, f, k))
  for p in procs_by_4:
    if p <= ppn:
      perf = intra_perf
    else:
      perf = inter_perf
    spatial_fwd_times.append(spatial_parallel_forward_time(p, n, c, h, w, f, k, perf))
    spatial_bwd_times.append(spatial_parallel_backward_time(p, n, c, h, w, f, k, perf))
    spatial_opt_times.append(spatial_parallel_opt_time(p, n, c, h, w, f, k, perf))
  sample_time = np.array(sample_fwd_times) + np.array(sample_bwd_times) + np.array(sample_opt_times)
  channel_time = np.array(channel_fwd_times) + np.array(channel_bwd_times) + np.array(channel_opt_times)
  channel_noag_time = np.array(channel_fwd_times) + np.array(channel_bwd_noag_times) + np.array(channel_opt_times)
  filter_time = np.array(filter_fwd_times) + np.array(filter_bwd_times) + np.array(filter_opt_times)
  filter_noag_time = np.array(filter_fwd_noag_times) + np.array(filter_bwd_times) + np.array(filter_opt_times)
  spatial_time = np.array(spatial_fwd_times) + np.array(spatial_bwd_times) + np.array(spatial_opt_times)
  fig, ax = plt.subplots(1, 1)
  ax.loglog(procs_by_2, sample_time, '.',
            procs_by_2, channel_time, '^',
            procs_by_2, channel_noag_time, 'v',
            procs_by_2, filter_time, '<',
            procs_by_2, filter_noag_time, '>',
            procs_by_4, spatial_time, '*',
            basex=2, basey=10)
  ax.set_xlabel('# Processes')
  ax.set_ylabel('Time (s)')
  ax.set_title('n={0} c={1} h={2} w={3} f={4} k={5}'.format(n, c, h, w, f, k))
  ax.legend(('Sample', 'Channel', 'Channel !AG', 'Filter', 'Filter !AG', 'Spatial'))
  plt.tight_layout()
  plt.savefig('n{0}_c{1}_h{2}_w{3}_f{4}_k{5}.pdf'.format(n, c, h, w, f, k))

if __name__ == '__main__':
  load_conv_data()
  # Extreme cases.
  plot_config_perf(256, 1024, 1, 64, 64, 1, 3)
  plot_config_perf(256, 1, 1024, 64, 64, 1, 3)
  plot_config_perf(256, 1, 1, 64, 64, 1024, 3)
  plot_config_perf(256, 1, 1, 2048, 2048, 1, 3)
  # ResNet.
  plot_config_perf(64, 256, 3, 224, 224, 64, 7)
  plot_config_perf(64, 256, 64, 56, 56, 64, 3)
  plot_config_perf(64, 256, 64, 56, 56, 128, 3)
  plot_config_perf(64, 256, 128, 28, 28, 128, 3)
  plot_config_perf(64, 256, 128, 28, 28, 256, 3)
  plot_config_perf(64, 256, 256, 14, 14, 256, 3)
  plot_config_perf(64, 256, 256, 14, 14, 512, 3)
  plot_config_perf(64, 256, 512, 7, 7, 512, 3)
  # Mesh.
  plot_config_perf(64, 1, 1, 2048, 2048, 64, 5)
  plot_config_perf(64, 1, 1, 2048, 2048, 64, 3)
  plot_config_perf(64, 1, 1, 1024, 1024, 64, 3)
  plot_config_perf(64, 1, 1, 512, 512, 64, 3)
  plot_config_perf(64, 1, 1, 256, 256, 64, 3)
  plot_config_perf(64, 1, 1, 128, 128, 64, 3)
  plot_config_perf(64, 1, 1, 64, 64, 64, 3)
  plot_config_perf(64, 1, 1, 32, 32, 64, 3)
  save_conv_data()
