// CUDA IPC + peer copy engine for cross-process expert weight migration.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    TORCH_CHECK(e == cudaSuccess, "CUDA error at ", __FILE__, ":", __LINE__,    \
                " -> ", cudaGetErrorString(e));                                \
  } while (0)

namespace {
std::unordered_map<int, void *> g_peer_base;  // peer rank -> mapped base pointer
}

std::vector<uint8_t> get_ipc_handle(torch::Tensor pool) {
  TORCH_CHECK(pool.is_cuda(), "pool must be a CUDA tensor");
  TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
  TORCH_CHECK(pool.storage_offset() == 0,
              "pool must own its storage from offset 0; got ",
              pool.storage_offset());
  cudaIpcMemHandle_t h;
  CHECK(cudaIpcGetMemHandle(&h, pool.data_ptr()));
  const uint8_t *raw = reinterpret_cast<const uint8_t *>(&h);
  return std::vector<uint8_t>(raw, raw + sizeof(h));
}

void open_ipc_handle(std::vector<uint8_t> blob, int peer_rank) {
  TORCH_CHECK(blob.size() == sizeof(cudaIpcMemHandle_t), "bad handle size ",
              blob.size());
  if (g_peer_base.count(peer_rank)) return;
  cudaIpcMemHandle_t h;
  std::memcpy(&h, blob.data(), sizeof(h));
  void *ptr = nullptr;
  CHECK(cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
  g_peer_base[peer_rank] = ptr;
}

void close_ipc_handles() {
  for (auto &kv : g_peer_base) cudaIpcCloseMemHandle(kv.second);
  g_peer_base.clear();
}

int64_t peer_base_ptr(int peer_rank) {
  auto it = g_peer_base.find(peer_rank);
  TORCH_CHECK(it != g_peer_base.end(), "rank ", peer_rank, " not opened");
  return reinterpret_cast<int64_t>(it->second);
}

// Pull `nbytes` starting at slot boundaries from a peer's pool into ours over
// the copy engine. The handle maps the peer's allocation *base*, so source and
// destination addresses are computed here from the caller's slot indices.
void async_copy_expert_slot(torch::Tensor local_pool, int64_t dst_slot,
                            int peer_rank, int64_t src_slot, int64_t nbytes,
                            int64_t stream_ptr) {
  TORCH_CHECK(local_pool.is_cuda() && local_pool.is_contiguous());
  auto it = g_peer_base.find(peer_rank);
  TORCH_CHECK(it != g_peer_base.end(), "rank ", peer_rank, " not opened");
  TORCH_CHECK(nbytes > 0, "nbytes must be positive");

  int64_t slot_bytes = local_pool.nbytes() / local_pool.size(0);
  int64_t dst_off = dst_slot * slot_bytes;
  TORCH_CHECK(dst_off >= 0 && dst_off + nbytes <= local_pool.nbytes(),
              "copy of ", nbytes, " bytes at slot ", dst_slot,
              " overruns local pool of ", local_pool.nbytes(), " bytes");

  char *dst = static_cast<char *>(local_pool.data_ptr()) + dst_off;
  char *src = static_cast<char *>(it->second) + src_slot * slot_bytes;
  int dev;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaMemcpyPeerAsync(dst, dev, src, peer_rank, nbytes,
                            reinterpret_cast<cudaStream_t>(stream_ptr)));
}

namespace {
__global__ void commit_kernel(int32_t *log2phy, const int32_t *new_log2phy,
                              int32_t *logcnt, const int32_t *new_logcnt,
                              int64_t n_entries, int64_t n_experts) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_entries) log2phy[i] = new_log2phy[i];
  if (i < n_experts) logcnt[i] = new_logcnt[i];
}
}  // namespace

// Swap in a new routing table with a single launch. No atomics: kernels on one
// stream are ordered, so a replay enqueued afterwards cannot observe a partial
// table.
void commit_routing_table(torch::Tensor log2phy, torch::Tensor new_log2phy,
                          torch::Tensor logcnt, torch::Tensor new_logcnt,
                          int64_t stream_ptr) {
  TORCH_CHECK(log2phy.scalar_type() == torch::kInt32 &&
                  logcnt.scalar_type() == torch::kInt32,
              "routing tables must be int32");
  TORCH_CHECK(log2phy.sizes() == new_log2phy.sizes(), "log2phy shape mismatch");
  TORCH_CHECK(logcnt.sizes() == new_logcnt.sizes(), "logcnt shape mismatch");
  TORCH_CHECK(log2phy.is_contiguous() && new_log2phy.is_contiguous() &&
                  logcnt.is_contiguous() && new_logcnt.is_contiguous());

  int64_t n_entries = log2phy.numel(), n_experts = logcnt.numel();
  int64_t n = std::max(n_entries, n_experts);
  const int threads = 256;
  commit_kernel<<<(n + threads - 1) / threads, threads, 0,
                  reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
      log2phy.data_ptr<int32_t>(), new_log2phy.data_ptr<int32_t>(),
      logcnt.data_ptr<int32_t>(), new_logcnt.data_ptr<int32_t>(), n_entries,
      n_experts);
  CHECK(cudaGetLastError());
}

bool enable_peer_access(int peer_rank) {
  int can = 0, dev;
  CHECK(cudaGetDevice(&dev));
  if (dev == peer_rank) return true;
  CHECK(cudaDeviceCanAccessPeer(&can, dev, peer_rank));
  if (!can) return false;
  cudaError_t e = cudaDeviceEnablePeerAccess(peer_rank, 0);
  // Any failure leaves a sticky error that would surface in unrelated calls, so
  // clear it here. Already-enabled is not a failure.
  if (e != cudaSuccess) {
    cudaGetLastError();
    return e == cudaErrorPeerAccessAlreadyEnabled;
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_ipc_handle", &get_ipc_handle);
  m.def("open_ipc_handle", &open_ipc_handle);
  m.def("close_ipc_handles", &close_ipc_handles);
  m.def("peer_base_ptr", &peer_base_ptr);
  m.def("async_copy_expert_slot", &async_copy_expert_slot);
  m.def("commit_routing_table", &commit_routing_table);
  m.def("enable_peer_access", &enable_peer_access);
}
