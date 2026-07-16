#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kThreads = 256;
constexpr int kGroupSize = 64;

template <typename scalar_t>
__global__ void nf4_linear_kernel(
    const scalar_t* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ output,
    int rows,
    int columns,
    int row_offset,
    int row_count,
    int samples) {
  const int local_row = blockIdx.x;
  const int sample = blockIdx.y;
  const int row = row_offset + local_row;
  if (local_row >= row_count || sample >= samples) return;

  float sum = 0.0f;
  const int base = row * columns;
  const scalar_t* in = input + sample * columns;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    const int index = base + column;
    const uint8_t byte = packed[index >> 1];
    const int code = ((index & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
    sum += static_cast<float>(in[column]) * static_cast<float>(scales[index / kGroupSize]) * static_cast<float>(code);
  }
  __shared__ float reduction[kThreads];
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[sample * row_count + local_row] = static_cast<scalar_t>(reduction[0]);
}

template <typename scalar_t>
__global__ void nf4_embedding_kernel(
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ packed,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ output,
    int columns,
    int count) {
  const int index_position = blockIdx.x;
  const int column = blockIdx.y * blockDim.x + threadIdx.x;
  if (index_position >= count || column >= columns) return;
  const int64_t row = indices[index_position];
  const int64_t offset = row * columns + column;
  const uint8_t byte = packed[offset >> 1];
  const int code = ((offset & 1) ? (byte >> 4) : (byte & 0x0f)) - 8;
  output[index_position * columns + column] = static_cast<scalar_t>(static_cast<float>(scales[offset / kGroupSize]) * static_cast<float>(code));
}

}  // namespace

torch::Tensor nf4_linear(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t columns,
    int64_t row_offset,
    int64_t row_count) {
  TORCH_CHECK(input.is_cuda() && packed.is_cuda() && scales.is_cuda(), "direct NF4 is CUDA-only");
  TORCH_CHECK(input.dim() == 2, "direct NF4 linear expects flattened [samples, columns] input");
  TORCH_CHECK(input.size(1) == columns, "direct NF4 linear column mismatch");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16, "direct NF4 supports fp16 or bf16 activations only");
  TORCH_CHECK(scales.scalar_type() == input.scalar_type(), "direct NF4 scales and activations must have the same dtype");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "direct NF4 packed weights must be uint8");
  TORCH_CHECK(row_offset >= 0 && row_count > 0 && row_offset + row_count <= rows, "direct NF4 row range is invalid");
  TORCH_CHECK(packed.numel() * 2 >= rows * columns, "direct NF4 packed weight is truncated");
  TORCH_CHECK(scales.numel() >= (rows * columns + kGroupSize - 1) / kGroupSize, "direct NF4 scales are truncated");

  auto contiguous_input = input.contiguous();
  auto output = torch::empty({contiguous_input.size(0), row_count}, contiguous_input.options());
  const dim3 blocks(row_count, contiguous_input.size(0));
  const auto stream = at::cuda::getDefaultCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, contiguous_input.scalar_type(), "nf4_linear_cuda", [&] {
    nf4_linear_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        contiguous_input.data_ptr<scalar_t>(), packed.data_ptr<uint8_t>(), scales.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), rows, columns, row_offset, row_count, contiguous_input.size(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor nf4_embedding(
    torch::Tensor indices,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t columns) {
  TORCH_CHECK(indices.is_cuda() && packed.is_cuda() && scales.is_cuda(), "direct NF4 is CUDA-only");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "direct NF4 embedding indices must be int64");
  TORCH_CHECK(scales.scalar_type() == at::kHalf || scales.scalar_type() == at::kBFloat16, "direct NF4 embedding scales must be fp16 or bf16");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "direct NF4 packed weights must be uint8");
  auto contiguous_indices = indices.contiguous();
  auto output = torch::empty({contiguous_indices.numel(), columns}, scales.options());
  const dim3 blocks(contiguous_indices.numel(), (columns + kThreads - 1) / kThreads);
  const auto stream = at::cuda::getDefaultCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scales.scalar_type(), "nf4_embedding_cuda", [&] {
    nf4_embedding_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        contiguous_indices.data_ptr<int64_t>(), packed.data_ptr<uint8_t>(), scales.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), columns, contiguous_indices.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("nf4_linear", &nf4_linear, "Packed NF4 linear (CUDA)");
  module.def("nf4_embedding", &nf4_embedding, "Packed NF4 embedding (CUDA)");
}

