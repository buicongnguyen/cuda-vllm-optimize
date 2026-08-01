# Kỹ năng cần làm chủ

## Thứ tự ưu tiên

### 1. Performance methodology và statistics — ưu tiên cao nhất

Phải làm chủ trước CUDA nâng cao:

- metric semantics: TTFT, TPOT, ITL, E2E, queueing;
- controlled experiments, A/B/A, warm-up, randomized order;
- confidence interval, variance, outliers, drift và noise floor;
- experiment ledger, artifact provenance, one-change-at-a-time;
- roofline model nhưng biết phân biệt estimate với measurement.

Bài tập đạt chuẩn: cùng một artifact chạy 10 replay, giải thích variance theo
batch/turn và phát hiện một regression 1% với false-positive rate hợp lý.

### 2. vLLM internals và serving systems

- V1 engine: request lifecycle, scheduler, continuous batching;
- KV cache manager cho attention và hybrid/SSM states;
- prefix caching, block allocation, eviction;
- chunked prefill và decode scheduling;
- CUDA Graph/torch.compile/custom-op dispatch;
- tokenizer/detokenizer và OpenAI streaming frontend.

Bài tập: trace một request từ HTTP đến từng model step, rồi chú thích thread,
queue, CPU/GPU boundary và cache state.

### 3. GPU performance engineering

- GPU hierarchy: SM, warp, registers, shared memory, L2, HBM;
- occupancy, coalescing, vectorization, bank conflicts;
- launch overhead, streams, events, synchronization, CUDA Graphs;
- Tensor Cores/FP8 formats, scaling và numerical error;
- Nsight Systems và Nsight Compute;
- khác biệt SM89 vs SM90 và MIG resource isolation.

Bài tập: profile cùng operator trên L4 và H200 MIG, dự đoán rồi giải thích vì sao
block size/occupancy tối ưu khác nhau.

### 4. Triton/CUDA kernel engineering

- Triton program model, masks, layouts, autotuning;
- stateful/in-place correctness và aliasing;
- fused reductions (RMSNorm), RoPE, activation/quantization;
- register pressure versus saved HBM traffic;
- PyTorch custom ops, fake tensors, compile compatibility;
- property tests, reference implementation, adversarial shapes/dtypes.

Bài tập: kernel phải pass exact/tolerance contract, benchmark shape matrix và
thắng end-to-end — không chỉ microbenchmark.

### 5. LLM architecture và numerical inference

- Transformer/GQA attention, KV cache math;
- convolution/SSM recurrent state và sequence boundaries;
- prefill vs decode computational shapes;
- weight/KV quantization, calibration, accumulated error;
- speculative decoding correctness, verification và rollback semantics.

Bài tập: tự tính memory bytes cho weight/KV/conv state và đối chiếu profiler.

### 6. Linux/container/CPU performance

- Docker layers, image digest, ABI/binary extension compatibility;
- cgroups CPU/RAM, `/dev/shm`, page faults;
- multiprocessing, GIL/event loop, OpenMP pools;
- allocators, sockets, HTTP streaming, observability overhead;
- reproducible builds và runtime manifest.

Bài tập: tạo overlay fail-fast khi base digest/signature sai và bisect một
integration regression bằng artifact matrix.

## Lộ trình 12 tuần

### Tuần 1–2: đo đúng

Xây replay deterministic, raw event schema, ERS calculator, A/B/A và dashboard
latency theo turn/batch. Hoàn thiện repo này bằng evaluator spec chính thức.

### Tuần 3–4: đọc vLLM source

Trace scheduler/KV/frontend; thêm instrumentation tối thiểu; chạy config ablations
trên một model nhỏ nhưng giữ workload thật.

### Tuần 5–6: Nsight và CUDA Graph

Học timeline, CPU gaps, graph capture, stream synchronization; giải thích critical
path bằng screenshots/counters thay vì estimate.

### Tuần 7–9: Triton correctness rồi performance

Viết RMSNorm/RoPE hoặc ShortConv toy kernel, property tests, autotune theo SM;
đo register/occupancy/bandwidth. Sau đó mới tích hợp custom op.

### Tuần 10: quantization

FP8 formats/scales, W8A8 và KV FP8; đo quality/performance/memory như ba metric
riêng biệt.

### Tuần 11: hybrid state và speculation

Mô phỏng accept/reject, snapshot/rollback conv state, multi-cache-group metadata;
đọc issue/PR và viết failure reproduction nhỏ.

### Tuần 12: race simulation

Giới hạn 5 “portal submissions” giả/ngày. Chấm không chỉ best ERS mà cả số điểm
information gain trên mỗi submission và khả năng giải thích mọi regression.

## Tiêu chí “master”

Bạn chưa master một optimization khi chỉ biết bật flag hoặc kernel benchmark
nhanh hơn. Master nghĩa là có thể:

1. dự đoán khi nào nó giúp hoặc hại;
2. chứng minh code path được dùng;
3. xác nhận numerical/semantic correctness;
4. đo end-to-end trên target và định lượng uncertainty;
5. rollback/bisect khi integration regression;
6. giải thích kết quả bằng counter/timeline có thể tái lập.
