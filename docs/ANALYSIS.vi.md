# Phân tích kỹ thuật và phản biện

## Phán quyết

Bài viết thể hiện khả năng học source nhanh, ý thức production và sự bền bỉ rất
tốt. Thành quả config tuning từ một baseline yếu lên khoảng 63 ERS, nếu số liệu
được tái lập, là kết quả đáng kể. Vấn đề chính không phải thiếu nỗ lực mà là quy
trình thực nghiệm chưa khóa biến số: metric chưa được kiểm toán, target hardware
đến muộn, baseline/version không được pin đủ chặt, và micro-kernel được viết trước
khi critical path được đo bằng profiler hệ thống.

Không nên diễn giải top 77/220 là “kernel fusion chỉ đem lại 1% nên vô nghĩa”.
Kết quả submission cuối giảm từ khoảng 63 xuống 51 cho thấy có confound hoặc
integration regression lớn hơn rất nhiều lợi ích kernel. Khi đó nhiệm vụ số một
là bisect regression, không phải viết kernel thứ tư.

## Những điểm làm tốt

### 1. Nhìn bài toán theo toàn stack

Tác giả đã đi qua model architecture, vLLM scheduler, CUDA/Triton kernels,
tokenizer, Docker và giới hạn CPU/RAM. Đó là đúng tầm nhìn cho inference serving:
điểm cuối cùng là kết quả của cả pipeline chứ không phải riêng GEMM.

### 2. Nhận ra khác biệt giữa GPU model và deployment shape

H200 `1g.18gb` chỉ có 1/7 SM và 1/8 memory/L2 resources theo tài liệu NVIDIA.
Vì vậy L4 (SM89), H100 full GPU và H200 MIG không phải các proxy tương đương.
Việc tự nhận ra sai lầm này là một bài học đúng và quan trọng.

### 3. Có correctness guard và fallback

Các custom path có dtype/mode guard và fallback là thiết kế production tốt.
48/48 test local cũng tốt hơn benchmark-only. Những bug về aliasing, RoPE và
rounding FP8 cho thấy tác giả thực sự kiểm tra numerical behavior.

### 4. Tối ưu vòng lặp build

Runtime overlay giúp iteration nhanh là một kỹ thuật hợp lý trong cuộc thi ngắn.
Giảm thời gian build từ phút xuống giây có giá trị gián tiếp lớn vì tăng số giả
thuyết có thể kiểm tra. Tuy nhiên overlay phải có guard về exact base digest và
ABI; nếu không, chính nó có thể tạo regression khó giải thích.

### 5. Công khai thất bại

Ghi lại negative results như `mamba-block-size`, layout, version, watermark và
spec decode là có ích. Repo này nâng chúng thành ledger có parent/config/evidence
để người khác phân biệt kết luận tổng quát với kết quả trên một image/MIG cụ thể.

## Những điểm cần sửa ngay trong nội dung

| Nội dung | Vấn đề | Cách viết đúng hơn |
|---|---|---|
| 10 attention + 6 ShortConv | Model card ghi 10 convolution + 6 GQA | Sửa layer count; tính lại ưu tiên kernel |
| 47 ms, 4 ms ≈ 42 ERS | Công thức trong bài cho 63.1851, gần đúng mốc “sau tuning ~63” | Xác nhận cặp latency bị gắn nhãn sai hay công thức/aggregation khác |
| GPU compute “thực sự” 0.017 ms | Đây là `FLOP / peak FLOPS`, chỉ là lower bound | Đo kernel duration/occupancy bằng Nsight; gọi đây là roofline estimate |
| Weight load 0.6 GB ở FP8 | 1.17B tham số FP8 đã xấp xỉ 1.17 GB, chưa kể scale/metadata | Giải thích byte count hoặc lấy DRAM counters |
| Spec decode hỏng chỉ vì rollback state | Issue #49112 cho thấy thêm lỗi multi-KV-group/metadata ở draft path | Mô tả failure theo từng version/PR và stack trace |
| Cần acceptance ≥90% | Không có ngưỡng phổ quát | Dùng break-even model theo draft/verify cost và accepted tokens |
| O3 luôn +2% | Current docs nói O3 hiện tương đương O2; version cũ có thể khác | Pin commit và báo A/B trên đúng image |
| fastokens là C++ | Current vLLM docs mô tả Rust backend, v0.23+ | Pin package/version và integration patch |
| shm 256 MB tránh OOM | shm limit thường không reserve toàn bộ ngay | Đo cgroup RAM và `/dev/shm`; ghi exception cụ thể |
| 35 / 21 / hơn 20 submissions | Ba con số trong bài không thống nhất | Xuất ledger và đếm chính xác accepted/crash/rejected |

Chi tiết máy đọc được nằm trong `data/claims.json`.

## Phân tích lại bottleneck

### Không được cộng các estimate như thể là timeline đo được

Bảng 0.017 + 0.88 + 0.5 + 0.3 + 1.6 + 0.7 ms trộn bốn loại đại lượng:

- lower bound từ peak FLOPS;
- roofline từ dung lượng/bandwidth lý thuyết;
- duration GPU kernels;
- wall-clock frontend/scheduler/network.

Các phần này có thể overlap, được amortize theo batch, hoặc đã nằm trong CUDA
Graph. Cộng chúng thành 100% tạo cảm giác chính xác giả. Với khoảng 70 sequence
đang decode, weight traffic được amortize trên cả batch; phép tính weight-load
“mỗi request” sẽ sai nếu không nêu mẫu số.

Đo đúng cần hai view đồng bộ:

1. Nsight Systems: CPU thread scheduling, Python gaps, CUDA API, graph launches,
   kernels, memcpy, synchronization.
2. Nsight Compute trên một replay nhỏ: achieved bandwidth, occupancy, registers,
   tensor-core utilization, launch shape và memory transactions.

Sau đó phân tách theo prefill/decode, cold/warm, batch size và conversation turn.

### TTFT không chỉ là prefill

TTFT gồm queueing + tokenize + scheduling + prefill + first decode + streaming.
Trong arrival Poisson, queueing và interference với active decodes có thể lớn hơn
tokenizer 0.2 ms. Prefix caching chỉ giúp nếu prompt turns thật sự lặp đúng token
prefix và cache chưa bị evict. Cần báo hit rate theo turn 1..6.

### TPOT không chỉ là một decode step

TPOT quan sát tại client bị ảnh hưởng bởi scheduler cadence, batch composition,
detokenize, event-loop, HTTP flush và burstiness. Nếu server phát nhiều token
trong một chunk, timestamp theo chunk không đủ để suy ra per-token latency. Metric
definition trong evaluator phải được sao chép chính xác.

## Vì sao submission cuối 51 ERS là tín hiệu quý nhất

Một regression khoảng 12 điểm lớn hơn lợi ích 1% của ba fusion. Candidate cuối
đã đổi nhiều thứ cùng lúc: code overlay, allocator, thread count, shm, tokenizer,
kernels và có thể image/config. Không thể biết thủ phạm từ một aggregate score.

Quy trình đúng:

1. Submit lại known-good artifact bằng exact digest để đo evaluator drift.
2. Nếu known-good vẫn tốt, tạo binary-search overlays: infrastructure only,
   tokenizer only, từng kernel riêng, rồi all kernels.
3. Mỗi image tự in manifest lúc boot: base digest, `vllm.__version__`, git SHA,
   hashes của file overlay, env allowlist và parsed engine config.
4. Startup smoke test đối chiếu import path và function signature; fail fast thay
   vì im lặng chạy half-patched package.

## Đánh giá từng hướng

### Config tuning

Đúng để bắt đầu, nhưng `max-num-seqs=256` không mặc nhiên tốt khi chỉ có 70 hội
thoại. Sau khi đủ chứa active set, giá trị lớn hơn có thể chỉ tăng state/graph
surface. `max-num-batched-tokens=8192` chủ yếu điều khiển prefill budget; phải
được tune cùng chunked prefill và decode priority, không độc lập.

FP8 weights/KV phải qua quality gate. KV FP8 tăng capacity, nhưng capacity chưa
chắc là bottleneck với model 1.2B và context thực tế. Đừng suy từ “ít bộ nhớ hơn”
thành “latency nhanh hơn”.

### Speculative decoding

Quyết định dừng hướng này trong một contest một tuần là hợp lý. Nhưng kết luận
“dead end hoàn toàn” quá rộng. Chính xác hơn: separate hybrid LFM drafter trên
version/PR đã thử chưa được vLLM hỗ trợ end-to-end, và kết quả ERS 36 không có
breakdown acceptance/draft/verify nên chưa chỉ ra một nguyên nhân duy nhất.

N-gram 22/30 A/B match cũng chưa phải correctness test đủ chặt. Với greedy decode,
output phải khớp token-for-token; với sampling, phải kiểm tra phân phối hoặc dùng
deterministic RNG semantics tương thích.

### Kernel fusion

Đây là kỹ năng khó và đáng ghi nhận. Tuy nhiên current vLLM source đã có compile
fusion families như QK-norm/RoPE, nên trước khi duy trì custom kernel cần kiểm tra
exact version có pattern đó không và model graph có match không. Với CUDA Graph,
fusion vẫn giảm memory traffic nhưng không còn tiết kiệm ba CPU round trips theo
cách mô tả đơn giản.

Ưu tiên kernel theo `frequency × measured duration × removable fraction`, không
theo số kernel launch trên sơ đồ PyTorch. Vì kiến trúc thật có 10 convolution
blocks, ShortConv có thể đáng ưu tiên hơn bài viết đã tính — nhưng chỉ profiler
mới kết luận được.

### Docker/frontend

Runtime overlay là công cụ development tốt nhưng artifact contest phải immutable.
Pin base bằng digest, `COPY` file allowlist thay vì cả directory, kiểm tra SHA256,
và chạy import/correctness smoke test trong build.

`OMP_NUM_THREADS=3` không tự động đúng chỉ vì có 3 vCPU; engine, API loop,
tokenizer và OpenMP sẽ cạnh tranh. Cần đo CPU utilization/run queue/context switch.
tcmalloc cũng là một candidate A/B, không phải default hiển nhiên.

## Kết luận

Điểm mạnh nổi bật là năng lực đào source và triển khai xuyên tầng. Điểm làm mất
score là scientific control: chưa xác nhận metric, chưa giữ target-fidelity, thay
nhiều biến cùng lúc, và không bisect regression cuối. Nâng quy trình thực nghiệm
sẽ có expected value cao hơn một custom kernel mới.
