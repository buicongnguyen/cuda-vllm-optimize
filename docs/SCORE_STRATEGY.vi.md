# Chiến lược để tăng điểm

## 1. Kiểm toán score trước

Công thức được cung cấp là:

```text
ERS = 100 × [0.5 × ((400 - TTFT) / 390)²
           + 0.5 × ((10 - TPOT) / 9)²]
```

Từ công thức này:

| TTFT | TPOT | ERS |
|---:|---:|---:|
| 47 ms | 4.0 ms | 63.185 |
| 47 ms | 2.909 ms | 72.000 |
| 47 ms | 2.684 ms | 74.000 |
| 20 ms | 1.602 ms | 91.000 |

Dòng cuối chỉ là nghiệm toán học, không chứng minh top 1 có đúng cặp latency đó.
Các mốc 74/91 trong bài cũng chưa có nguồn chính thức trong repo.

Tại 47/4, đạo hàm gần đúng:

- giảm 1 ms TTFT: +0.232 ERS;
- giảm 1 ms TPOT: +7.407 ERS.

Vì vậy 0.1 ms TPOT tương đương khoảng 3.2 ms TTFT về điểm cục bộ. Bài viết nói
trọng số 50/50 nhưng bỏ qua scale chuẩn hóa rất khác nhau; chiến lược đúng phải
ưu tiên TPOT cho tới khi evaluator chính thức chứng minh công thức khác.

## 2. Roadmap theo expected score

### P0 — Loại bỏ regression 63 → 51

Đây có thể trả lại khoảng 12 điểm, lớn hơn mọi micro-optimization đã nêu.

- Rebuild/submit known-good digest.
- Tách infrastructure, allocator, tokenizer và từng kernel.
- Assert loaded source hashes ở startup.
- So sánh raw TTFT/TPOT, không chỉ ERS.

### P1 — Làm batch decode ổn định và graphable

Với khoảng 70 conversations, cơ hội lớn là amortize weight/scheduler overhead qua
batch và giảm CPU gaps:

- profile histogram active decode batch sizes;
- capture CUDA Graph sizes đúng vùng histogram thay vì một dải rộng mù quáng;
- kiểm tra full/piecewise graph support cho exact hybrid model/version;
- benchmark async scheduling nếu version hỗ trợ và correctness constraints cho
  structured/spec decode không liên quan;
- tune `max-num-seqs` quanh 70–96, không mặc định 256;
- tune prefill budget để arrivals mới không phá cadence decode.

Mục tiêu là giảm TPOT client-observed và variance, không chỉ kernel time.

### P2 — Prefix reuse và TTFT theo turn

Multi-turn tạo natural shared prefix trong cùng conversation. Đo:

- cache hit tokens/turn;
- eviction và memory headroom;
- TTFT turn 1 so với turns 2–6;
- effect của block size và hash backend chỉ khi quality exact.

Nếu prompts giữa turns được serialize khác nhau hoặc cache evict sớm, prefix
caching có thể không tạo lợi ích như kỳ vọng.

### P3 — Frontend trên 3 vCPU

- Flamegraph/tokenizer benchmark trên đúng prompt distribution.
- A/B fastokens với exact output IDs và streaming text.
- Tắt request/access logs không cần thiết.
- Đo one-process vs multiprocess/API worker contention.
- A/B allocator và `OMP_NUM_THREADS`; không cộng dồn trước khi xác nhận.
- Giữ HTTP client connection, tránh flush/syscall thừa nếu luật cho phép.

0.2 ms/request có thể hữu ích cho TTFT nhưng không được nhầm với 0.2 ms TPOT.

### P4 — Precision và memory

- FP8 weight: xác nhận kernel backend thực dùng FP8 trên SM90 MIG, không dequant
  fallback; đo quality.
- FP8 KV: đo hit/capacity/attention bandwidth; dùng calibrated scales nếu luật
  cho phép và quality cần.
- Giảm `max-model-len` chỉ khi workload không cần 8192; capacity thừa không tự
  làm request nhanh, nhưng memory headroom có thể giúp batching/graphs.

### P5 — Fusion có profiler chứng minh

Thứ tự candidate sau khi sửa layer count:

1. ShortConv state/update path xuất hiện 10 lần/layer stack.
2. Q/K norm + RoPE ở 6 GQA blocks, sau khi kiểm tra compile pass hiện có.
3. Activation + quant nếu intermediate traffic thật sự tồn tại trong graph.

Mỗi kernel cần benchmark shape matrix theo observed decode/prefill batches,
register count SM90, correctness adversarial và end-to-end A/B. Không dùng L4
pass làm bằng chứng performance cho H200 MIG.

### P6 — Spec decode chỉ là research branch

Chỉ quay lại khi:

- multi-group draft metadata support chạy end-to-end;
- ShortConv state commit/rollback correctness pass;
- draft model memory không làm giảm batch/cache lợi ích;
- measured break-even dưới workload thật dương.

Nếu contest chỉ một tuần và 5 submissions/ngày, expected value của hướng này thấp
hơn scheduler/CUDA Graph/bisect.

## 3. Ma trận thí nghiệm đề xuất

Sau baseline, không chạy full Cartesian grid. Làm sequential DOE:

1. `max-num-seqs`: 72, 80, 96 (local only, chọn vùng tốt).
2. `max-num-batched-tokens`: quanh prompt-length percentiles, không chỉ powers of 2.
3. chunked prefill on/off và partial-prefill budget.
4. graph capture set derived từ batch histogram.
5. prefix cache on/off, báo theo turn.
6. từng frontend candidate.
7. từng kernel overlay.

Candidate phải thắng A/B/A và vượt confidence/noise threshold trước khi portal.

## 4. Mốc thực dụng

Nếu công thức đúng và TTFT giữ quanh 47 ms:

- Top-8 74 yêu cầu TPOT khoảng 2.684 ms: cần giảm 1.316 ms từ mốc 4 ms.
- 72 yêu cầu khoảng 2.909 ms: cần giảm 1.091 ms.
- Ba kernel chỉ cải thiện 1% của 4 ms, tức khoảng 0.04 ms, đem lại xấp xỉ
  0.30 ERS ở vùng baseline — không thể tự nó bù khoảng cách.

Do đó cơ hội điểm lớn phải đến từ một thay đổi cấp execution: batching/graph,
backend/precision path thật, loại bỏ CPU stall lớn, hoặc sửa regression artifact.
