# Chiến lược để tăng điểm

## 1. Kiểm toán score trước

Công thức được cung cấp là:

```text
ERS = 100 × [0.5 × ((400 - TTFT) / 390)²
           + 0.5 × ((10 - TPOT) / 9)²]
```

Biểu thức nguyên văn chỉ có nghĩa trong khoảng 10–400 ms TTFT và 1–10 ms TPOT.
Ngoài khoảng đó, bình phương thưởng điểm cho latency tệ hơn (1000 ms / 20 ms cho
khoảng 180), nên calculator và `racebench score` clamp mỗi thành phần về [0, 1].
Mọi số trong tài liệu này nằm trong khoảng nên không đổi.

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
trọng số 50/50 nhưng bỏ qua scale chuẩn hóa rất khác nhau.

Đạo hàm cục bộ chưa đủ để xếp ưu tiên; còn phải xem mỗi metric còn bao nhiêu
điểm để lấy. Tại 47/4, TTFT đóng góp 40.96/50 nên dù giảm tới mốc 10 ms cũng chỉ
thêm tối đa 9.04 ERS; TPOT đóng góp 22.22/50 nên còn tối đa 27.78 ERS tới mốc
1 ms. Cả leverage lẫn headroom đều nghiêng về TPOT, nên chiến lược đúng là ưu
tiên TPOT cho tới khi evaluator chính thức chứng minh công thức khác.

## 2. Roadmap theo expected score

### P0 — Loại bỏ regression 63 → 51

Đây có thể trả lại khoảng 12 điểm, lớn hơn mọi micro-optimization đã nêu.

- Rebuild/submit known-good digest.
- Tách infrastructure, allocator, tokenizer và từng kernel.
- Assert loaded source hashes ở startup.
- So sánh raw TTFT/TPOT, không chỉ ERS.

### P1 — Làm batch decode ổn định và graphable

Decode batch thực tế nhỏ hơn 70 rất nhiều. Causality giới hạn mỗi conversation
tối đa một request đang chạy, và Little's law cho số request đang chạy trung bình
= λ × latency: khoảng 1.8 ở λ = 7 giả định trên RTX 4080. Ở batch nhỏ như vậy,
mỗi decode step vẫn phải đọc toàn bộ weights: 2.34 GB BF16 trên 736 GB/s của RTX
4080 Super là khoảng 3.2 ms (roofline estimate), gần với TPOT ≈ 3.9 ms đo được.
Cơ hội vì vậy nằm ở bytes/step, launch và CPU gaps hơn là batching lớn:

- profile histogram active decode batch sizes thay vì giả định 70;
- capture CUDA Graph sizes đúng vùng histogram (chủ yếu batch nhỏ);
- kiểm tra full/piecewise graph support cho exact hybrid model/version;
- async scheduling đã bật mặc định (log vLLM 0.25.1: “Asynchronous scheduling is
  enabled”); thí nghiệm hợp lệ duy nhất là ablation `--no-async-scheduling`;
- đừng screen `max-num-seqs` ≥ 70: giá trị đó không bao giờ làm request phải chờ,
  chỉ đổi graph capture range và memory reservation;
- tune prefill budget chỉ khi histogram cho thấy prefill và decode thật sự chung batch.

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

Nếu contest chỉ một tuần và 5 submissions/thành viên/ngày, expected value của
hướng này thấp hơn scheduler/CUDA Graph/bisect — với các mốc tới khoảng 78 ERS.
Nhưng mục 4 cho thấy trên MIG `1g.18gb` mỗi step đọc weights mất ít nhất ~2.17 ms,
nên điểm trên khoảng 85–88 chỉ đạt được khi sinh nhiều hơn một token mỗi step.
Nếu mục tiêu là các mốc đó, speculation không còn là “research branch” mà là con
đường duy nhất.

## 3. Ma trận thí nghiệm đề xuất

Sau baseline, không chạy full Cartesian grid. Làm sequential DOE:

1. Đo histogram số request đang chạy và prompt length trước. `max-num-seqs` ≥ 70
   không bao giờ bind, nên chỉ thử giá trị nhỏ hơn khi graph capture hoặc memory
   thật sự quan trọng.
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

### Sàn vật lý của TPOT (roofline estimate, chưa đo)

Ở batch ~2, mỗi decode step phải đọc toàn bộ weights. Online FP8 của vLLM chỉ
quantize các lớp Linear; embedding/lm_head (65,536 × 2,048, tied) vẫn BF16. Mỗi
step vì vậy đọc khoảng 1.036 GB FP8 + 0.268 GB BF16 ≈ 1.30 GB. MIG `1g.18gb` có
1/8 memory của H200 (4.8 TB/s), tức khoảng 0.6 TB/s, nên TPOT không thể dưới
khoảng **2.17 ms** khi mỗi step sinh một token.

- Mốc 72 (2.909 ms) cần khoảng 75% peak bandwidth; mốc 74 (2.684 ms) cần khoảng 81%.
- Trần ERS với một token mỗi step: khoảng 78.8 ở TTFT 47 ms, 85.3 ở 20 ms và 87.8
  ngay cả ở 10 ms.
- Mốc 91 cần TPOT ≤ 1.60 ms ở 20 ms (≤ 1.85 ms ở 10 ms), dưới sàn này. Nếu mốc đó
  có thật trên leaderboard, hoặc công thức/aggregation khác với bản được trích,
  hoặc đội đó sinh nhiều hơn một token mỗi weight read (speculative decoding).

Con số 0.88 ms trong bài khớp với 0.6 GB ÷ (4.8 TB/s ÷ 7) = 0.875 ms, tức là dùng
tỷ lệ SM (1/7) cho bandwidth và 0.6 GB cho weights; cả hai đều sai hướng làm sàn
trông thấp hơn thực tế (chưa đối chiếu nguyên văn bài).

Do đó cơ hội điểm lớn phải đến từ một thay đổi cấp execution: batching/graph,
backend/precision path thật, loại bỏ CPU stall lớn, hoặc sửa regression artifact.
