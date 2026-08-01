# Luồng ra quyết định có thể kiểm chứng

“Thinking flow” ở đây là bản ghi quyết định có thể audit, không phải độc thoại
nội tâm. Mỗi bước phải để người khác tái lập từ observation → hypothesis → test →
evidence → decision.

## Gate 0 — Đóng băng luật chơi

Trước khi tối ưu, lưu nguyên văn:

- công thức, unit, clipping và cách aggregate TTFT/TPOT;
- prompt/token distribution, output length, sampling settings;
- Poisson lambda, giới hạn concurrency và causal semantics của 6 turns;
- image/base packages được phép, quy tắc quality và network;
- hardware profile chính xác, power/clock policy và warm-up behavior.

Nếu thiếu một mục, đánh dấu `UNKNOWN` như `configs/contest.example.json`; không tự
điền rồi biến giả định thành fact.

## Gate 1 — Baseline có tái lập được không?

Chạy exact artifact ít nhất ba lần trên target. Lưu raw request records. Baseline
pass khi:

- startup manifest giống nhau;
- output quality/token counts giống expectation;
- confidence interval đủ nhỏ để phân biệt mức cải thiện định thử;
- rerun xen kẽ A/B/A không có drift rõ rệt.

Nếu không pass, tối ưu lúc này chỉ là đo noise.

## Gate 2 — Vẽ critical path bằng dữ liệu

Thu thập bốn lớp evidence:

1. Client: arrival, submit, first byte/token, every stream chunk, completion.
2. Server: queueing, tokenize, schedule, prefill/decode, detokenize.
3. System: CPU run queue, context switches, cgroup memory, page faults.
4. GPU: CUDA API gaps, graph launch, kernel duration, achieved bandwidth.

Phân nhóm theo cold/warm, turn, prompt/output bucket và active batch size. Một
mean toàn cục che mất nguyên nhân.

## Gate 3 — Xếp giả thuyết theo expected value

Dùng một bảng trước khi code:

| Trường | Câu hỏi |
|---|---|
| Metric | TTFT, TPOT hay quality nào sẽ đổi? |
| Max gain | Nếu loại bỏ 100% vùng này thì tối đa tăng bao nhiêu ERS? |
| Probability | Evidence nào cho thấy optimization sẽ hit path thật? |
| Cost | Bao nhiêu giờ/GPU/submission? |
| Risk | Crash, numerical drift, ABI mismatch hay regression tail? |
| Kill criterion | Kết quả nào khiến dừng ngay? |

Ưu tiên gần đúng bằng `(expected ERS gain × probability) / total cost`.

## Gate 4 — Local gates trước portal

Mỗi candidate đi qua:

1. Static/source gate: feature có thật sự active cho hybrid model không?
2. Correctness gate: greedy exact match, sampling semantics, long/short contexts,
   batch shapes và fallback path.
3. Target microbenchmark: kernel hoặc component thực sự nhanh hơn trên SM90 MIG.
4. End-to-end replay: cùng seed/workload, ít nhất A/B/A.
5. Robustness: memory headroom, cold start, p95/p99, no hidden fallback.
6. Portal: chỉ khi improvement vượt noise floor và expected score gain đáng kể.

## Gate 5 — Một submission trả lời đúng một câu hỏi

Với 5 lượt/ngày, portal là dụng cụ xác nhận external validity, không phải compiler
hay unit test. Một lịch ngày mẫu:

1. Known-good control để phát hiện evaluator drift.
2. Candidate có expected value cao nhất.
3. Replicate candidate nếu thắng; nếu thua, chạy diagnostic sibling.
4. Candidate thứ hai chỉ khi độc lập với candidate đầu.
5. Giữ một lượt cho confirm/rollback cuối ngày.

Mỗi row trong `experiments/ledger.csv` có parent và `one_change`. Nếu thay nhiều
biến, gọi đó là integration build và không dùng nó để kết luận từng optimization.

## Decision tree theo bottleneck

- Nếu client TTFT cao nhưng GPU idle: kiểm tra queue/event loop/tokenizer/HTTP.
- Nếu prefill chặn decode: tune chunked prefill và scheduler budgets.
- Nếu decode có CPU gaps giữa graph launches: scheduler/async/CUDA Graph trước.
- Nếu GPU liên tục và DRAM gần trần: giảm bytes, quantization hoặc fusion.
- Nếu kernels nhỏ, rời rạc và ngoài graph: fusion/graph capture.
- Nếu KV pressure/eviction cao: context cap, FP8 KV và prefix-cache policy.
- Nếu candidate local thắng nhưng portal thua: artifact/config/hardware diff và
  evaluator semantics trước khi đổi thuật toán.

## Kill criteria cho các hướng đắt

### Speculative decoding

Dừng nếu init/correctness chưa pass trên exact vLLM commit, hoặc nếu
`accepted_tokens / target_forward` không bù draft + verify cost dưới batch-size
distribution thật. Acceptance percent một mình không đủ.

### Custom kernel

Dừng nếu operator đã được compile-fuse, nằm trong CUDA Graph và phần removable
nhỏ hơn noise end-to-end; hoặc nếu registers/occupancy trên SM90 làm kernel fused
chậm hơn unfused.

### Version upgrade

Dừng “upgrade all”. Bisect by commit/feature hoặc backport một patch. Base image
đổi đồng thời torch, CUDA, Triton, scheduler và kernels nên không phải một biến.
