# LFM RaceBench

**Website:** [Experiment flow](https://buicongnguyen.github.io/cuda-vllm-optimize/) ·
[Learning path](https://buicongnguyen.github.io/cuda-vllm-optimize/learn.html)

Một repository **benchmark-first** để phân tích và tối ưu serving
`LiquidAI/LFM2.5-1.2B-Instruct` bằng vLLM trên H200 MIG `1g.18gb`, 3 vCPU và
8 GB RAM. Repo được dựng từ bài viết “Một tuần ép từng millisecond tối ưu hóa
AI inference – Viettel AI Race”, nhưng không coi mọi con số trong bài là ground
truth.

## Kết luận nhanh

- Hướng làm tốt nhất trong bài là nhận ra đây là bài toán serving đồng thời,
  memory movement, scheduler và launch overhead — không phải chỉ là FLOPS.
- Hai dữ kiện quan trọng trong bài đang mâu thuẫn với nguồn hoặc với chính phép
  tính: kiến trúc thật là **10 convolution + 6 GQA**, và công thức được trích dẫn
  cho **63.1851 ERS** tại `TTFT=47 ms, TPOT=4 ms`, không phải khoảng 42.
- Với chính công thức đó, tại TTFT 47 ms, muốn đạt 72 ERS cần TPOT khoảng
  **2.909 ms**. Gần điểm này, giảm 0.1 ms TPOT đáng giá khoảng 0.74 điểm ERS;
  giảm 1 ms TTFT chỉ đáng giá khoảng 0.23 điểm.
- Speculative decoding cho draft model LFM hybrid hiện không phải một flag dễ
  bật. vLLM issue #49112 chỉ ra draft model cần nhiều KV-cache group trong khi
  proposer vẫn giả định một group/`AttentionMetadata`.
- Kernel fusion có giá trị học thuật và có thể có giá trị thực tế, nhưng chỉ nên
  viết sau khi Nsight Systems chứng minh kernels vẫn được launch rời bên ngoài
  CUDA Graph/compile fusion và chiếm phần đủ lớn của critical path.

Đọc phân tích đầy đủ tại [docs/ANALYSIS.vi.md](docs/ANALYSIS.vi.md), luồng ra
quyết định tại [docs/DECISION_FLOW.vi.md](docs/DECISION_FLOW.vi.md), chiến lược
điểm số tại [docs/SCORE_STRATEGY.vi.md](docs/SCORE_STRATEGY.vi.md), và lộ trình
kỹ năng tại [docs/SKILLS_ROADMAP.vi.md](docs/SKILLS_ROADMAP.vi.md).

## Repo này cung cấp gì?

```text
src/racebench/          ERS calculator, metric definitions, workload plan, validators
tests/                  regression tests cho công thức và dữ liệu benchmark
data/claims.json        audit: verified / contradicted / inferred / unverified
experiments/ledger.csv  sổ thí nghiệm one-change-at-a-time
configs/                workload và vLLM args mẫu, có ghi rõ phần chưa biết
docs/                   GitHub Pages site + analysis, decision flow, score strategy
```

Repo không giả vờ cung cấp Triton kernels chưa được kiểm tra trên SM90 MIG, không
chép mã từ fork bên ngoài, và không gọi một config là “winning config” khi chưa
có raw evaluator output.

## Chạy nhanh

Yêu cầu Python 3.11+, không có runtime dependency bên ngoài.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v

racebench score --ttft 47 --tpot 4 --target 72
racebench validate-claims data/claims.json
racebench validate-ledger experiments/ledger.csv
racebench workload --conversations 70 --turns 6 --rate 7 --seed 2025
```

`--rate 7` chỉ là giá trị mẫu vì bài viết nói arrival Poisson nhưng không cho
lambda. Phải thay bằng tham số chính thức trước khi dùng kết quả.

## Protocol thí nghiệm tối thiểu

1. Pin image bằng digest, vLLM commit, model revision, CUDA driver và MIG profile.
2. Tái lập baseline ít nhất ba lần trên target hardware; giữ raw per-request
   timestamps và token counts.
3. Chỉ đổi một biến; chạy correctness gate trước performance gate.
4. Báo cáo mean, median, p95, confidence interval và hit-rate/batch-size histogram,
   không chỉ một ERS tổng.
5. Chỉ submit portal khi candidate vượt noise floor local trên H200 MIG.

## Nguồn chính

- [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [NVIDIA H200 MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
- [vLLM issue #49112](https://github.com/vllm-project/vllm/issues/49112)
- [vLLM optimization levels](https://docs.vllm.ai/en/latest/design/optimization_levels/)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [Referenced hybrid-vllm fork](https://github.com/minh-vt/hybrid-vllm)

## Trạng thái

Đây là analysis/harness repo sẵn sàng đưa lên GitHub. Nó chưa phải fork vLLM và
chưa được benchmark trên GPU của cuộc thi trong workspace hiện tại.
