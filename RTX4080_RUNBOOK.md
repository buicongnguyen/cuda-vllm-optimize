# Chạy LFM RaceBench trên RTX 4080 Super với WSL2 Ubuntu 22.04

Đây là hướng dẫn thực thi đã được kiểm chứng trên máy Windows có RTX 4080
Super 16 GB. Mục tiêu là tái tạo **phương pháp thí nghiệm** bằng vLLM, không
phải tái tạo điểm tuyệt đối của H200 MIG hoặc hệ thống chấm cuộc thi.

## Đường chạy khuyến nghị

Nếu repository đang nằm trên Windows, hãy chạy mọi lệnh điều khiển từ
PowerShell. Script sẽ tự copy source sang filesystem Linux nhanh hơn trong WSL.

```mermaid
flowchart LR
    A["Windows repository"] --> B["PowerShell bootstrap"]
    B --> C["Ubuntu 22.04 WSL2"]
    C --> D["WSL ext4 mirror"]
    D --> E["Pinned Python + vLLM"]
    E --> F["Doctor"]
    F --> G["Smoke: 4 requests"]
    G --> H["Baseline: 420 requests"]
    H --> I["A/B/A: 1,260 requests"]
    I --> J["Decision + drift guard"]
```

Ba lệnh quan trọng nhất, chạy tại repository root trong PowerShell:

```powershell
# 1. Cài đặt và xác nhận model có thể serve
.\scripts\rtx4080_bootstrap.ps1 -Run smoke

# 2. Đo một baseline 70 conversations × 6 turns
.\scripts\rtx4080_bootstrap.ps1 -Run baseline

# 3. So sánh R0 → candidate → R0′ và kiểm tra drift
.\scripts\rtx4080_bootstrap.ps1 -Run aba
```

Không cần mở server thủ công, không cần Docker và không cần cài CUDA Toolkit
đầy đủ trong Ubuntu.

## Trước khi bắt đầu

Bạn cần:

- Windows 10/11 với NVIDIA driver đang hoạt động;
- RTX 4080 Super 16 GB;
- Ubuntu 22.04 được cài dưới WSL2;
- khoảng 25 GB disk trống và Internet cho lần cài/model download đầu tiên;
- đóng game, Stable Diffusion, trình render hoặc chương trình khác đang dùng VRAM.

Driver GPU nằm trên Windows. **Không cài `nvidia-driver-*` bên trong Ubuntu
WSL.** WSL sẽ map driver Windows vào Linux.

## Bước 1 — xác nhận đúng Ubuntu 22.04 WSL2

Mở PowerShell. Các lệnh sau không cần Administrator nếu WSL đã được cài:

```powershell
wsl --list --verbose
wsl -d Ubuntu-22.04 -- cat /etc/os-release
wsl -d Ubuntu-22.04 -- nvidia-smi
```

Kết quả cần thấy:

```text
NAME             STATE    VERSION
Ubuntu-22.04     ...      2

PRETTY_NAME="Ubuntu 22.04..."
NVIDIA GeForce RTX 4080 SUPER
```

Nếu chưa có distro, chỉ bước cài này cần PowerShell chạy với Administrator:

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
```

Sau đó restart Windows nếu được yêu cầu, chạy `wsl -d Ubuntu-22.04` một lần
và hoàn tất việc tạo Linux username/password.

## Bước 2 — cài hai công cụ hệ thống nhỏ

Mở Ubuntu:

```powershell
wsl -d Ubuntu-22.04
```

Trong terminal Ubuntu, chạy:

```bash
sudo apt-get update
sudo apt-get install -y curl rsync
exit
```

`curl` tải trình quản lý Python `uv`. `rsync` copy repository từ `/mnt/c` sang
WSL ext4. Setup sẽ tự cung cấp compiler cho Triton nếu Ubuntu chưa có
`gcc`/`clang`; bạn không cần tự cài CUDA Toolkit hoặc `nvcc`.

## Bước 3 — đi đến repository trong PowerShell

Ví dụ:

```powershell
cd C:\Users\YOUR_NAME\source\repos\cuda_vllm_optimize
Test-Path .\scripts\rtx4080_bootstrap.ps1
```

`Test-Path` phải trả `True`. Không chạy script từ thư mục `scripts`; hãy đứng ở
repository root như ví dụ trên.

## Bước 4 — setup và smoke test

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run smoke
```

Lần đầu có thể mất nhiều phút vì phải tải Python, vLLM/PyTorch/CUDA wheels và
model. Script thực hiện tuần tự:

1. chọn chính xác distro `Ubuntu-22.04`;
2. xác nhận GPU qua `nvidia-smi`;
3. mirror Windows checkout vào `~/src/cuda-vllm-optimize` trên WSL ext4 và ghi
   commit nguồn (thêm hậu tố `-dirty` nếu có thay đổi chưa commit);
4. tạo venv `~/.venvs/lfm-racebench-rtx4080` với Python 3.12;
5. pin vLLM 0.25.1 và model revision;
6. chạy doctor;
7. start vLLM, đợi health endpoint, chạy warm-up không ghi nhận, gửi 4
   streaming requests được đo rồi stop server;
8. lưu config, manifest, server log và raw request records.

Checkpoint thành công ở cuối output:

```text
Overall: READY
requested: 4
successful: 4
failed: 0
Artifacts: /home/<linux-user>/src/cuda-vllm-optimize/results/rtx4080/...
```

Smoke chỉ xác nhận environment và request flow hoạt động. Không dùng TTFT,
TPOT hoặc ERS của bốn request này làm benchmark.

### Chỉ setup, không chạy model

```powershell
.\scripts\rtx4080_bootstrap.ps1
```

### Chỉ chạy doctor sau khi setup

```powershell
.\scripts\rtx4080_bootstrap.ps1 -DoctorOnly
```

`-DoctorOnly` là kiểm tra **sau setup**. Nếu venv chưa tồn tại, launcher sẽ yêu
cầu chạy setup trước.

## Bước 5 — chạy baseline 420 requests

Chỉ tiếp tục khi smoke có `failed: 0`:

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run baseline
```

Workload mặc định:

| Thuộc tính | Giá trị |
|---|---:|
| Conversations | 70 |
| Turns mỗi conversation | 6 |
| Tổng requests | 420 |
| Arrival model | Poisson |
| Local assumed rate | 7 requests/s |
| Max output | 64 tokens |
| Seed | 2025 |

Turn sau luôn đợi turn trước của cùng conversation, nên đây không phải 420
independent prompts. `7 requests/s` là giả định local vì bài viết không công
bố lambda chính thức.

Checkpoint thành công:

```text
requested: 420
successful: 420
failed: 0
metric_eligible: 420
```

## Bước 6 — chạy thí nghiệm R0/B/R0′

```powershell
.\scripts\rtx4080_bootstrap.ps1 -Run aba
```

Một block mặc định chạy ba server tuần tự:

```text
R0  baseline (--no-enable-prefix-caching)
 ↓  stop server hoàn toàn
B   cùng config + --enable-prefix-caching --mamba-cache-mode=align
 ↓  stop server hoàn toàn
R0′ baseline quay lại
 ↓
paired statistics + bootstrap 95% CI (theo conversation) + score gate
```

Trên LFM2 (hybrid conv + GQA), bật prefix caching không phải một thay đổi đơn lẻ:
vLLM 0.25.1 tự chuyển conv-state cache sang mode `align` (chia chunk theo block và
checkpoint state). Config ghi rõ cả hai để diff thể hiện đúng bundle; hit rate
xem bằng `grep "Prefix cache hit rate" B-candidate-server.log`.

Mỗi server mới trước tiên phục vụ 4 warm-up conversations × 2 turns không được
ghi nhận (`--warmup-conversations`, mặc định 4). Server vừa khởi động phải JIT
một số Triton kernels ở những request đầu tiên; không warm-up thì vài request
đó chi phối mean TTFT (xem kết quả bên dưới).

Để request `c012-t04` là **cùng một prompt** ở R0, B và R0′, replay mặc định dùng
`--history canned` (assistant turns cố định thay vì câu trả lời của chính run đó —
vLLM không batch-invariant nên câu trả lời có thể khác giữa các run) và
`--output-length fixed` (mọi request decode đúng `max-tokens`, cùng mẫu số TPOT).
Mỗi request ghi `prompt_sha256` và `output_text`; comparison chỉ pair các prompt
trùng hash. `--history live --output-length natural` vẫn có cho workload thực tế
hơn, nhưng khi đó chỉ turn 1 pair chính xác.

Mỗi stage có 420 requests, tổng cộng 1,260. Trên máy đã kiểm chứng, block mất
khoảng bốn phút sau khi environment và model cache đã sẵn sàng. Terminal có
thể im lặng trong lúc replay; không đóng cửa sổ nếu chưa thấy error hoặc
`Artifacts:`.

Đọc trường sau trong `comparison.json`:

```json
{
  "decision": {
    "classification": "inconclusive_outlier_dominated",
    "promote": false
  }
}
```

Quy tắc — mục tiêu là ERS, nên CI của ΔERS so với trung điểm R0/R0′ quyết định
hướng; từng latency metric chỉ là chẩn đoán:

- `candidate_score_gain_pending_correctness`: CI của ΔERS nằm trên 0 và qua các
  kiểm tra drift; vẫn cần correctness và nhiều block lặp lại. Metric nào chậm hơn
  được ghi là trade-off;
- `uncertain`: CI của ΔERS đi qua 0;
- `reject_score_loss` hoặc `reject_failures`: không tiếp tục candidate;
- `inconclusive_outlier_dominated`: mean paired delta nói “faster” nhưng median
  paired delta không âm — thường là cold start; một cải thiện tail thật phải còn
  nguyên khi bỏ các arrivals đầu;
- `inconclusive_due_to_drift`: R0′ cải thiện cùng chiều và ít nhất bằng gain của B
  (kể cả khi CI của drift đi qua 0), hoặc R0′ fail nhiều request hơn R0;
- `incomplete_without_baseline_return`: thiếu R0′, không được promote.

Correctness đi kèm: `pairing.output_match_rate` của B↔R0 không nên thấp hơn của
R0′↔R0 (noise của chính baseline); exact match 100% không phải chuẩn vì vLLM không
batch-invariant.

Request fail không làm dừng block: replay ghi lại chúng (exit code 3) và gate
`reject_failures` quyết định. Hai run chỉ được so khi cùng model, seed, rate và
workload shape; nếu khác, `rtx4080_compare.py` từ chối pair.

## Bước 7 — tìm và mở kết quả từ Windows

Khi dùng PowerShell route, benchmark **không chạy từ `/mnt/c`**. Source được
mirror và kết quả nằm trong WSL:

```bash
~/src/cuda-vllm-optimize/results/rtx4080/<timestamp>/
```

Liệt kê từ PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'ls -lt $HOME/src/cuda-vllm-optimize/results/rtx4080 | head'
```

Mở bằng Windows Explorer:

```powershell
explorer.exe \\wsl.localhost\Ubuntu-22.04\home
```

Sau đó chọn Linux username của bạn → `src` → `cuda-vllm-optimize` →
`results` → `rtx4080`.

Mỗi result directory có:

```text
experiment-plan.json       exact commands, seed, workload và candidate diff
doctor.json                environment gate
R0-*.args                  server arguments thực sự đã dùng
R0-*-manifest.json         GPU, driver, package versions và source commit
R0-*-server.log            model load, graph capture, warning/error
R0-*.jsonl                 raw record của từng request + summary
comparison.json            chỉ có ở A/B/A
```

## Alternative — chạy hoàn toàn bên trong WSL

Chỉ dùng route này nếu repository đã được clone trực tiếp vào filesystem Linux,
ví dụ `~/src/cuda-vllm-optimize`. Đừng trộn command của hai route trong cùng
một lần setup.

```powershell
wsl -d Ubuntu-22.04
```

```bash
sudo apt-get update
sudo apt-get install -y git curl

mkdir -p ~/src
cd ~/src
ssh -T git@github.com        # xác nhận SSH key cũng có trong WSL
git clone git@github.com:buicongnguyen/cuda-vllm-optimize.git
cd cuda-vllm-optimize

bash scripts/rtx4080_setup_wsl.sh
source ~/.venvs/lfm-racebench-rtx4080/bin/activate

python scripts/rtx4080_lab.py doctor
python scripts/rtx4080_lab.py run --mode smoke
python scripts/rtx4080_lab.py run --mode baseline
python scripts/rtx4080_lab.py run --mode aba
```

Trong route này, source và results đều nằm ngay trong clone WSL hiện tại; setup
không tạo thêm mirror. SSH agent/key của Windows không tự động luôn xuất hiện
trong WSL; nếu `ssh -T` fail, cấu hình GitHub SSH key trong WSL trước hoặc clone
read-only bằng HTTPS.

## Cấu hình 16 GB đang dùng

Baseline nằm tại
[`configs/vllm/rtx4080-r0.args`](configs/vllm/rtx4080-r0.args):

- `--max-model-len=4096` để giảm state/graph pressure;
- `--gpu-memory-utilization=0.88` để chừa VRAM cho runtime;
- `--max-num-seqs=80`: ≥ 70 nên không bao giờ làm request phải chờ (mỗi conversation
  tối đa một request đang chạy; replay đo được trung bình ~1.8 request đang chạy);
- `--max-num-batched-tokens=4096`;
- `dtype=auto`, không trộn quantization vào baseline đầu tiên;
- pinned model revision và seed.

Launcher tự đặt `VLLM_USE_FLASHINFER_SAMPLER=0`. Với wheel vLLM 0.25.1 trên
Ada, FlashInfer sampler có thể rơi vào JIT path đòi full `nvcc`; torch sampler
là path đã chạy thành công trên stock Ubuntu WSL2 của máy này.

## Kết quả đã kiểm chứng và cách diễn giải

Full A/B/A ngày 2026-08-02 đã hoàn tất 1,260/1,260 requests, zero failures.
Block này chạy bằng harness **trước** khi có warm-up:

| Stage | Mean TTFT | Median TTFT | Mean TPOT | Quoted-formula ERS |
|---|---:|---:|---:|---:|
| R0 | 31.579 ms | 16.783 ms | 4.138 ms | 65.832 |
| B · prefix cache | 19.729 ms | 17.738 ms | 3.943 ms | 70.184 |
| R0′ | 17.743 ms | 16.344 ms | 3.973 ms | 70.460 |

Theo mean, B trông nhanh hơn R0 11.8 ms và R0′ trông như bị “drift”. Raw data
cho thấy cả hai là **một hiện tượng**: năm request đầu của R0 mất 897–1,082 ms
TTFT vì server vừa khởi động JIT Triton kernels trong lúc phục vụ chúng (log vLLM
ghi `JIT compilation during inference`). Median — không nhạy với năm request đó —
cho thấy B chậm nhất. Median paired delta B−R0 là +1.35 ms trong khi mean paired
delta là −11.8 ms.

Phân tích lại, bỏ 10 arrivals đầu của mỗi stage (ngưỡng chọn sau khi đã xem data,
nên chỉ dùng để chẩn đoán):

| Paired delta, n = 410 | TTFT | TPOT |
|---|---:|---:|
| R0′ − R0 (drift) | −0.31 ms, CI [−0.71, +0.004] · uncertain | +0.002 ms · uncertain |
| B − R0 (prefix cache) | +0.67 ms, CI [0.27, 1.00] · slower | −0.045 ms, CI [−0.085, −0.019] · faster |
| B − R0′ | +0.97 ms, CI [0.70, 1.24] · slower | −0.048 ms · faster |
| **ΔERS, B − trung điểm R0/R0′** | **+0.14, CI [−0.02, +0.36] · uncertain** | |

Kết luận vẫn là **không promote**, nhưng lý do khác với bản đầu. Không phát hiện
drift — dù CI của R0′−R0 vẫn cho phép drift tới ~0.7 ms, nên kết luận TTFT dựa
vào việc B chậm hơn **cả** R0 lẫn R0′. Prefix caching làm TTFT chậm hơn ở turns
1–4 và nhanh hơn ở turns 5–6; phạt ở turn 1, nơi chưa có gì để reuse, có thể đến
từ mode `align` đi kèm hơn là từ caching. TPOT nhanh hơn chút ít nhưng median
paired delta ≈ 0, nên phần đó nằm ở tail. Score gate cho ΔERS **uncertain**.

Block cũ còn dùng live history, nên chỉ turn 1 là prompt giống hệt giữa các stage.
Harness đã được sửa (warm-up, canned history, fixed output length, prompt hash,
score gate); cần chạy lại block trên GPU không có ứng dụng khác trước khi quyết định.

Harness cũ còn một lỗi đo nhỏ: `completion_tokens` có tính token EOS, nhưng TPOT
kết thúc ở content chunk cuối, trước finish chunk của EOS. Với 322–328 request dừng
bằng EOS mỗi stage, TPOT thấp khoảng 2% (mean R0 4.138 → ~4.208 ms). Lỗi này
như nhau giữa các stage nên không đổi chiều so sánh, và đã được sửa.

Summary máy đọc được:
[`data/rtx4080-verified-aba-summary.json`](data/rtx4080-verified-aba-summary.json).

## Logic cho bước tối ưu tiếp theo

Sau mỗi A/B/A block:

1. Có request fail hoặc output sai → sửa correctness, không đọc performance.
2. Mean và median paired delta ngược dấu → xem các request chậm nhất trước
   (cold start, tail); đừng gọi đó là gain hay drift.
3. R0′ drift lớn → ổn định nhiệt độ/clocks/background load và repeat block.
4. CI đi qua 0 → candidate chưa thắng noise; repeat trước khi thêm flag khác.
5. TTFT và TPOT đi ngược chiều → tách queue/prefill khỏi decode để profile.
6. Signal ổn định → dùng Nsight Systems tìm critical path.
7. Chỉ dùng Nsight Compute với kernel đã chọn.
8. Chỉ viết/fuse kernel khi measured contribution có thể vượt noise floor.

## Troubleshooting theo thứ tự

### `Ubuntu-22.04` không tồn tại hoặc VERSION không phải 2

```powershell
wsl --list --verbose
wsl --set-version Ubuntu-22.04 2
```

### `nvidia-smi` không chạy trong WSL

Update Windows NVIDIA driver và chạy `wsl --update`. Không cài Linux display
driver trong WSL.

### PowerShell chặn `.ps1`

Chỉ bypass cho process hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\rtx4080_bootstrap.ps1 -Run smoke
```

### `-DoctorOnly` báo environment chưa tồn tại

Đúng behavior: chạy setup trước.

```powershell
.\scripts\rtx4080_bootstrap.ps1
.\scripts\rtx4080_bootstrap.ps1 -DoctorOnly
```

### OOM trong graph capture

Đóng ứng dụng dùng GPU. Nếu vẫn OOM, copy baseline args thành config mới và thử
`--gpu-memory-utilization=0.82`. Không trộn kết quả eager và graph trong cùng
baseline family.

### Triton báo thiếu C compiler

Chạy lại setup. Script cài portable Zig compiler wrapper nếu không có
`gcc`/`clang`.

### FlashInfer yêu cầu `nvcc`

Chạy qua `rtx4080_lab.py`/PowerShell launcher. Nếu serve thủ công, export:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### Windows checkout thay đổi nhưng WSL chưa thấy

Chạy lại bootstrap không kèm `-DoctorOnly` (ví dụ `.\scripts\rtx4080_bootstrap.ps1`
hoặc `-Run smoke`). Setup luôn rsync Windows source sang WSL trước khi run, rồi
chạy lab từ chính tree vừa sync, và giữ nguyên các result directories cũ.
`-DoctorOnly` không sync.
