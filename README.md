# STAR-RIS RSMA MADDPG - MISO hardening branch

Mã nguồn nghiên cứu tối ưu phân bổ tài nguyên cho mạng STAR-RIS hỗ trợ RSMA
bằng DRL. Bản Structured-BS V2 dùng BS nhiều anten (`M=4` mặc định), MISO
RSMA đầy đủ và một lớp giải mã vật lý RZF/MRT thay cho việc học trực tiếp mọi
hệ số beamforming phức.

## Trạng thái kết quả

Các CSV trong `paper_results/` là snapshot lịch sử từ cây source cũ và không
được dùng để tuyên bố kết quả MISO/Q1. Chúng chỉ còn giá trị tham khảo/audit cho
đến khi chạy lại toàn bộ ma trận thí nghiệm bằng source và config đã freeze.

Kết luận hiện tại nên viết thận trọng:

- MADDPG, TD3, TD3-Matched, DDPG và PPO đều phải được rerun trên MISO.
- Không tuyên bố MADDPG vượt TD3/TD3-Matched nếu chưa có paired statistical
  tests trên locked MISO test bank.
- Các smoke run chỉ xác nhận code chạy được, không phải bằng chứng khoa học.

## Điểm MISO/Q1 đã khóa bằng test

- Quy ước kênh: `h_eff,k` được lưu sao cho
  `h_eff,k^H w = h_d,k^H w + g_k^H Phi_k G w`.
- Observation không chứa `prev_reward`, tránh state cũ phụ thuộc lambda cũ khi
  replay reward được recompute theo lambda hiện hành.
- Time-limit truncation kết thúc episode nhưng không cắt bootstrap target; chỉ
  `terminated` mới được lưu như terminal.
- Ablation `disable_qos_penalty` tắt cả dual lambda, shaping bonus và augmented
  quadratic penalty.
- Common-rate split dùng inverse-tanh logits trước softmax để có thể đi sát
  biên simplex.
- Golden regression fixture đã được regenerate cho MISO (`M=4`).
- Analytical phase prior is derived directly from `h_d^H q + g^H Phi G q` and covered by a nonzero-direct-link oracle.
- MADDPG critics use one canonical global state; overlapping local observations are not concatenated.
- PPO uses distinct terminal/bootstrap and episode-boundary masks in GAE.
- Primary V2 config uses `bs_action_mode: structured_rzf` and a bounded residual around the corrected analytical RIS phase prior.
- The BS actor learns stream powers, common-rate split, common-beam weights and a low-dimensional MRT/RZF residual; it no longer emits raw complex beamformers.

## Structured-BS V2 compatibility note

BS action dimension changed from 44 to 14 (default M=4, K=4), and the flat
action dimension changed from 140 to 110. **All old checkpoints, replay files,
40 training shards and the opened v1 final-test package are incompatible with
V2.** Retrain every algorithm/seed and use `config/seed_split.v2.yaml`. See
[`STRUCTURED_BS_V2.md`](STRUCTURED_BS_V2.md).

## Chạy nhanh

```bash
python3 -m pytest -q
python3 main.py --config config/smoke.yaml --quick
```

## Chạy shard Kaggle

Mỗi job nên chạy một cặp algorithm-seed:

```bash
python3 main.py --config config/config.yaml --train-only --algos td3_matched --seeds 1000 --run-id miso_q1_structured_bs_v2
```

Sau khi đủ shard, gộp bằng:

```bash
python3 main.py --aggregate-only --load-shards results_revised/shards/miso_q1_structured_bs_v2
```

Không commit hoặc in nội dung `kaggle.json`; đó là credential nhạy cảm.

## Cấu trúc

```text
config/              cấu hình smoke, pilot, paper
env/                 môi trường STAR-RIS RSMA MISO
algorithms/          MADDPG, DDPG, TD3, PPO
experiments/         train/evaluate/checkpointing
tests/               unit, physics oracle, regression, smoke tests
latex_thesis/        bản thảo luận văn và bảng/hình
paper_results/       historical snapshot, không phải kết quả MISO final
```
