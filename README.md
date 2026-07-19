# STAR-RIS RSMA MADDPG - MISO hardening branch

Mã nguồn nghiên cứu tối ưu phân bổ tài nguyên cho mạng STAR-RIS hỗ trợ RSMA
bằng DRL. Bản hiện tại dùng BS nhiều anten (`M=4` mặc định), beamforming phức
cho common/private streams, và SINR MISO đầy đủ.

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

## Chạy nhanh

```bash
python3 -m pytest -q
python3 main.py --config config/smoke.yaml --quick
```

## Chạy shard Kaggle

Mỗi job nên chạy một cặp algorithm-seed:

```bash
python3 main.py --config config/config.yaml --train-only --algos td3_matched --seeds 1000 --run-id miso_q1
```

Sau khi đủ shard, gộp bằng:

```bash
python3 main.py --aggregate-only --load-shards results_revised/shards/miso_q1
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
