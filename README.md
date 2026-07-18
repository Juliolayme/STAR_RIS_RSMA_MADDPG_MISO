# STAR-RIS RSMA MADDPG — MISO edition

> This repository uses a multi-antenna BS (`M=4` by default), complex common/private beamforming, and full MISO RSMA SINR coupling. See [MISO_MIGRATION.md](MISO_MIGRATION.md). Existing SISO paper-result files in the imported snapshot are historical and must not be presented as MISO results.

# DRL Resource Allocation in STAR-RIS Assisted RSMA Networks

Mã nguồn kèm luận văn thạc sĩ *"Tối ưu phân bổ tài nguyên sử dụng học tăng cường
sâu trong mạng STAR-RIS hỗ trợ RSMA"* (Nguyễn Duy Thanh, UTH, 2026).

MADDPG-CTDE ba tác tử (BS Power / RIS-Reflection / RIS-Transmission) với ba cải
tiến: residual phase (±π/4 quanh nghiệm giải tích), phần thưởng Lagrangian thích
nghi, và lịch trình hai pha **freeze-λ** (đóng băng hệ số phạt ở 45% cuối để
phần thưởng dừng → đường hội tụ phẳng kiểm chứng được).

## Kết quả chính (khớp Chương 4 luận văn)

- **Cấu hình chuẩn N=32** (8 seeds × 2000 ep, lặp lại 2 lần lệch <1%):
  MADDPG **3,41 ± 0,12 b/s/Hz** — vượt DDPG +73% và PPO +35% (Welch p < 1e-7),
  tương đương TD3; suy luận ≈1,3 ms/hành động (nhanh hơn BCD ~13×).
- **Scalability N=16→128** (10 seeds/điểm, 100 lần huấn luyện):
  lợi thế phân rã tăng theo N — gain MADDPG−TD3 từ −0,12 (N=16) lên **+0,83
  b/s/Hz (N=96)**, CI tách rời từ N≥64 (p < 1e-3); CI của TD3 nở ~5× trong khi
  MADDPG giữ ổn định.

## Cấu trúc

```
config/ env/ algorithms/ networks/ utils/ experiments/   # thư viện (Python 3.10+, PyTorch)
main.py                       # pipeline chạy local end-to-end (python main.py --quick để smoke test)
notebooks/
  star-ris-rsma-maddpg-v20.ipynb                # runner chính (N=32, 8 seeds) — chạy trên Kaggle
  star-ris-rsma-maddpg-v20_scalability_N.ipynb  # runner scalability (sweep N, resume per-seed)
results/
  main_8seed/       # 2 lần chạy 8-seed (frozen-v21, frozen-v21-2) + results_summary.md
  scalability_N/    # 5 notebook theo N + notebook merge + 5 CSV raw per-seed
latex_thesis/       # toàn văn luận văn (XeLaTeX/tectonic): chapters/ figures/ tables/ + main.pdf
```

## Chạy lại trên Kaggle

1. Upload toàn bộ repo này thành một Kaggle Dataset.
2. Mỗi job chỉ chạy một cặp algorithm-seed (CLI chủ động enforce vì
   DDPG/TD3/PPO chưa có checkpoint resume đầy đủ):
   `python main.py --config config/config.yaml --train-only --algos td3_matched --seeds 1000 --run-id paper_v1`.
3. Sau khi thu đủ shard, chạy
   `python main.py --aggregate-only --load-shards results_revised/shards/paper_v1`.
   Aggregator xác minh config/checkpoint SHA từ từng `shard_manifest.json`, không
   dùng `run_meta.json` của subset job. Xem toàn bộ ma trận hoàn tất trong
   `completed_runs.csv`.
4. Số liệu tái lập nhờ seeding tất định (`experiments/train.py::_set_seed`).

Danh sách seed chuẩn nằm duy nhất tại `config/seed_split.v1.yaml`; hash được
đăng ký và kiểm tra trong `config/config.yaml`. Final-test ScenarioBank chỉ được
tạo sau code/config freeze. Cấu hình paper mặc định bao gồm `td3_matched`.

## Ghi chú

- Lịch sử thí nghiệm trung gian (v14–v19, checkpoints/logs của run cũ) đã được
  dọn khỏi cây làm việc; xem `git log` nếu cần khôi phục.
- Siêu tham số duy nhất tại `config/config.yaml` (đã là bộ v20: freeze-λ=0,55,
  λ_max=13, residual ±π/4, 2000 episodes, 8 seeds).
