# Paper results — dữ liệu chứng minh số liệu trong luận văn

Các CSV tổng hợp (aggregated) sinh bởi `main.py --aggregate-only --final-paper`
từ các shard huấn luyện trên Kaggle. Mỗi `run_meta.json` ghi `source_sha`
(hash cây nguồn) khớp tag `paper-freeze-v1` (`b83e6976…`), `config_sha`, danh
sách seed và môi trường chạy — bảo đảm số liệu tái lập được từ đúng bản mã đã khóa.

## main_N32/ — thí nghiệm chính (5 thuật toán × 8 seed, N=32)
- `algorithm_comparison.csv` — sum-rate, QoS, R_c, P_c/Pmax mỗi thuật toán
- `significance.csv` — paired t-test + Holm (MADDPG vs TD3/TD3-Matched/DDPG/PPO)
- `latency_cpu.csv` — độ trễ suy luận CPU một luồng (2000 lần gọi)
- `ablation.csv` — loại trừ STAR-RIS / công suất
- `sumrate_vs_power.csv` — quét P_max
- `model_complexity.csv` — số tham số (MADDPG vs TD3-Matched khớp)
- `ao_local_search*.csv` — mốc tham chiếu Hybrid AO Local Search
- `results_raw.csv` — dữ liệu thô (tidy) mọi metric/seed/scenario
- `completed_runs.csv`, `run_meta.json` — provenance

## scalability/ — quét N ∈ {16,32,64,96,128} (MADDPG vs TD3 × 8 seed)
- `scalability_summary.csv` — bảng tổng: sum-rate + p_Holm theo N (kết luận: tương đương mọi N)
- `N<N>/` — algorithm_comparison, significance, provenance mỗi N

## Kết luận chính (trung thực)
MADDPG **tương đương thống kê** với TD3/TD3-Matched ở mọi N (p_Holm ≥ 0,53),
chỉ vượt PPO có ý nghĩa; không tìm thấy lợi thế phân rã đa tác tử. Chi tiết
trong `latex_thesis/` (Chương 4 + Kết luận).
