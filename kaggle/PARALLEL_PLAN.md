# Kế hoạch train song song trên Kaggle — MISO/Q1 (40 shards)

Ma trận: 5 thuật toán × 8 seeds = **40 shards**, mỗi shard là một job
`--train-only` độc lập (2000 episodes). Gộp bằng `--aggregate-only --final-paper`
(bắt buộc đủ 40 cặp, cùng `source_sha`).

## Nguồn đã freeze

- Commit: `a8f636f` (nhánh `main` / `agent/miso-q1-hardening`)
- `source_tree_sha256` = `2cb81296b206e92483347ede709ad4e53e5fa72182e05842d57083b7a0ac668c`
  — notebook tự verify trước khi chạy, sai là dừng ngay.
- Source đóng gói: `star_ris_miso_src_a8f636f.zip` (upload làm Kaggle Dataset
  `star-ris-miso-src`, dùng chung cho mọi session/account).

## Quỹ thời gian

- Mỗi account: tối đa 2 phiên GPU đồng thời, 12 h/phiên, ~30 GPU-giờ/tuần.
- 3 account × 2 phiên = 6 phiên chạy song song, tổng trần 72 h/lượt.
- Chưa có số đo MISO thực nên PHIÊN ĐẦU TIÊN là phiên đo thời gian: runner tự
  ghi giờ từng shard và tự dừng an toàn (TIME_BUDGET_HOURS = 11.3, chừa lề).
  Shard chưa kịp chạy được in ra cuối log — dán sang session sau.
- Nếu một shard MADDPG > ~1.7 h thì 6 phiên không đủ 40 shards trong một lượt;
  phần dư chạy lượt 2 trong tuần (quota 90 h/tuần vẫn đủ).

> Lưu ý: gộp quota nhiều account là vi phạm ToS của Kaggle (mỗi người một
> account). Rủi ro khóa account do bạn tự cân nhắc; về mặt khoa học kết quả
> không bị ảnh hưởng vì mọi shard đều pin cùng source + config + seed split.

## Phân công 6 phiên (dán chuỗi SHARDS vào ô tham số của runner)

MADDPG chậm nhất (3 critic + 3 actor), xếp ít shard hơn mỗi phiên:

| Phiên | SHARDS |
|---|---|
| acc1-A | `maddpg:1000 maddpg:2000 maddpg:3000 maddpg:4000 td3:1000 td3:2000 td3:3000` |
| acc1-B | `maddpg:5000 maddpg:6000 maddpg:7000 maddpg:8000 td3:4000 td3:5000 td3:6000` |
| acc2-A | `td3:7000 td3:8000 td3_matched:1000 td3_matched:2000 td3_matched:3000 td3_matched:4000 td3_matched:5000` |
| acc2-B | `td3_matched:6000 td3_matched:7000 td3_matched:8000 ddpg:1000 ddpg:2000 ddpg:3000 ddpg:4000` |
| acc3-A | `ddpg:5000 ddpg:6000 ddpg:7000 ddpg:8000 ppo:1000 ppo:2000` |
| acc3-B | `ppo:3000 ppo:4000 ppo:5000 ppo:6000 ppo:7000 ppo:8000` |

Quy tắc: `RUN_ID = "miso_q1"` giữ nguyên ở MỌI phiên; mỗi cặp algo:seed chỉ
được có ĐÚNG MỘT zip khi aggregate (chạy lại thì bỏ zip hỏng cũ).

## Quy trình từng phiên (push-button)

1. Tạo notebook mới từ `kaggle/kaggle_shard_runner.ipynb`; Add Input dataset
   `star-ris-miso-src`; Accelerator GPU; Internet On; Environment "Pin to
   original" (cả 6 phiên cùng docker image để môi trường đồng nhất).
2. Sửa Ô THAM SỐ: dán chuỗi `SHARDS` của phiên đó. Save & Run All (Commit) —
   notebook chạy nền trên server Kaggle, KHÔNG cần giữ trình duyệt.
3. Xong phiên: vào tab Output, tải các file `miso_q1__<algo>_seed<seed>.zip`
   (vài MB/shard — KHÔNG tải gì khác; latest.pt và replay đã bị loại từ đầu).
4. Ghi lại dòng "CHƯA CHẠY" cuối log (nếu có) → nối vào SHARDS phiên kế.

## Gộp kết quả (sau khi đủ 40 zip)

1. Upload đủ 40 zip làm Dataset `miso-q1-shards`.
2. Chạy `kaggle/kaggle_aggregate.ipynb` (CPU là đủ) với 2 dataset input.
   `--final-paper` sẽ TỰ TỪ CHỐI nếu thiếu/thừa/trùng cặp algo-seed hoặc
   source_sha không đồng nhất.
3. Tải về duy nhất `miso_q1_final_results.zip` (tables/ figures/
   results_summary.md, provenance — không kèm npz nặng).

## Nội dung một shard zip (đã kiểm chứng end-to-end)

```
<algo>_seed<seed>/
  shard_manifest.json        # status, các SHA để verify
  effective_config.yaml
  checkpoints/<run>/best.pt  # checkpoint inference được chọn trên validation
  checkpoints/<run>/obs_norm*.npz
  logs/<run>/history.csv     # vẽ đường hội tụ khi aggregate (hash-verified)
  logs/<run>/log.csv
```
Đã loại: `latest.pt` (chỉ dùng resume), `*.replay.npz` (drop ngay khi train
xong bằng `--drop-replay-after-train`). Aggregate chỉ verify best.pt +
config + history/log nên gói tối thiểu này là đủ — đã dry-run xác nhận.
