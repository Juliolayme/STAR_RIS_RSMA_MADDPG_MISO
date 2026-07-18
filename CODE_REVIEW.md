# Báo cáo Review Source Code — STAR-RIS RSMA MADDPG (MISO)

Ngày review: 2026-07-18. Phạm vi: toàn bộ mã Python trong gói
`STAR_RIS_RSMA_MADDPG_MISO_ready.zip` (env, algorithms, networks, utils,
experiments, main.py, tests) ở góc nhìn senior research engineer:
đúng đắn vật lý kênh / RSMA, đúng đắn thuật toán RL (MADDPG/DDPG/TD3/PPO),
tính tái lập (reproducibility) và chất lượng test.

**Kết quả tổng thể:** codebase được viết cẩn thận, nhiều cơ chế chống lỗi
nghiên cứu (ScenarioBank playback, khóa test bank, dual-reward recompute,
provenance SHA). Tìm thấy **1 lỗi runtime thật trong env**, **1 nhóm lỗi lớn
trong test suite do MISO migration chưa hoàn tất** (63/115 test fail), và một
số điểm không nhất quán tài liệu. Tất cả đã được sửa trong commit này.
Trạng thái test sau khi sửa: **112 passed, 3 skipped, 0 failed**.

---

## 1. Lỗi đã sửa (kèm commit này)

### P0-A. `env/star_ris_env.py` — NaN scale khi `K_r = 0` xóa trắng quan sát h_eff
`_obs_parts_base()` chuẩn hóa khối quan sát `h_eff` bằng
`alpha_d[:K_r].mean()`. Khi cấu hình hợp lệ `num_users_reflection = 0`
(toàn bộ user ở vùng T — constructor cho phép `0 <= K_r <= K`), mean của
slice rỗng là **NaN**; sau đó `_finite()` âm thầm thay NaN bằng 0, nghĩa là
**toàn bộ khối Re/Im của h_eff trong observation của mọi agent bị ghi đè
thành 0** — agent mất hoàn toàn thông tin kênh hiệu dụng mà không có bất kỳ
cảnh báo/lỗi nào (đã tái hiện được bằng thực nghiệm). Các sweep `k_sweep`
hoặc ablation thay đổi phân bố user có thể rơi vào trường hợp này.

**Sửa:** fallback về mean của toàn bộ `alpha_d` khi `K_r = 0`.

### P0-B. Test suite bị bỏ lại phía sau MISO migration — 63/115 test fail
Env đã migrate sang MISO và ép `num_bs_antennas >= 2`, nhưng
`tests/conftest.py` vẫn dùng `num_bs_antennas: 1`, khiến **63 test fail ngay
tại constructor** (`ValueError: MISO formulation requires num_bs_antennas >= 2`)
— tức là gần như toàn bộ lưới an toàn (physics, schema, dual update, QoS
metrics, checkpoint, shard flow, PPO consistency…) không chạy được và mọi
regression thật sự sẽ không bị phát hiện.

**Sửa:** `conftest.base_env_cfg` chuyển sang `num_bs_antennas: 4` (khớp
default MISO trong `config/config.yaml`).

### P0-C. Hai test còn dùng shape SISO bên trong
Sau khi sửa P0-B, còn 2 test dùng công thức/shape SISO cũ:
- `test_channel_dynamics.py::test_reward_computed_on_observed_channel_then_evolve`:
  tái dựng h_eff bằng `Σ conj(g)·coeff·G` (vô hướng SISO) → lỗi broadcast.
  Sửa thành cascade MISO `h_d[k] + Σ_n conj(g_k[n])·coeff_n·G[n,:]`.
- `test_env_physics.py::test_analytical_fallback_when_direct_link_vanishes`:
  gán `env._h_d = zeros(K)` (SISO) và kiểm tra alignment theo từng phần tử
  G. Sửa thành `zeros((K, M))` và kiểm tra alignment trên tổng theo anten
  `Σ_m G[n,m]` — đúng với heuristic `_analytical_phases()` của MISO.

### P0-D. Golden fixture legacy không thể replay trên env MISO
`tests/fixtures/golden_static_block.npz` được sinh trước MISO migration:
M=1, layout action cũ (softmax power, ví dụ dim 33 thay vì 38), `h_d` shape
`(K,)` thay vì `(K, M)`. Docstring của env vẫn tuyên bố `static_block` được
giữ để "reproduce golden fixture bit-exact" — **tuyên bố này hiện không còn
đúng**: env từ chối M=1 ngay tại constructor nên 3 test regression fail bằng
ValueError thay vì báo skip có chủ đích.

**Sửa:** test skip tường minh với lý do rõ ràng khi fixture có M < 2.
**Việc cần làm tiếp (không thể tự động):** chạy lại
`tests/make_golden_fixture.py` dưới mô hình MISO để khôi phục regression gate,
hoặc gỡ hẳn formulation `static_block` nếu không còn nhu cầu đối chiếu.

### P1-E. Tài liệu/comment mâu thuẫn SISO vs MISO
- Docstring đầu `env/star_ris_env.py` mô tả "SISO, M = 1, h_eff scalar" và
  "Agent 0 size = (K+1)+K" trong khi code là MISO beamforming
  (`2·M·(K+1)+K`). → cập nhật docstring đúng với code.
- `config/config.yaml`: `num_bs_antennas: 4  # M = 1 (SISO scenario)` →
  sửa comment.
- `main.py::_write_report`: report tự động vẫn in "SISO downlink" và
  "M = 1 (SISO BS)" trong phần Limitations → in đúng M từ config.

---

## 2. Những điểm đã kiểm tra và XÁC NHẬN ĐÚNG

- **Vật lý RSMA MISO** (`_rsma_rates`): SINR chung
  `|h_k^H w_c|² / (Σ_j |h_k^H w_j|² + σ²)` — private chưa SIC nằm đủ trong
  mẫu số; rate chung lấy `min_k`; SINR riêng sau SIC loại đúng thành phần
  common và thành phần của chính user. Đồng nhất thức
  `R_sum = R_c + Σ R_p` và `Σ per_user = R_sum` được test.
- **Cascade STAR-RIS** (`_effective_channels`): `Σ_n conj(g_k[n])·φ_n·G[n,:]`,
  ES: `β_r + β_t = 1`, chọn hệ số R/T theo vùng user — đúng.
- **Chiếu công suất beamformer**: chuẩn hóa Frobenius toàn bộ W về đúng
  `P_max` (dùng full power — quy ước phổ biến, nhất quán với các baseline AO
  dùng ràng buộc đẳng thức simplex).
- **Thứ tự MDP động**: observe h_t → act → reward trên h_t → kênh evolve
  (Gauss-Markov `ρ·h + √(1-ρ²)·ε`) → next_obs trên h_{t+1}; RNG kênh tách
  khỏi RNG policy; ScenarioBank có đủ `max_steps` innovation, chỉ số 0..T-1
  không tràn (kể cả `advance_to_next_block` cho AO baseline).
- **MADDPG chuẩn CTDE (Lowe et al. 2017)**: critic tập trung trên
  (joint obs, joint act); khi update actor i, hành động của agent j≠i lấy từ
  policy hiện tại và detach; reward hợp tác broadcast; target soft-update.
- **Dual/Lagrangian**: reward tách `base_reward` + `c_gap` để recompute
  reward theo λ hiện hành khi sample replay (tránh critic học trên hỗn hợp
  reward cũ); PPO giữ λ cố định trong một rollout, update sau khi consume —
  đúng với on-policy; DualUpdater dùng gap CÓ DẤU (λ giảm được khi có slack),
  chiếu về [0, λ_max]; two-stage freeze đúng như mô tả.
- **Observation normalizer**: buffer lưu RAW obs, normalize đúng một lần tại
  select_action/learn; freeze sau `obs_norm_freeze_after_env_steps`; PPO lưu
  đúng vector đã normalize dùng để tính log_prob/value (old_log_prob khớp
  đến sai số số học — có test riêng).
- **Thống kê** (`utils/metrics.py`): đã đối chiếu số học `_student_t_sf`
  với `scipy.stats.t.sf` trên lưới df∈[1,100], t∈[0,10]: sai số tuyệt đối
  lớn nhất **1.3e-4** — đủ chính xác cho p-value báo cáo. Welch t-test khớp
  scipy tới 6 chữ số ở df nhỏ. Holm–Bonferroni, paired permutation
  (enumeration đủ 2ⁿ khi n≤20), CI Student-t đều đúng.
- **Model selection**: lexicographic (feasible → max sum-rate; infeasible →
  min max-violation → min mean-violation → max sum-rate) chỉ trên validation
  bank; test bank chỉ được build ở job aggregate — đúng như thiết kế chống
  data leakage.
- **Provenance/shard**: source_sha/config_sha/checkpoint_sha nhất quán,
  từ chối trộn shard khác source; atomic write manifest.

## 3. Nhận xét không chặn (không sửa code, cần biết)

1. **Truncation = done**: các training loop đặt `done = terminated or truncated`
   và cắt bootstrap tại time-limit (`γ·(1-done)·Q_next`). Với episode dài cố
   định 50 bước, đây là bias nhỏ và là simplification phổ biến, nhưng nếu muốn
   đúng chuẩn nên bootstrap qua truncation (chỉ cắt khi terminated).
2. **`ObservationNormalizer.enabled = False`** vẫn áp z-score với mean=0/var=1
   và clip ±10 — gần identity nhưng không phải identity tuyệt đối nếu obs thô
   vượt 10.
3. **`equal_power_mode`** dùng `self._h_eff` của bước trước (kênh hiện tại +
   cấu hình RIS cũ) làm hướng matched-filter — xấp xỉ hợp lý cho baseline,
   nên ghi chú trong luận văn.
4. **Nhánh `continue` khi reward non-finite** trong các training loop bỏ qua cả
   việc cập nhật `obs = next_obs` (obs bị lệch 1 bước). Thực tế không thể xảy
   ra vì env đã clamp reward về hữu hạn — nhưng nếu ai đó gỡ clamp thì đây là
   bẫy tiềm ẩn.
5. **`TrainingCheckpoint.load` thiếu sidecar replay** (sau
   `--drop-replay-after-train`) sẽ KeyError `"replay"` khi restore thay vì báo
   lỗi thân thiện; manifest đã đánh dấu `resumable_training=false` nên rủi ro
   thấp.
6. **AO-Grid** (`_coarse_ao_grid`) bỏ qua tham số `action` của agent hoàn toàn
   (by design) nhưng vẫn chạy qua `_decode_action` phần đầu — tốn vài phép
   tính vô ích mỗi bước, không sai.
7. Golden fixture cần **regenerate dưới MISO** (xem P0-D) nếu muốn giữ
   regression gate cho `static_block`.

## 4. Trạng thái kiểm thử

| | Trước review | Sau khi sửa |
|---|---|---|
| Passed | 52 | **112** |
| Failed | **63** | 0 |
| Skipped | 0 | 3 (golden fixture SISO, skip có lý do) |

Chạy bằng: `python -m pytest tests/ -q` (PyTorch 2.13 CPU, Python 3.11).
