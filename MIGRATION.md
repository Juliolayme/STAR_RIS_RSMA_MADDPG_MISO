# MIGRATION - Refactor 2026-07-16 theo review.txt

Tai lieu nay liet ke moi thay doi pha vo tuong thich (config, metric, API,
hanh vi mac dinh), danh sach thi nghiem bat buoc chay lai, va cac claim trong
luan van da duoc viet lai. Doc kem AUDIT_REPORT.md (mo ta loi goc) va
KAGGLE_RERUN_CHECKLIST.md (quy trinh chay lai).

## 1. Thay doi hanh vi mac dinh (QUAN TRONG NHAT)

| Truoc | Sau |
|---|---|
| Kenh co dinh 50 step/episode (static block) | `env_formulation: dynamic_mdp` - moi step la mot coherence block, small-scale tien hoa Gauss-Markov voi `channel_rho` = 0.95; thu tu: observe h_t -> act -> reward tren h_t -> evolve -> next_obs chua h_{t+1} |
| Geometry co dinh cho ca run | `resample_positions_on_reset: true` (moi episode mot geometry) |
| Khong co switching cost | Phase/power/beta switching cost (chuan hoa theo chieu, = 0 tai t = 0) |
| RNG chung cho moi thu | RNG streams tach: geometry / channel / misc(random-RIS); channel doc lap voi action |
| Reward: -lambda*sum(d_k^2) + bonus + power penalty chet | `alpha*R_sum/R_ref - sum_k lambda_k*c_k - 0.5*w*sum_k max(c_k,0)^2 - switching costs` voi c_k = R_min - R_k (co dau); bonus tat mac dinh; power penalty XOA (softmax luon cho tong = P_max) |
| lambda vo huong, heuristic nhan/chia theo nguong satisfaction | lambda_k per-user, projected dual gradient tren EMA cua c_k co dau |
| Chon best.pt theo raw training return | Lexicographic tren VALIDATION scenarios: feasible iff max_k(R_min - mean R_k) <= `model_select_constraint_tolerance`; feasible -> max sum-rate; infeasible -> min max-violation -> min mean-violation -> max sum-rate |
| Obs da chuan hoa (statistics troi) trong replay | Replay luu RAW obs; agent so huu normalizer, normalize trong select_action/learn; freeze sau `obs_norm_freeze_after_env_steps` |

Che do cu van tai lap duoc voi `env_formulation: static_block`
(physics kiem chung boi tests/test_legacy_regression.py so voi golden fixture).

## 2. Config key: cu -> moi

| Key cu | Key moi / thay the |
|---|---|
| (khong co) | `env_formulation`, `channel_rho`, `resample_positions_on_reset` |
| `qos_lambda_min`, `qos_lambda_max` | `dual_lambda_max` (chieu xuong [0, max]) |
| `qos_target_satisfaction`, `qos_lambda_increase`, `qos_lambda_decrease` | XOA - thay bang `dual_lr`, `dual_ema` |
| `qos_lambda_freeze_fraction` | `two_stage_dual_freeze_fraction` (ten moi nhan manh day la heuristic 2 pha, khong phai co che hoi tu) |
| `qos_penalty_type`, `reward_beta` | XOA - reward co dinh dang Lagrangian surrogate |
| `reward_gamma` (power penalty) | XOA (so hang chet) |
| `reward_qos_bonus` luon bat | `enable_qos_shaping_bonus: false` (bonus chi cho ablation) |
| (hardcode K*5) | `reward_rate_reference` |
| (khong co) | `augmented_penalty_weight` (w; ten tranh nham voi channel_rho) |
| (khong co) | `phase_switching_cost`, `power_switching_cost`, `beta_switching_cost` |
| (khong co) | `obs_include_ris_state`, `obs_norm_freeze_after_env_steps`, `analytical_phase_min_direct`, `star_ris_hardware_model` |
| `evaluation.seeds` | `config/seed_split.v2.yaml` (validation 11..55; 9101..9505 la legacy da bi xem; opened v1 test 70001..70005; fresh locked v2 test 81011,81023,81041,81071,81101), verified by the SHA-256 registered in `config/config.yaml` |
| (khong co) | `training.model_select_constraint_tolerance`, `evaluation.ao_local_search_max_n` |

## 3. Metric: cu -> moi

| Ten cu (mo ho) | Ten moi (tuong minh) |
|---|---|
| `qos_satisfied` (thuc chat: TAT CA user dat) | `all_users_qos_satisfied` |
| `per_user_satisfied_frac` | `user_qos_fraction` = (1/K) sum_k 1[R_k >= R_min] |
| `qos_prob` (trong log/bang - lay tu all-users) | `user_qos_fraction` HOAC `all_users_qos_prob`, ghi ro tung cho |
| (khong co) | `min_user_rate`, `mean_qos_deficit`, `max_qos_deficit`, `per_user_qos_satisfied`, `qos_constraint_signed` |
| `qos_lambda` | `qos_lambda_mean` + `qos_lambda_vec` (log tung lambda_k) |
| `reward_qos`, `reward_pwr` | `reward_dual`, `reward_aug`, `reward_switch` |

LUU Y NOI DUNG: con so 0.39-0.45 trong ket qua cu la xac suat TAT CA 4 user
dong thoi dat QoS (all_users_qos_prob), KHONG phai ty le user dat nhu luan van
cu mo ta. Luan van da duoc sua tuong ung.

## 4. API thay doi

- `MADDPG/DDPGAgent/TD3Agent/PPOAgent`: them `attach_obs_normalizer(s)`,
  `freeze_normalizers()`, `save_inference()/load_inference()`,
  `weights_state_dict()/train_state_dict()/replay_state()`. Ham
  `select_action(s)` nhan RAW obs (normalize ben trong). Cac ham train driver
  KHONG con truyen obs_norm ra ngoai.
- `evaluate_agent(...)`: bo tham so `obs_norm` (nhan nhung bo qua), them
  `scenarios=` (ScenarioBank playback); tra ve metric ten moi + per-episode
  values + `c_bar_per_user`.
- `env._bcd_optimize` -> `env._coarse_ao_grid`; `ris_mode="bcd"` -> `"ao_grid"`
  (alias cu con hoat dong voi DeprecationWarning).
- Moi: `env.observation_schema()/action_schema()/export_schema()`,
  `env.get_state()/set_state()`, `env/scenario_bank.py`,
  `experiments/checkpointing.py` (TrainingCheckpoint - resume chinh xac, chi
  luu tai ranh gioi episode, co replay buffer), `experiments/baselines_ao.py`
  (`AOHybridLocalSearch` - "Hybrid AO Local Search (SLSQP + projected
  gradient)", tham chieu local search, KHONG phai upper bound),
  `algorithms/complexity.py` (dem tham so + FLOPs + `matched_td3_hidden_sizes`).
- `main.py`: output vao `results_revised/`; them `--resume`, algo moi
  `td3_matched`; power sweep/ablation/diagnostics chay qua TAT CA training
  seeds; latency CPU/GPU hai bang rieng; paired statistics + Holm-Bonferroni
  (family chinh: MADDPG vs {TD3, DDPG, PPO} tren sum-rate, m=3).
- `plot_training_convergence` nhan ma tran [n_seeds, n_episodes]; vung bong =
  Student-t 95% CI qua seeds (KHONG con rolling std).

## 5. Thi nghiem BAT BUOC chay lai (ket qua cu khong con hop le)

Xem KAGGLE_RERUN_CHECKLIST.md de biet quy trinh chi tiet. Tom tat:

1. Main comparison: 8 training seeds x {MADDPG, DDPG, TD3, TD3-Matched, PPO},
   dynamic_mdp, danh gia tren test ScenarioBank khoa.
2. Power sweep (10-35 dBm) qua 8 seeds.
3. Ablation 8 cells (AO-Grid thay BCD).
4. Scalability N in {16, 32, 64, 96, 128} (retrain per N), MADDPG vs TD3
   paired + Holm.
5. Latency benchmark CPU (1 thread) va GPU (bang rieng).
6. Hybrid AO Local Search reference cho N = 16, 32.
7. Feasibility analysis cua R_min = 0.3 (AO local search / max-min tren nhieu
   scenario) truoc khi giu bat ky phat bieu QoS nao.

## 6. Claim luan van da doi (chi tiet trong cac file .tex)

| Claim cu | Claim moi |
|---|---|
| "Dam bao QoS tung nguoi dung" | "Toi uu can bang throughput-QoS bang penalty/Lagrangian surrogate; rang buoc ky vong E[R_k] >= R_min" |
| "Primal-dual, hoi tu saddle point" | "Projected dual-gradient tren c_k co dau; lich 2 pha (two-stage dual freeze) la heuristic, khong co bao dam hoi tu" |
| "BCD - can tren toi uu" | "AO-Grid: heuristic luoi tho; tham chieu manh hon: Hybrid AO Local Search; khong phai can tren" |
| "Vung bong = CI 95% tren 8 seed" | Mo ta dung phuong phap CI qua seed (hinh cu phai ve lai) |
| "Latency do tren cung CPU" | Ghi trung thuc thiet bi cu (device=auto/T4); bang moi CPU/GPU rieng |
| "Thich nghi truc tuyen voi kenh thay doi" | "Suy luan thoi gian thuc dieu kien theo CSI (real-time CSI-conditioned inference)"; dong thoi formulation dynamic_mdp lam claim nay co co so sau khi chay lai |
| "QoS 0.39-0.45 = ty le UE dat" | "= xac suat tat ca K UE dong thoi dat R_min (all_users_qos_prob)" |
| "MADDPG vuot TD3" (khong dieu kien) | "Tuong duong thong ke tai N=32; khac biet tu N>=64 CHI khi paired test (Holm) ung ho sau khi chay lai" |

## 7. Ngoai pham vi dot nay (viec tuong lai)

Differential-evolution reference; information-matched/shared-critic MADDPG
variants (ngoai TD3-Matched); imperfect CSI / phase quantization / coupled
phase / insertion loss; Dockerfile; GitHub Actions CI; CITATION.cff;
AO local search cho N > 32; luu replay buffer trong moi checkpoint dinh ky
(hien chi trong TrainingCheckpoint tai ranh gioi episode).
