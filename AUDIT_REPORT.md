# AUDIT REPORT — STAR-RIS RSMA MADDPG (refactor theo review.txt)

Ngay lap: 2026-07-15. Trang thai cap nhat dan trong qua trinh sua.
Moi muc gom: mo ta loi, file/dong lien quan (truoc khi sua), cach kiem chung, trang thai.

Quy uoc trang thai: [PENDING] chua sua | [FIXED] da sua + test | [PARTIAL] sua mot phan (ghi ro) | [DOCUMENTED] khong sua code, chi sua tai lieu/claim.

---

## P0 — Loi bat buoc sua truoc khi nop

### P0-1. Bai toan chua phai MDP dong
- **Loi:** `max_steps=50`, `channel_block_steps=50` — kenh sinh o `reset()` va giu nguyen ca episode; action khong anh huong transition. 50 step = lap lai bai toan tinh tren cung mot channel realization. Mau thuan voi gamma=0.95, Bellman target, va claim "thich nghi truc tuyen".
- **File:** `env/star_ris_env.py` (reset/step, truoc refactor dong 122-124, 628-655), `config/config.yaml` (max_steps, channel_block_steps).
- **Kiem chung:** doc `_sample_channels` chi duoc goi trong `reset()` va khi `step_count % channel_block_steps == 0`; voi block=50=max_steps thi khong bao gio refresh trong episode.
- **Sua:** them `env_formulation: dynamic_mdp | static_block | contextual_bandit` (default dynamic_mdp). Dynamic: small-scale tien hoa Gauss-Markov h_{t+1} = rho*h_t + sqrt(1-rho^2)*eps sau khi reward duoc tinh tren h_t; switching cost lam action co anh huong qua thoi gian; obs chua trang thai RIS hien hanh (Markov). Contextual bandit: max_steps=1, gamma phai =0 (ValueError neu khac). Legacy static_block giu nguyen hanh vi cu, doi chieu golden fixture.
- **Test:** `tests/test_channel_dynamics.py`, `tests/test_legacy_regression.py`.
- **Trang thai:** [FIXED]

### P0-2. QoS metric trong code khac dinh nghia luan van
- **Loi:** code tao `frac_sat = mean(per_user >= R_min)` va `qos_satisfied = all users dat`; train/eval/notebook dung `qos_satisfied` (ALL users) nhung luan van dinh nghia "ty le UE dat nguong". So 0.39-0.45 thuc te la xac suat CA 4 user dong thoi dat QoS.
- **File:** `env/star_ris_env.py` (truoc refactor 671-674, 715, 723), `experiments/train.py` (94-104), `experiments/evaluate.py`, `main.py`, notebooks, luan van chuong 4.
- **Kiem chung:** grep `qos_satisfied` — duoc gan `bool(qos_viol_l1 < 1e-6)` (all users), nhung caption luan van noi "ty le UE".
- **Sua:** env tra ve ten tuong minh: `user_qos_fraction`, `all_users_qos_satisfied`, `min_user_rate`, `mean_qos_deficit`, `max_qos_deficit`, `per_user_qos_satisfied`, `per_user_rate`. Xoa ten mo ho `qos_satisfied`. Cap nhat toan bo pipeline + luan van (dinh nghia 0.39-0.45 = xac suat tat ca K UE dong thoi dat R_min).
- **Test:** `tests/test_qos_metrics.py`.
- **Trang thai:** [FIXED]

### P0-3. Khong the noi "dam bao QoS" khi ket qua ~40%
- **Loi:** luan van tuyen bo dam bao QoS tung nguoi dung trong khi metric bao cao ~0.39-0.45.
- **File:** `latex_thesis/chapters/01_mo_dau.tex`, `02_chuong_1.tex`, `05_chuong_4.tex`, `06_ket_luan.tex`.
- **Sua:** doi thanh "toi uu can bang throughput-QoS bang penalty/Lagrangian surrogate cua rang buoc QoS"; rang buoc trinh bay dang ky vong E[c_k] <= 0 (tuong duong E[R_k] >= R_min), khong phai bao dam tung step/tung user.
- **Trang thai:** [DOCUMENTED] (sua luan van; feasibility analysis bang solver manh ghi vao KAGGLE_RERUN_CHECKLIST.md nhu viec phai chay)

### P0-4. Co che lambda khong phai primal-dual chuan
- **Loi:** `_qos_lambda_update` la heuristic nhan/chia (`*1.02 / *0.97`) theo nguong satisfaction, khong phai projected dual gradient; khong co bao dam saddle point nhung luan van claim.
- **File:** `experiments/train.py` (truoc refactor 68-91), `config/config.yaml` (qos_lambda_increase/decrease/target_satisfaction).
- **Sua:** dual update chieu (projected) tung user, nhat quan voi reward qua cung ham rang buoc c_k = R_min - R_k:
  `ema_k = dual_ema*ema_k + (1-dual_ema)*mean_episode(c_k)`; `lambda_k = clip(lambda_k + dual_lr*ema_k, 0, dual_lambda_max)`.
  Reward: `alpha*R_sum/R_ref - sum_k lambda_k*c_k - 0.5*w*sum_k max(c_k,0)^2 - switching costs`.
  Freeze giu lai duoi ten `two_stage_dual_freeze_fraction`, mo ta la heuristic 2 pha, khong claim hoi tu. Bonus satisfaction chuyen sau co `enable_qos_shaping_bonus` (default false).
- **Test:** `tests/test_dual_update.py`.
- **Trang thai:** [FIXED]

### P0-5. "BCD upper bound" khong phai upper bound
- **Loi:** `_bcd_optimize` chi la grid search tho (1 beta chung cho N phan tu, 5 gia tri beta, 6 ty le P_c, private/common chia deu, khong QoS constraint, 20 vong lap gan nhu lap lai) nhung duoc goi "true optimization-based upper bound".
- **File:** `env/star_ris_env.py` (truoc refactor 328-388), `experiments/ablation.py` (9-15), luan van chuong 4.
- **Sua:** (1) doi ten trung thuc `_coarse_ao_grid` / nhan "AO-Grid", xoa moi tu "upper bound/optimum/can tren"; (2) them baseline manh hon `AOHybridLocalSearch` (SLSQP cho power/split + projected gradient cho beta vector per-element + phase, multi-start, monotonic check, cung objective + switching cost voi DRL) — goi la "AO local-search reference", chay cho N=16/32.
- **Test:** `tests/test_ao_baseline.py`, `tests/test_ao_local_search.py`.
- **Trang thai:** [FIXED]

### P0-6. Vung bong hoi tu khong phai CI 95%
- **Loi:** `plot_training_convergence` ve moving-mean +/- rolling std THEO THOI GIAN cua mot duong mean, nhung luan van goi la "khoang tin cay 95% tren 8 hat giong".
- **File:** `utils/plotting.py` (truoc refactor 74-105), `main.py` (aggregate mean truoc khi ve).
- **Sua:** ham nhan ma tran [n_seeds, n_episodes]; smooth tung seed rieng; mean qua seeds; Student-t CI 95% qua seeds tai tung episode; input 1D thi khong ve band.
- **Test:** `tests/test_ci.py`.
- **Trang thai:** [FIXED]

### P0-7. Power sweep va ablation khong dung 8 training seed
- **Loi:** `main.py` dung `trained_main = runs[0]` (seed dau tien) cho power sweep, ablation, phase histogram, h_eff distribution; `ablation.py` ghi ro "Uses ONE trained MADDPG agent".
- **File:** `main.py`, `experiments/ablation.py`.
- **Sua:** sweep/ablation/diagnostics vong qua tat ca training seeds (learned); baseline khong hoc chay tren scenario bank, KHONG nhan ban qua training seeds; CI tinh qua training seeds; xuat raw tidy CSV (algorithm, training_seed, evaluation_seed, scenario_id, Pmax, N, K, metric, value, config_sha, checkpoint_sha).
- **Trang thai:** [FIXED]

### P0-8. So sanh latency CPU/GPU chua cong bang
- **Loi:** `latency_benchmark` khong ep model ve CPU; voi device=auto tren Kaggle T4, Actor chay GPU con BCD chay NumPy CPU, roi so truc tiep.
- **File:** `experiments/evaluate.py` (truoc refactor 126-155).
- **Sua:** `benchmark_latency_cpu` (deepcopy model ve CPU, torch.set_num_threads(1) + restore trong finally, inference_mode, warm-up, >=2000 calls, mean/median/std/p95, do ca preprocessing) va `benchmark_latency_gpu` (cuda.synchronize truoc/sau). Hai bang xuat rieng, khong so cheo. AO benchmark cung CPU 1 thread. Metadata phan cung/torch/threads luu kem.
- **Trang thai:** [FIXED]

### P0-9. Notebook co dau hieu result-driven tuning
- **Loi:** cac comment "ep learned-RIS bam sat analytical", "learned-RIS >= analytical", "hinh hoi tu thuyet phuc reviewer" + assertion PASS/FAIL learned>=analytical trong notebooks va results/*.ipynb.
- **File:** `notebooks/star-ris-rsma-maddpg-v20.ipynb`, `notebooks/star-ris-rsma-maddpg-v20_scalability_N.ipynb`, `results/**/*.ipynb`.
- **Sua:** notebooks trong `notebooks/` duoc cap nhat sang API/config moi, xoa comment dinh huong ket qua va assertion; seed split train/validation/test locked trong config; `results/` chuyen sang `results_legacy/` voi checksum va danh dau NOT FOR FINAL REPORTING (giu nguyen noi dung lam bang chung lich su).
- **Trang thai:** [FIXED]

---

## P1 — Loi quan trong cho tinh dung va tai lap

### P1-1. Mo ta observation/action trong luan van khong khop code
- **Loi:** luan van mo ta obs chua |h_eff|^2, rate tung user, QoS flags, pha hien tai; code thuc te chua Re/Im h_eff, prev power/split, prev reward, Re/Im kenh tho. Thu tu action RIS-R trong luan van nguoc voi code ([beta, phase] trong code).
- **File:** `env/star_ris_env.py`, `latex_thesis/chapters/03_chuong_2.tex` (414-455).
- **Sua:** env co `observation_schema()` / `action_schema()` (name, start, stop, mo ta) + `export_schema()` ghi JSON; luan van viet lai theo schema; obs bo sung trang thai RIS hien hanh (can cho Markov property trong dynamic MDP).
- **Test:** `tests/test_schema.py`.
- **Trang thai:** [FIXED]

### P1-2. Replay buffer luu obs da chuan hoa bang statistics thay doi
- **Loi:** obs duoc normalize (update=True) roi moi luu; mean/var tiep tuc troi -> critic hoc tren nhieu he toa do khac nhau.
- **File:** `experiments/train.py` (truoc refactor 164-178, 292-318).
- **Sua:** buffer luu RAW obs; agent so huu normalizer (`attach_obs_normalizer`), normalize dung mot lan trong `select_action(s)`/`learn()`; normalizer update trong warmup roi FREEZE theo `obs_norm_freeze_after_env_steps`; eval dung cung frozen normalizer qua agent.
- **Test:** `tests/test_normalization.py`.
- **Trang thai:** [FIXED]

### P1-3. Geometry chi sinh mot lan cho moi training run
- **Loi:** vi tri user sinh o constructor, khong sinh lai o reset — moi seed hoc tren 1 bo tri user.
- **File:** `env/star_ris_env.py` (truoc refactor 147-151, 628-640).
- **Sua:** `resample_positions_on_reset: true` (default cho dynamic_mdp); geometry RNG stream rieng; static_block giu hanh vi cu.
- **Trang thai:** [FIXED]

### P1-4. Power penalty la so hang chet
- **Loi:** softmax bao dam P_c + sum P_k = P_max nen `power_excess` luon 0; `reward_gamma` khong co tac dung.
- **File:** `env/star_ris_env.py` (truoc refactor 675-676, 689).
- **Sua:** xoa khoi reward va config; ghi chu trong MIGRATION.md (huong nang luong hieu qua de lai tuong lai).
- **Trang thai:** [FIXED]

### P1-5. Checkpoint chua du de resume
- **Loi:** chi luu weights; thieu optimizer, replay, steps, OU, normalizer, lambda, RNG, config.
- **File:** `algorithms/maddpg/agent.py` (truoc refactor 199-218).
- **Sua:** tach `save_inference` (weights + normalizer + hash) va `TrainingCheckpoint` experiment-level trong `experiments/checkpointing.py` (episode index, dual EMA, best-val state, replay, normalizer + freeze state, OU, optimizer, RNG python/numpy/torch/CUDA, env state, effective config); `main.py --resume`.
- **Test:** `tests/test_checkpoint.py` (exact resume voi replay fixture nho).
- **Trang thai:** [FIXED]

### P1-6. Best model chon theo noisy training return
- **Loi:** `best.pt` luu khi raw training episode return cao nhat (chiu anh huong kenh thuan loi, noise, lambda dang doi).
- **File:** `experiments/train.py` (truoc refactor 222-224).
- **Sua:** chon theo tieu chi lexicographic tren validation scenarios: feasible khi `max_k(R_min - mean R_k) <= model_select_constraint_tolerance`; trong nhom feasible chon sum-rate cao nhat; neu khong co: min max-violation -> min mean-violation -> max sum-rate. Khong bao gio dung test seeds.
- **Test:** `tests/test_model_selection.py`.
- **Trang thai:** [FIXED]

### P1-7. Baseline chua parameter-matched
- **Loi:** MADDPG vs TD3/DDPG khac nhau dong thoi nhieu yeu to (so actor, obs, tong tham so...).
- **Sua:** them `td3_matched` (tong tham so trainable khop MADDPG sai lech <=5%, assert luc khoi tao); xuat `tables/model_complexity.csv` (actor/critic/total/inference params, FLOPs). Cac control khac (shared critic, no residual...) ghi vao MIGRATION.md la viec tuong lai.
- **Trang thai:** [PARTIAL] (td3_matched + complexity table trong pham vi; cac control khac ngoai pham vi dot nay)

### P1-8. Thong ke chua dung paired design + multiple comparisons
- **Loi:** Welch unpaired mac du cac thuat toan dung cung seeds; 9 phep kiem dinh khong hieu chinh.
- **File:** `main.py`, `utils/metrics.py`.
- **Sua:** paired t-test + paired permutation test tren seed-level means; Holm-Bonferroni voi family dinh nghia ro (3 so sanh chinh MADDPG vs {TD3, DDPG, PPO} tren sum-rate); Cohen's d paired; CI cua hieu; training ScenarioBank co dinh theo seed de pairing hop le; RNG streams tach (geometry/channel/policy/replay/random-RIS/solver).
- **Trang thai:** [FIXED]

### P1-9. Gia thiet STAR-RIS ly tuong chua nhan manh
- **Sua:** config them nhan `star_ris_hardware_model: ideal_independent_phase_es` (chi la nhan tai lieu); luan van ghi ro "Ideal independent-phase ES STAR-RIS"; cac ablation phan cung (quantization, coupled phase, insertion loss) ngoai pham vi, ghi MIGRATION.md.
- **Trang thai:** [DOCUMENTED]

### P1-10. Analytical prior dua vao direct link rat yeu
- **Loi:** pha cua h_d voi T-user bi chan 25 dB co the vo nghia vat ly.
- **File:** `env/star_ris_env.py` (`_analytical_phases`).
- **Sua:** fallback: khi |h_d| cua user tham chieu < nguong, can pha cascaded ve reference 0 (chi bu pha cua conj(g)*G).
- **Test:** trong `tests/test_env_physics.py`.
- **Trang thai:** [FIXED]

---

## Hang muc bo sung tu cac vong review ke hoach

| Muc | Noi dung | Trang thai |
|---|---|---|
| Golden fixture | Physics-only, sinh truoc refactor, dtype-aware tolerance, metadata day du | [FIXED] (`tests/fixtures/golden_static_block.npz`) |
| ScenarioBank | Pre-generate geometry + kenh khoi tao + innovation arrays tung step; channel RNG tach khoi policy RNG | [FIXED] |
| Seed split | [11,22,33,44,55] ha cap thanh validation (da bi xem); test seeds moi locked [9101,9202,9303,9404,9505] | [FIXED] |
| reward_rate_reference | Thay hardcode K*5 | [FIXED] |
| Switching cost | Chuan hoa theo so chieu; gom ca beta switching cost; = 0 tai t=0 | [FIXED] |
| Smoke config | `config/smoke.yaml` voi warmup < smoke steps, di qua actor/critic/dual update + eval + checkpoint reload | [FIXED] |
| Legacy/revised outputs | `results_legacy/` (NOT FOR FINAL REPORTING, MANIFEST.sha256) vs `results_revised/` | [FIXED] |
| No icon/emoji trong code | Thay ky tu dac biet (lambda hy lap trong tqdm) bang ASCII | [FIXED] |

---

## Pre-run source review (14 blockers) — da sua

| # | Blocker | Cach sua | Trang thai |
|---|---|---|---|
| 1 | Eval dung final agent nhung hash best.pt | train_* tra `agent`=rebuild tu best.pt (load_maddpg/single/ppo_inference); moi eval/sweep/ablation/latency dung best.pt; luu lambda vec tai best episode | [FIXED] |
| 2 | PPO old_log_prob/value lech do re-normalize | select_action cache `last_norm_obs`; rollout luu obs DA normalize; learn() khong normalize lai; test `rollout_logprob_consistency` | [FIXED] |
| 3 | Checkpoint replay luu full 300k | replay_state luu `[0:size]`; TrainingCheckpoint luu replay sang sidecar `.replay.npz` nen (compressed) | [FIXED] |
| 4 | Capture truoc validation/selector | Doi thu tu: eval + selector.consider TRUOC capture; test resume voi eval_every==checkpoint_every | [FIXED] |
| 5 | AO goi `_evolve_channels()` khong tang step_count | Them API cong khai `env.advance_to_next_block()` (tang step_count, innovation 0,1,2); main.py khong goi private | [FIXED] |
| 6 | AO coi block tuong quan la mau doc lap | Trung binh block trong moi scenario truoc, CI qua scenario means; cot `ci_unit=scenario_mean` | [FIXED] |
| 7 | ScenarioBank hash ca seed list | Nested loop (evaluation_seed, episode_idx); scenario_id `test_seed9101_ep000`; luu split/seed/idx | [FIXED] |
| 8 | results_raw thieu metadata | Moi row co evaluation_seed, episode_idx, scenario_id, config_sha, checkpoint_sha; baseline co solver_config_sha | [FIXED] |
| 9 | lambda vec bi thay bang mean khi eval | `_make_env`/`evaluate_agent` nhan `qos_lambda_vec_override`; goi set_qos_lambda_vec | [FIXED] |
| 10 | Chua pin CPU threads | `configure_cpu_threads(1,1)` trong conftest + main (khi cpu) | [FIXED] |
| 11 | Legacy tolerance qua chat cho rate | Channels/h_eff strict 1e-10; rate_common/per_user_rate/sum_rate rtol 1e-7 | [FIXED] |
| 12 | Smoke khong xong vi AO | smoke.yaml `ao_local_search_max_n: 0`; test rieng `tests/test_ao_smoke.py` | [FIXED] |
| 13 | Thieu regression tests | Them: evaluated==best.pt, checkpoint_sha khop, PPO consistency, AO innovation 0/1/2, raw metadata, replay size | [FIXED] |
| 14 | Artifacts lan vao release | Output vao `results_revised/run_<confighash>_<timestamp>/`; `.gitignore` loai results/logs/checkpoints/replay | [FIXED] |

---

## Research-critical review 2 (9 muc) — da sua

| # | Van de | Cach sua | Trang thai |
|---|---|---|---|
| 1 | Critic hoc reward cu (lambda thay doi) | Buffer luu `base_reward` + `c_gap`; recompute reward = clip(base - r_scale*dot(lambda_cur, c)) luc sample; `set_current_lambda` sau moi dual update; tests reward khac theo lambda + dung sau freeze | [FIXED] |
| 2 | PPO rollout tron nhieu lambda | Lambda co dinh trong 1 rollout; dual.update CHI sau PPO learn; rollout luu obs da normalize (item 2 cu); test dual chi sau learn | [FIXED] |
| 3 | Actor gradient MADDPG khong chuan | Actor-i loss dung a_i=actor_i(o_i) co grad, a_j=actor_j(o_j).detach() (khong dung replay actions cua agent khac); test actor gradient | [FIXED] |
| 4 | Test seeds da bi xem | [9101..9505] -> legacy_eval_seeds; them dev_seeds [101,202,303] cho smoke/quick/pilot; test_seeds placeholder [70001..] phai regenerate luc code freeze | [FIXED] |
| 5 | skip-train khong load duoc run cu | `--load-run-dir`: doc run_meta.json, load checkpoints/normalizers, eval khong train lai; test e2e | [FIXED] |
| 6 | Diagnostics doc state sau transition | info tra `phi_r_applied/phi_t_applied/h_eff_applied_abs` (truoc evolution); collector doc tu info | [FIXED] |
| 7 | Chua ho tro chay phan tan | `--seeds` subset, `--run-id` co dinh, `--load-run-dir` aggregate, `--drop-replay-after-train` | [FIXED] |
| 8 | TD3-Matched chua la so sanh chinh | Them vao PRIMARY_FAMILY (Holm m=4); caption ghi ro yeu cau thang TD3-Matched cho claim phan ra | [FIXED] |
| 9 | Return lam metric chinh | Bo Return khoi bang so sanh; them MaxExpViolation/MeanExpViolation (c_bar) + physical metrics | [FIXED] |

---

## Review CODE V4 (10 muc) — da sua

| # | Blocker/Van de | Cach sua | Trang thai |
|---|---|---|---|
| P0-1 | Shard chua khoa theo source | `source_tree_sha256()` (hash *.py + config/*.yaml + requirements.txt, loai results/logs/checkpoints/__pycache__); ghi vao shard_manifest, completed_runs.csv, run_meta.json, results_raw.csv, best.pt meta; `validate_shard_group` TU CHOI shard khac source_sha ke ca khi config_sha giong | [FIXED] |
| P0-2 | Aggregate mat training history | Manifest luu history_csv/history_sha + log_csv/log_sha; aggregate xac minh hash, nap history tung seed, dung lai ma tran [n_seeds, n_episodes], ve du convergence/sum-rate/QoS/lambda/reward-decomposition; fail neu thieu history | [FIXED] |
| P0-3 | Train-only van tao locked test bank | API `build_eval_bank(cfg, split)`; `_get_validation_bank` chi tao validation; test spy xac nhan train-only chi materialize split "validation" | [FIXED] |
| P0-4 | Aggregate chap nhan thieu algo/seed | `--final-paper` (bat buoc ma tran dang ky truoc `evaluation.required_algorithms x required_training_seeds`, tu choi thieu/du/trung); mac dinh strict theo config; `--allow-partial` chi cho smoke/pilot | [FIXED] |
| P0-5 | Notebook monolithic + seed_split path | Notebook chinh sinh 40 lenh train-only + cell aggregate `--final-paper`; scalability dung `main.load_config()` (embed seed lists + resolved_sha256) va shard theo N x algorithm x seed | [FIXED] |
| 6 | pytest treo o integration test | Config `evaluation.latency_num_calls/latency_warmup_calls/diagnostic_steps`; conftest test dat 5/1/4 -> full suite 107 test / ~30 s | [FIXED] |
| 7 | AO thieu wall time, params chua khoa | `solve_time_ms` moi solve + wall_time_ms moi scenario (CSV + raw rows); `evaluation.ao_solver` dang ky truoc, tham gia solver_config_sha | [FIXED] |
| 8 | Manifest thieu moi truong chay | `environment` (python/torch/numpy/cuda/gpu/platform/hostname/source_sha) trong moi shard manifest + run_meta | [FIXED] |
| 9 | drop-replay lam metadata sai | Sau khi xoa sidecar: `resumable_training=false`, `replay_dropped=true` trong manifest | [FIXED] |
| 10 | Tolerance/feasibility chua khoa | Comment pre-registration cho `model_select_constraint_tolerance`; `experiments/feasibility.py` (AO objective="maxmin", validation-only, tu choi split test) | [FIXED] |
