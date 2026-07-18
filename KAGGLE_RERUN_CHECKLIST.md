# KAGGLE RERUN CHECKLIST - tai sinh toan bo so lieu sau refactor

Muc dich: moi bang/hinh trong Chuong 4 cua luan van PHAI duoc thay bang ket
qua tu pipeline moi (dynamic_mdp). Ket qua trong `results_legacy/` KHONG duoc
dung lam bang chung cuoi cung. Luan van o trang thai PENDING cho den khi
checklist nay hoan tat.

## 0. Quy tac bat di bat doi (EXPERIMENT PROTOCOL)

1. `config/seed_split.v1.yaml` la nguon duy nhat cho seed split; SHA-256 bat
   buoc la `d73ba5aea6d037570f2634cbc87175db259a6e91f0fecee74519eecd1f118854`.
   Loader se dung ngay neu file va hash khong khop.
2. Tune sieu tham so CHI bang validation scenarios [11, 22, 33, 44, 55].
   Tap legacy [9101, 9202, 9303, 9404, 9505] da bi xem trong qua trinh phat
   trien va KHONG phai final test.
3. Locked final-test scenarios [70001, 70002, 70003, 70004, 70005] chi duoc
   tao/chay sau khi code va config da khoa; bao cao dung ket qua thu duoc, ke ca
   khi khong nhu ky vong.
4. KHONG bao gio: chinh tham so sau khi xem ket qua test roi chay lai test;
   dat dieu kien "learned phai thang analytical"; chon checkpoint theo test.
5. Luu lai moi lan tune (config + validation score) vao tuning_history.csv
   (tu tao, cot: date, config_sha, thay_doi, val_sum_rate, val_max_violation).
6. Truoc khi train dai: `python -m pytest tests -q` phai PASS.

## 1. Chuan bi (1 lan)

- [ ] Upload repo (sau refactor) len Kaggle dataset hoac clone tu remote.
- [ ] `pip install -r requirements.txt` (pinned).
- [ ] `python -m pytest tests -q` -> all pass.
- [ ] `python main.py --config config/smoke.yaml --algos maddpg ddpg td3 td3_matched ppo` -> chay het,
      kiem tra results_revised/tables/results_raw.csv co cot scenario_id,
      config_sha, checkpoint_sha.
- [ ] Khoa config chinh: commit/ghi lai sha256 cua config/config.yaml
      (pipeline tu luu effective_config.yaml + hash).

## 2. Thi nghiem chinh (GPU T4/P100, ~vai giay/episode)

Notebook: `notebooks/star_ris_rsma_main.ipynb`

- [ ] Moi Kaggle job train DUNG MOT cap algorithm-seed. Quy tac nay duoc CLI
      enforce cho `--train-only`, vi DDPG/TD3/PPO chua co full resumable
      checkpoint; job bi ngat chi lam mat mot seed. Vi du:
      `python main.py --config config/config.yaml --train-only --algos td3_matched --seeds 1000 --run-id paper_v1`.
- [ ] Lap lai cho 8 seeds x {maddpg, ddpg, td3, td3_matched, ppo}. Moi job chi
      ghi `results_revised/shards/paper_v1/<algorithm>_seed<seed>/` va khong
      ghi run_meta.json, bang hay hinh aggregate.
- [ ] Aggregate sau khi thu du shard (BAT BUOC --final-paper cho bang so lieu
      Chuong 4 - tu choi neu thieu/du/trung cap algorithm-seed hoac source_sha
      khong dong nhat):
      `python main.py --aggregate-only --load-shards results_revised/shards/paper_v1 --final-paper`.
      Voi pilot/smoke thieu shard dung `--allow-partial` (khong duoc dung cho
      so lieu chinh thuc). Neu source aggregate khac source train, --final-paper
      dung ngay. Kiem tra `completed_runs.csv` (co cot source_sha);
      `results_raw.csv` phai co hai cot training_source_sha va
      evaluation_source_sha;
      aggregator scan/verify manifest + source/config/checkpoint SHA va nap
      history.csv da hash de ve hinh hoi tu; khong tin run_meta.json.
- [ ] Truoc khi mo locked test: chay feasibility validation-only
      `python -m experiments.feasibility --config config/config.yaml --split validation --scenarios-per-seed 2 --out feasibility_validation.csv`.
      Ket qua freeze: feasible 10/10, min max-min rate 1.422 > 0.3;
      `model_select_constraint_tolerance: 0.0`.
- [ ] Chay AO penalty diagnostic validation-only:
      `python -m experiments.ao_diagnostics --config config/config.yaml --split validation --scenarios-per-seed 2 --out ao_solver_validation.csv`.
      Freeze `ao_solver={n_starts:3,max_outer:8,tol:1e-5,pg_steps:40,pg_lr:0.3}`;
      bao cao trung thuc converged 0/10 (fixed-budget local reference), mean/p95
      18.48/18.92 s, max violation 0.30, objective CV 0.120. Khong tune bang final test.
      (sau do KHONG doi nua).
- [ ] Thu ve toan bo `results_revised/` + `logs/` + `checkpoints/*/best.pt`.
- [ ] Kiem tra: significance.csv co p_holm; algorithm_comparison.csv co
      UserQoSFraction VA AllUsersQoSProb rieng biet.

Dan so lieu vao luan van:
- [ ] Bang 4.x so sanh thuat toan  <- tables/algorithm_comparison.tex
- [ ] Bang significance           <- tables/significance.tex
- [ ] Bang model complexity (moi) <- tables/model_complexity.tex
- [ ] Hinh hoi tu (CI dung)       <- figures/training_convergence.*
- [ ] Hinh QoS/lambda             <- figures/training_user_qos_fraction.*,
                                     figures/qos_lambda.*
- [ ] Hinh power sweep            <- figures/sumrate_vs_power.*, qos_vs_power.*
- [ ] Bang/hinh ablation          <- tables/ablation.tex, figures/ablation.*
- [ ] Pareto                      <- figures/pareto_sr_vs_qos.*
- [ ] Hinh phase/|h_eff|          <- figures/phase_histogram.*, h_eff_distribution.*

## 3. Latency (2 job rieng)

- [ ] Job CPU-only (Kaggle CPU instance): chay main voi --skip-train sau khi
      tai checkpoints, hoac goi benchmark_latency_cpu truc tiep. Thu
      tables/latency_cpu.{csv,tex} + latency_cpu_meta.json.
- [ ] Job GPU: tables/latency_gpu.{csv,tex} + latency_gpu_meta.json.
- [ ] Dan vao luan van HAI bang rieng; khong so sanh cheo CPU/GPU;
      cau "do tren cung phan cung CPU" chi duoc giu neu dung bang CPU.

## 4. Scalability N (retrain per N)

Notebook: `notebooks/star_ris_rsma_scalability_N.ipynb`

- [ ] N in {16, 32, 64, 96, 128}: train {maddpg, td3} x 8 seeds moi N.
- [ ] scalability_runs/scalability_summary.csv: MADDPG vs TD3 paired t +
      Holm theo N.
- [ ] Chi giu cau "MADDPG vuot TD3 tu N>=64" neu p_holm < 0.05 tai cac N do;
      tai N=32 giu "tuong duong thong ke".

## 5. AO references + feasibility

- [ ] Hybrid AO Local Search cho N=16 va N=32 tren test bank, stratified it
      nhat mot scenario tu moi locked test seed (tables/ao_local_search.csv va
      tables/ao_local_search_scenarios.csv; chay tu main.py hoac script rieng).
- [ ] Feasibility analysis cua R_min = 0.3: chay AO local search (objective
      max-min rate) tren >= 1000 scenario; bao cao ty le scenario kha thi.
      Neu ty le thap, luan van KHONG duoc phat bieu kha nang thoa man QoS
      cung; giu framing surrogate/expected-rate.

## 6. Sau khi co so lieu moi

- [ ] Thay TOAN BO so trong Chuong 4 + Ket luan + tables/*.tex bang so moi
      (cac cho can thay da duoc danh dau "PIPELINE-CU" trong ghi chu dau
      Chuong 4).
- [ ] Ve lai moi hinh tu results_revised/figures (hinh cu dung rolling-std
      lam "CI" khong duoc giu).
- [ ] Cap nhat gia tri cu the: sum-rate, QoS (ghi ro metric nao), latency,
      % so voi AO-Grid (thay cho "% can tren").
- [ ] Xoa ghi chu PENDING dau Chuong 4.
- [ ] Luu raw: results_raw.csv (co training_seed, scenario_id, config_sha,
      checkpoint_sha) kem ban nop.
