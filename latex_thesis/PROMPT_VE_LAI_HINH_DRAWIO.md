# PROMPT VẼ LẠI 14 HÌNH MINH HỌA LUẬN VĂN (draw.io) — CHUẨN IEEE

> Dùng file này làm đề bài cho skill draw.io: vẽ lại từng hình theo mô tả.
> Mỗi hình xuất **PDF (vector, crop sát nội dung, nền trong suốt/trắng)** + PNG 300dpi,
> đặt đúng **tên file** ghi ở đầu mỗi mục (thay thế file cũ trong `latex_thesis/figures/`).

---

## A. QUY TẮC STYLE CHUNG (áp dụng cho MỌI hình)

1. **KHÔNG nhúng tiêu đề trong hình** (kiểu "Hình 2.1 – ..."). Caption đã có trong LaTeX — hình chỉ chứa nội dung.
2. **Font:** Helvetica/Arial. Cỡ chữ khi in ở bề rộng 14cm phải ≥ 8pt → trong draw.io dùng 12–14px cho nhãn thường, 16px cho nhãn nhóm. **Không chữ tràn ra ngoài khối, không chữ đè lên nhau.**
3. **Màu:** thiết kế an toàn khi in đen-trắng. Viền đen 1px; nền khối trắng; chỉ dùng **1 mức xám nhạt (#F2F2F2)** cho khối cần nhấn và **1 màu nhấn duy nhất (xanh đậm #1F4E79)** cho phần tử "đề xuất/học được". Không gradient, không bóng đổ.
4. **Mũi tên:** nét liền = luồng tín hiệu/dữ liệu chính; **nét đứt** = kênh gián tiếp/gradient/phản hồi; đầu mũi tên kiểu classic nhỏ, nhất quán toàn bộ.
5. **Ký hiệu toán:** dùng đúng như luận văn (Unicode/HTML): θ, β, λ, φ, γ, π, ℝ, ∑, P_max viết `P_max`, chỉ số dưới bằng subscript của draw.io. Nhất quán: pha θ (không dùng φ lẫn lộn), residual Δφ.
6. **Chính tả:** "Tác tử" (KHÔNG phải "Tác từ"), "Trạm gốc (BS)", "Người dùng (UE)".
7. **Tỷ lệ khung:** hình đơn cột luận văn chèn `width=0.85\linewidth` → vẽ khung ~ **1600×900 tới 1600×1100 px** (ngang); riêng flowchart h3 dạng dọc ~ **1100×1600 px**.
8. **Số liệu phải khớp phiên bản CUỐI của luận văn (SAU REFACTOR 2026-07-16)**:
   - Episodes **E = 2000**; warmup 5000 bước; noise OU decay **45 000 bước** (σ 0,4→0,05)
   - λ_k per-user: khởi tạo 1,0; **cập nhật projected dual gradient
     λ_k ← clip(λ_k + 0,01·EMA(c_k), 0, 20) với c_k = R_min − R_k (có dấu, EMA 0,9)**
   - **Two-stage dual freeze: đóng băng λ từ episode ρ_f·E = 0,55·2000 = 1100 (heuristic)**
   - Residual pha: **θ = θ_prior + (π/4)·Δφ** (±45°) — KHÔNG phải π/2
   - Kênh ĐỘNG Gauss–Markov: h_{t+1} = 0,95·h_t + sqrt(1−0,95²)·ε (mỗi step một khối coherence)
   - Reward: α·R_sum/R_ref − Σλ_k·c_k − (w/2)·Σmax(c_k,0)² − switching costs (KHÔNG còn b_sat, b_power)
   - Seeds: 8 training; validation [11..55]; legacy [9101..9505]; locked final test [70001..70005]. Source: config/seed_split.v1.yaml, SHA-256 d73ba5aea6d037570f2634cbc87175db259a6e91f0fecee74519eecd1f118854
   - Kích thước MDP MỚI: local obs 73+577+401; canonical state **681**; action 44+64+32=**140**; critic input **821**

---

## B. MÔ TẢ TỪNG HÌNH

### H1 — `h1_1_kien_truc_6g` (Hình 1.1, Chương 1)
**Mục đích:** Kiến trúc 6G tích hợp 3 tầng.
**Bố cục:** 3 dải ngang chồng nhau (trên→dưới):
- **TẦNG VŨ TRỤ**: icon vệ tinh LEO + vệ tinh GEO.
- **TẦNG TRÊN KHÔNG**: UAV/drone, HAPS (khinh khí cầu), **RIS trên không** (tấm phẳng có lưới ô vuông gắn trên drone).
- **TẦNG MẶT ĐẤT**: trạm gốc gNB, small cell, tấm **RIS gắn tường nhà**, cụm UE (điện thoại) + cảm biến IoT.
**Liên kết:** mũi tên nét đứt hai chiều nối giữa các tầng (vệ tinh↔HAPS↔gNB↔UE). 
**Chú thích chân hình (1 dòng nhỏ):** "6G: >1 Tbps · độ trễ <0,1 ms · 10⁷ thiết bị/km²".
**Sửa so với bản cũ:** bỏ tiêu đề nhúng; icon to rõ hơn, giảm chữ; 3 nhãn tầng đặt bên trái dải, in đậm.

### H2 — `h1_2_oma_noma_rsma` (Hình 1.2)
**Mục đích:** So sánh cách chiếm tài nguyên của OMA / NOMA / RSMA.
**Bố cục:** 3 panel cạnh nhau, mỗi panel là hệ trục: trục ngang "Tần số/Thời gian", trục dọc "Công suất".
- **OMA:** 2 khối chữ nhật đặt CẠNH nhau theo trục ngang (UE1, UE2 — hai mức xám khác nhau có nhãn).
- **NOMA:** 2 khối CHỒNG lên nhau theo trục dọc trên cùng dải tần (UE1 dưới, UE2 trên).
- **RSMA:** 2 khối riêng (luồng riêng UE1, UE2) chồng như NOMA + **1 khối "Luồng chung (common)" màu nhấn #1F4E79 phủ ngang trên cùng** cả dải.
**Legend chung dưới 3 panel:** ▢ Luồng riêng UE1 · ▢ Luồng riêng UE2 · ▉ Luồng chung.
**Sửa:** bản cũ chữ chú giải quá nhỏ và lệch; panel phải đều nhau, cùng kích thước trục.

### H3 — `h1_3_rsma_tin_hieu` (Hình 1.3)
**Mục đích:** Sơ đồ khối xử lý tín hiệu RSMA 1 lớp (phát → thu).
**Bố cục ngang, 2 cụm:**
- **PHÍA PHÁT (khung trái):** W₁, W₂ → khối "Bộ tách thông điệp" → nhánh phần chung W₁ᶜ,W₂ᶜ → khối "Bộ kết hợp" → sᶜ; nhánh phần riêng W₁ᵖ,W₂ᵖ → khối "Mã hóa"; các luồng nhân công suất √P_c, √P_k → khối tổng "Σ" → anten phát x.
- **PHÍA THU (khung phải, cho UE k):** anten thu y_k → khối "Giải mã luồng chung ŝᶜ" → khối "SIC (trừ ŝᶜ)" → khối "Giải mã luồng riêng ŝ_k". Mũi tên một chiều trái→phải xuyên suốt.
**Sửa:** bản cũ khối quá nhỏ so chữ; dàn lại đều, mỗi khối 1–2 dòng chữ.

### H4 — `h1_4_star_ris_vs_ris` (Hình 1.4)
**Mục đích:** RIS thường (phủ 180°) vs STAR-RIS (phủ 360°).
**Bố cục:** 2 panel.
- **Panel trái "RIS truyền thống":** BS trái; tấm RIS đứng giữa; nửa mặt phẳng TRƯỚC tô cung 180° nhạt + UE nhận được; nửa SAU có UE kèm dấu ✕ và nhãn "vùng chết".
- **Panel phải "STAR-RIS":** cùng bố cục nhưng cung phủ **360°** (2 nửa: nhãn "R – phản xạ" phía trước, "T – truyền qua" phía sau), UE cả 2 phía đều nhận tia.
**Sửa:** nhấn tương phản 2 cung phủ; nhãn R/T đặt trong cung; bỏ chữ thừa.

### H5 — `h1_5_vong_lap_rl` (Hình 1.5)
**Mục đích:** Vòng lặp tác tử–môi trường chuẩn RL.
**Bố cục:** 2 khối lớn: "TÁC TỬ (chính sách π(a|s))" trên, "MÔI TRƯỜNG (STAR-RIS RSMA)" dưới. Mũi tên phải: hành động **a_t** (nhãn: pha RIS, công suất, tỷ lệ tách). Mũi tên trái: **s_{t+1}, r_{t+1}** (nhãn: trạng thái kênh, phần thưởng).
**Ghi trong khối môi trường (1 dòng):** "K=4 UE · N=32 · Rayleigh block-fading".
**Sửa:** tối giản — đây là hình khái niệm, KHÔNG nhồi công thức J(π); 2 khối + 2 mũi tên + nhãn là đủ.

### H6 — `h1_6_maddpg_ctde` (Hình 1.6)
**Mục đích:** Kiến trúc MADDPG với CTDE (khái niệm, chưa gắn số chiều cụ thể).
**Bố cục:** 2 khung lớn cạnh nhau.
- **Khung trái "GIAI ĐOẠN 1 — HUẤN LUYỆN TẬP TRUNG":** 3 khối Actor (Actor₀ BS · Actor₁ RIS-R · Actor₂ RIS-T; mỗi khối ghi "quan sát cục bộ o_i → hành động a_i"); **3 khối Critic Q₀,Q₁,Q₂ xếp chồng** (nhãn chung "Critic tập trung: nhận (o,a) toàn cục") — LƯU Ý: luận văn dùng **3 critic riêng**, bản cũ vẽ 1 critic là SAI; khối "Replay buffer 𝒟"; khối nét đứt "Target networks (τ=0,005)". Mũi tên nét liền o_i,a_i → Critic; mũi tên **nét đứt** gradient ∇ từ Critic về từng Actor; buffer → Critic (lô mẫu).
- **Khung phải "GIAI ĐOẠN 2 — THỰC THI PHÂN TÁN":** chỉ 3 Actor + khối "Hệ thống STAR-RIS RSMA"; mũi tên o_i vào, a_i ra. Ghi chú dưới: "Bỏ Critic khi triển khai — mỗi Actor chỉ cần quan sát cục bộ".
**Sửa:** 3 critic thay vì 1; sửa "Tác từ"→"Tác tử"; giảm chữ trong khối (chi tiết siêu tham số bỏ hết — đã có Bảng 3.1).

### H7 — `h2_1_cau_hinh_he_thong` (Hình 2.1, Chương 2)
**Mục đích:** Cấu hình hệ thống: BS MISO → STAR-RIS → 2 vùng người dùng.
**Bố cục (ngang):**
- Trái: icon **Trạm gốc (BS)** + nhãn "M = 4 anten, P_max = 30 dBm".
- Giữa: tấm **STAR-RIS đứng dọc** (lưới chấm; nửa trên nhãn **R**, nửa dưới nhãn **T**; chú thích "N = 32 phần tử, chế độ ES").
- Phải-trên: khung **"Vùng phản xạ (R)"** chứa UE_R1, UE_R2, UE_R3 (CÙNG PHÍA với BS).
- Phải-dưới (hoặc tách hẳn bên kia tấm RIS): khung **"Vùng truyền qua (T)"** chứa UE_T1; giữa BS và UE_T1 vẽ **bức tường/chướng ngại** + nhãn "chặn −25 dB".
**Kênh (mũi tên):** BS→RIS nét liền đậm nhãn **G**; RIS→UE_Rk nét liền nhãn **h^R_{r,k}**; RIS→UE_T1 nét liền nhãn **h^T_{r,1}** (xuyên qua tấm); BS→từng UE nét **đứt** nhãn **h_{d,k}** (đường tới UE_T1 đi qua tường, gắn dấu ✕).
**⚠️ SỬA LỖI bản cũ:** nhãn khung bên trái đang ghi sai "Vùng Phản Xạ (T)" → phải là **"(R)"**. Bỏ các khối công thức dài (tín hiệu phát, kênh hiệu dụng) — thuộc về text, không nhét vào hình.

### H8 — `h2_2_kenh_truyen` (Hình 2.2)
**Mục đích:** Ba loại kênh + tham số suy hao.
**Bố cục:** rút gọn của H7 (chỉ BS, RIS, 1 UE_R, 1 UE_T), mỗi mũi tên kênh gắn **thẻ nhỏ**: 
- h_{d,k}: "trực tiếp, α_d = 3,5 (+25 dB chặn với UE_T)"
- G: "BS→RIS, α_G = 2,2"
- h^X_{r,k}: "RIS→UE, α_r = 2,5"
**Chú thích chân:** "Mọi kênh: Rayleigh i.i.d. + suy hao log-distance PL(d)=PL₀+10α·log₁₀(d/d₀), PL₀=30 dB @ 1 m".

### H9 — `h2_3_star_ris_es` (Hình 2.3)
**Mục đích:** Nguyên lý 1 phần tử STAR-RIS chế độ ES.
**Bố cục:** giữa là 1 ô phần tử (vuông); mũi tên **sóng tới** từ trái; tách thành 2 mũi tên ra:
- lên-trái: "phản xạ: √(β^R_n)·e^{jθ^R_n}" 
- xuyên-phải: "truyền qua: √(β^T_n)·e^{jθ^T_n}"
**Khối ràng buộc (khung nhấn):** "Bảo toàn năng lượng (ES): β^R_n + β^T_n = 1, ∀n".
**Sửa:** bỏ ma trận Φ dài dòng; 1 phần tử + 2 tia + 1 ràng buộc là đủ.

### H10 — `h2_4_bai_toan_p0` (Hình 2.4)
**Mục đích:** Cấu trúc bài toán P0.
**Bố cục dạng khối phân cấp:**
- Trên cùng (khối nhấn): "**max R_sum** theo (P, Θ, C)".
- Hàng giữa, 3 khối "BIẾN": P = {P_c, P_1..P_K} · Θ = {θ^R_n, θ^T_n, β^R_n} · C = {c_1..c_K}.
- Hàng dưới, 6 khối nhỏ "RÀNG BUỘC": C1 tổng công suất ≤ P_max · C2 công suất ≥ 0 · C3 Σc_k=1 · **C4 QoS: R_k ≥ R_min ∀k** (viền màu nhấn) · C5 β^R+β^T=1 · C6 θ ∈ [0,2π).
- Chân hình 1 dòng: "Phi lồi (SINR phân thức, pha e^{jθ}, biến ghép) → giải bằng MADDPG".
**Sửa:** bản cũ quá nhiều chữ; mỗi khối tối đa 1 dòng công thức.

### H11 — `h2_5_mdp_anh_xa` (Hình 2.5)
**Mục đích:** Ánh xạ P0 → MDP (3 cột: S, A, R).
**Bố cục 3 cột:**
- **Cột 1 "TRẠNG THÁI CRITIC 𝒮 (681 chiều)":** 3 khối: Agent 0 (BS) — 26 · Agent 1 (RIS-R) — 280 · Agent 2 (RIS-T) — 148; mỗi khối 1 dòng mô tả ngắn (vd "kênh G, h_r; pha hiện tại").
- **Cột 2 "KHÔNG GIAN HÀNH ĐỘNG 𝒜 (140 chiều)":** Agent 0 — 44 (beamforming phức + common split) · Agent 1 — 64 (Δφ^R + β^R) · Agent 2 — 32 (Δφ^T); ghi chú pha chính: "θ = π(a+1)"; residual chỉ là ablation.
- **Cột 3 "PHẦN THƯỞNG r_t (chia sẻ)":** khối công thức r_t = 1,5·R̃_sum − λ_t·Σ[max(0,R_min−R_k)]² + b_pwr + b_sat; khối "λ thích nghi: ×1,02 nếu QoS<0,5 · ×0,97 nếu QoS>0,6 · λ∈[0,3; **13**]"; khối màu nhấn "**FREEZE: đóng băng λ từ episode 0,55·E = 1100**".
**⚠️ SỬA SỐ bản cũ:** π/2→π/4; 1,05→1,02; 1,02→0,97; 15,0→13,0; [0,3, 15]→[0,3, 13]; THÊM khối freeze (bản cũ không có).

### H12 — `h2_6_3agents` (Hình 2.6)
**Mục đích:** Phân rã 3 tác tử + Critic tập trung (bản "kỹ thuật" có số chiều).
**Bố cục:** hàng trên 3 khối Actor: "Actor₀ BS: 73 → 256 → 256 → 44" · "Actor₁ RIS-R: 577 → 256 → 256 → 64" · "Actor₂ RIS-T: 401 → 256 → 256 → 32"; hàng dưới 1 khung "Critic tập trung Q_i (×3): đầu vào (s,a) = 681+140 = **821** → 256 → 256 → 1 · chỉ dùng khi huấn luyện". Mũi tên o_i,a_i từ actor xuống critic; nét đứt gradient ngược lên.
**Chân hình:** "Phần thưởng chia sẻ r_t cho cả 3 tác tử (cooperative)".

### H13 — `h2_7_overall_workflow` (Hình 2.7)
**Mục đích:** Luồng hoạt động end-to-end 2 lớp.
**Bố cục:** 2 khung ngang chồng nhau:
- **Khung trên "LỚP VẬT LÝ":** chuỗi khối trái→phải: "Thông điệp W₁..W_K" → "Mã hóa RSMA (tách/kết hợp/precoder)" → "BS phát" → "STAR-RIS (N=32, ES)" → tách 2 nhánh "UE vùng R (3)" và "UE vùng T (1, −25 dB)" → "Giải mã: common → SIC → private" → khối đo "R_sum, R_k, QoS".
- **Khung dưới "LỚP ĐIỀU KHIỂN (MADDPG)":** 3 khối Actor + khối "Critic ×3 (chỉ huấn luyện)" + khối "Replay 𝒟" + khối nhấn "Phần thưởng Lagrangian thích nghi + **freeze-λ @ ep 1100**".
- Mũi tên phản hồi nét đứt từ khối đo (lớp vật lý) xuống khối phần thưởng; mũi tên hành động từ Actor lên BS/STAR-RIS.
**Sửa:** thêm freeze-λ; giảm mật độ chữ.

### H14 — `h3_algorithm_flowchart` (Hình 3.1, Chương 3) — **HÌNH HỎNG NẶNG NHẤT, VẼ LẠI TỪ ĐẦU**
**Mục đích:** Flowchart huấn luyện MADDPG (khớp Algorithm 1).
**Bố cục DỌC một cột chính (khung ~1100×1600):** dùng đúng ký pháp flowchart: oval = bắt đầu/kết thúc, hình thoi = điều kiện/vòng lặp, chữ nhật = xử lý, chữ nhật xám nhạt = bước liên quan mạng nơ-ron.
1. (oval) **BẮT ĐẦU**
2. Khởi tạo Actor μ_i, Critic Q_i, target networks, buffer 𝒟, λ←1,0
3. Warmup: 5000 bước hành động ngẫu nhiên → 𝒟
4. (thoi) **episode e = 1…E (E = 2000)** ← SỬA: bản cũ ghi 1000
5. Lấy mẫu kênh mới h_d, G, h_r; reset trạng thái
6. (thoi) **bước t = 1…T (T = 50)**
7. Tính pha tiên nghiệm θ_prior từ kênh
8. (xám) Sinh hành động a_i = μ_i(o_i) + noise OU; ánh xạ vật lý (θ = θ_prior + (π/4)Δφ; softmax công suất)
9. Thực thi: tính h_eff, SINR, R_k, R_sum → phần thưởng r_t; lưu (o,a,r,o′) vào 𝒟
10. (xám) Nếu |𝒟|>B: lấy mini-batch → cập nhật Critic (Bellman) → Actor (DPG) → soft-update target (τ=0,005)
11. → quay lại (6) hết T bước
12. Tính ρ_QoS của episode
13. **(thoi, viền màu nhấn) e < 0,55·E ?** — nhánh **Có** → "Cập nhật λ: ×1,02 / ×0,97, kẹp [0,3; 13]"; nhánh **Không** → "**ĐÓNG BĂNG λ** (giữ nguyên)" ← BƯỚC MỚI, bản cũ không có
14. Giảm σ noise (tuyến tính, sàn tại ep ≈ 900) → quay lại (4)
15. (oval) **KẾT THÚC** — trả về μ_θ0, μ_θ1, μ_θ2
**Yêu cầu bắt buộc:** có mũi tên vòng lặp rõ ràng (t-loop và e-loop, đi vòng bên phải/trái); KHÔNG chữ đè lên nhau (lỗi chính bản cũ); mỗi khối ≤ 2 dòng.

---

## C. THỨ TỰ ƯU TIÊN KHI VẼ
1. **H14 (flowchart)** — hỏng nặng + thiếu freeze-λ (sai nội dung).
2. **H11 (MDP)** — sai 4 con số + thiếu freeze.
3. **H7 (cấu hình hệ thống)** — sai nhãn vùng (R).
4. **H6 (CTDE)** — sai 1 critic → 3 critic; lỗi "Tác từ".
5. Các hình còn lại: đúng nội dung, chỉ cần vẽ đẹp/chuẩn style lại.
