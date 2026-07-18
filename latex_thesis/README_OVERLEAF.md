# Hướng dẫn Build Luận văn trên Overleaf

## Thông tin chung
- **Tên đề tài:** Tối ưu phân bổ tài nguyên sử dụng học tăng cường sâu trong mạng STAR-RIS hỗ trợ RSMA
- **Tác giả:** Nguyễn Duy Thành
- **Trường:** Đại học Giao thông vận tải TP. Hồ Chí Minh (UTH)
- **Format:** Theo file `HUONG-DAN-CHI-TIET-TRINH-BAY-LUAN-VAN-THAC-SI.docx`
- **Compiler:** **XeLaTeX** (BẮT BUỘC, không dùng pdfLaTeX)

---

## Cấu trúc thư mục

```
LUAN_VAN_THANH_LATEX/
├── main.tex                  # File chính
├── uth-thesis.cls            # Class file (font, lề, margins theo UTH)
├── references.bib            # 58 entries IEEE
├── chapters/
│   ├── 00_front_matter.tex   # Lời cam đoan, cảm ơn, tóm tắt, ToC, danh mục
│   ├── 01_mo_dau.tex         # Mở đầu (5 trang)
│   ├── 02_chuong_1.tex       # Ch1 - Cơ sở lý thuyết (~22 trang)
│   ├── 03_chuong_2.tex       # Ch2 - Mô hình hệ thống & Bài toán (~22 trang)
│   ├── 04_chuong_3.tex       # Ch3 - Phương pháp MADDPG đề xuất (~17 trang)
│   ├── 05_chuong_4.tex       # Ch4 - Kết quả mô phỏng (~16 trang)
│   └── 06_ket_luan.tex       # Kết luận (3 trang)
├── figures/
│   ├── h1_1...h2_6 (12 PNG)  # Hình minh hoạ Ch1-Ch2 (drawio export)
│   └── training_*, ablation_*, pareto_*, etc. (14 PDF)  # Hình Ch4
└── tables/
    └── 5 file .tex           # Bảng dữ liệu IEEE (đã có \caption + \label)
```

---

## Cách upload lên Overleaf

### Bước 1: Tải lên project ZIP
1. Truy cập https://overleaf.com → Đăng nhập
2. Click **New Project** → **Upload Project**
3. Chọn file ZIP đã tải về (cuối cùng) → đợi upload xong

### Bước 2: Đổi compiler thành XeLaTeX
1. Vào project mới tạo
2. Click **Menu** (góc trên bên trái) → **Settings**
3. Tại mục **Compiler**, chọn **XeLaTeX**
4. **Main document:** chọn `main.tex`
5. Click **Save** hoặc đóng Menu

### Bước 3: Compile
1. Bấm **Recompile** (hoặc Ctrl+Enter / Cmd+Enter)
2. Lần đầu compile sẽ chậm (~30s) vì cần load font Times New Roman
3. Sau khi compile thành công → PDF hiển thị bên phải

### Bước 4: Tải PDF về
- Click icon **Download PDF** (góc trên bên phải khu vực preview)
- File PDF khoảng 80-100 trang, đầy đủ format UTH

---

## Xử lý compile chậm / timeout trên Overleaf Free

Overleaf Free giới hạn **30s compile time**. Với luận văn 80-100 trang dùng XeLaTeX + polyglossia, lần compile đầu có thể vượt 30s. Có 4 cách xử lý:

### Cách 1 — Compile từng phần khi đang sửa (KHUYẾN NGHỊ)
Mở `main.tex`, bỏ comment 1 trong các dòng `\includeonly`:
```latex
\includeonly{chapters/04_chuong_3}     % chỉ compile Ch3 → < 10s
\includeonly{chapters/05_chuong_4}     % chỉ compile Ch4 → < 10s
```
Khi build BẢN HOÀN CHỈNH để nộp: comment lại dòng trên (như mặc định).

### Cách 2 — Sử dụng Overleaf Free Trial (14 ngày)
Click **Start free trial** ngay trên Overleaf → có 240s compile time + nhiều benefits khác. Khi hết trial vẫn tải PDF được.

### Cách 3 — Compile lần 2 trở đi (cache)
Overleaf cache kết quả compile lần trước. Sau lần đầu (timeout), thử **Recompile** lại 2-3 lần — các lần sau thường nhanh hơn (15-25s) nhờ cache.

### Cách 4 — Compile local (TeX Live trên máy)
Cài MacTeX/TeX Live → mở terminal:
```bash
cd LUAN_VAN_THANH_LATEX/
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```
Compile local không bị giới hạn thời gian.

---

## Lưu ý quan trọng

### Font Times New Roman trên Overleaf
- Overleaf đã cài sẵn font Times New Roman (do XeLaTeX hỗ trợ tự nhiên)
- Nếu compile fail với lỗi `Times New Roman not found`, sửa trong `uth-thesis.cls`:
  ```latex
  \setmainfont{TeX Gyre Termes}  % fallback miễn phí tương thích Times New Roman
  ```

### Logo UTH
- File `uth-thesis.cls` để chỗ trống cho logo (rule trắng `\rule{0pt}{4cm}`)
- Để thêm logo: copy file `logo_uth.png` (300dpi PNG) vào folder `figures/`,
  sau đó sửa trong `uth-thesis.cls`:
  ```latex
  % Trong \makecoverpage và \makeinnercoverpage:
  % Thay rule trắng bằng:
  \includegraphics[width=3cm]{figures/logo_uth.png}
  ```

### Nếu có lỗi LaTeX
- Lỗi `polyglossia + Vietnamese`: đảm bảo compiler là **XeLaTeX**, không phải pdfLaTeX
- Lỗi `cannot find file references.bib`: đảm bảo file đã được upload, vào lại Main → BibTeX run
- Lỗi `\cite{...} undefined`: chạy `bibtex main` trong Overleaf (thường tự động)

---

## Mapping Citations ↔ Paper File

Tổng số: **58 entries** trong `references.bib`. Toàn bộ 37 citations dùng trong luận văn đều được cross-check OK.

Các file paper trong folder `papers/` của project được ánh xạ:

| Citation key | File PDF trong `papers/` |
|---|---|
| `mao2022rsma` | 220103192v3.pdf |
| `mao2018rsma` | (well-known, đặt rời) |
| `liu2021starris` | (well-known IEEE WC Mag 2021) |
| `mu2022simultaneously` | (well-known IEEE TWC 2022) |
| `wu2020smart` | (well-known IEEE CommMag 2020) |
| `hieu2021optimal` | hieu2021.pdf / 2107.00238v2.pdf |
| `irkicatal2024deep` | 2403.05974v2.pdf / 240305974v2.pdf |
| `hua2023learning` | Learning-based_Reconfigurable_Intelligent_Surface-aided_*.pdf |
| `wu2024deep` | 2209.08456v3.pdf |
| `diamanti2023energy` | OJCOMS-DQL2023_published_paper.pdf |
| `bao2025heuristic` | 2501.12311v1.pdf |
| `faramarzi2024meta` | 2403.08648v1.pdf / AARIS_new.pdf / Main.pdf |
| `ma2025drl` | 2501.15091v1.pdf / Deep_Reinforcement_Learning_for_Energy_*.pdf |
| `zhang2024model` | 2405.01515v2.pdf |
| `cernaloli2023meta` | 2307.08822v2.pdf |
| `wang2024rsbnn` | 2407.06530v1.pdf |
| `huang2024starrisuav` | 251201202v1.pdf |
| `iqbal2024twin` | Twin_Delayed_Deep_Deterministic_*.pdf |
| `soleymani2024starris` | STAR-RIS-aided_RSMA_for_the_URLLC_*.pdf |
| `gomes2024performance` | Performance_of_STAR-RIS-Aided_RSMA_Networks_*.pdf |
| `amiri2024meta` | Resource_Allocation_in_STAR-RIS-Aided_SWIPT_*.pdf |
| `singh2024robust` | Robust_UAV-Integrated_Active_STAR-RIS_*.pdf |
| `ibrahim2025cognitive` | Cognitive-Radio_Functionality_*.pdf |
| `zhang2024covertstar` | FINALVERSION (1).pdf |
| `zhang2024starrismatrans` | FINALVERSION.pdf |
| `yang2024covertambc` | FINAL_VERSION5.pdf |
| `jin2024arisrsma` | ARIS-RSMA_Enhanced_ISAC_*.pdf |
| `to2024fairness` | Fairness-Aware_Secure_Communication_*.pdf |
| `kamal2024optimizing` | Optimizing_Secure_Multi-User_ISAC_*.pdf |
| `perdana2024adaptive` | Adaptive_Ground_User_Pairing_in_Flying-STAR-RIS_*.pdf |
| `noor2025aimulti` | TSP_CMES_73200.pdf / A_Comprehensive_Survey_*.pdf |
| `priya2024noma` | Survey_of_Cooperative_NOMA_*.pdf |
| `adams2024rsma` | RSMA-enabled_RIS-assisted_Integrated_*.pdf |
| `jang2024rsma` | electronics-13-04579.pdf |
| `noman2026drl6g` | Deep_Reinforcement_Learning_for_Resource_Management_*.pdf |
| `dange2026collaborative` | Collaborative_Multi-Agent_DRL_*.pdf |
| `mirza2024multi` | WIP_GC2024ws__Changes_.pdf |
| `mousavi2025fed` | 2501.07126v1.pdf |
| `truong2024energy` | truong_ijsac_2024_open.pdf |
| `nguyen2025star` | 9649_Văn_ban_*.pdf (VN paper) |
| `le2024qoe` | 9141_Văn_ban_*.pdf (VN paper) |
| `pham2020d2dnoma` | Tap_chi_KHDT_tap_9_*.pdf (VN paper) |

Các seminal papers **không nằm trong folder `papers/`** mà tôi tự thêm vào (well-known references):
- `saad2020vision6g` - Saad et al. "A Vision of 6G Wireless Systems", IEEE Network 2020
- `tataria20216g` - Tataria et al. "6G Wireless Systems", Proc. IEEE 2021
- `liu2021starris` - Liu et al. "STAR: Simultaneous Transmission and Reflection", IEEE WC Mag 2021
- `mu2022simultaneously` - Mu et al. "Simultaneously Transmitting and Reflecting RIS", IEEE TWC 2022
- `wu2020smart` - Wu & Zhang "Towards Smart and Reconfigurable Environment", IEEE CommMag 2020
- `lowe2017multi` - Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive", NeurIPS 2017
- `sutton2018reinforcement` - Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed., MIT Press 2018
- `mnih2015human` - Mnih et al. "Human-level control through DRL", Nature 2015
- `lillicrap2016continuous` - Lillicrap et al. "Continuous Control with DRL", ICLR 2016 (DDPG)
- `fujimoto2018addressing` - Fujimoto et al. "Addressing Function Approximation Error", ICML 2018 (TD3)
- `schulman2017proximal` - Schulman et al. "Proximal Policy Optimization", arXiv 2017 (PPO)

---

## Checklist trước khi nộp

- [ ] Compile thành công, không có error
- [ ] Page count: 80-100 trang
- [ ] Mục lục tự động đầy đủ chương/mục
- [ ] Danh mục bảng (~6 bảng): có
- [ ] Danh mục hình (~26 hình): có
- [ ] Tất cả `\cite{...}` đều có entry trong `references.bib`
- [ ] Bibliography theo IEEE style (`\bibliographystyle{IEEEtran}`)
- [ ] Logo UTH đã thêm vào (nếu cần - hiện đang để trống `\rule{0pt}{4cm}`)
- [ ] Tên tác giả, người hướng dẫn, ngành, mã số đúng (xem `main.tex`)

---

## Liên hệ hỗ trợ

Nếu gặp lỗi compile, kiểm tra log trong Overleaf:
- Click **Logs and output files** (biểu tượng `i` ở góc dưới preview)
- Tìm dòng `Error:` hoặc `Fatal`

Lỗi thường gặp:
1. **Font not found** → đảm bảo XeLaTeX, không phải pdfLaTeX
2. **Polyglossia error** → cập nhật Overleaf TeX Live image (Menu → Settings → TeX Live version → 2023+)
3. **Missing figure** → kiểm tra file `figures/h*.png` và `figures/*.pdf` đã upload đầy đủ
