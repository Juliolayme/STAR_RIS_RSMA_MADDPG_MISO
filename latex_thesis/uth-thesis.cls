% =============================================================================
%  uth-thesis.cls — LaTeX class for UTH (Đại học GTVT TP.HCM) Master Thesis
%  Format theo: HUONG-DAN-CHI-TIET-TRINH-BAY-LUAN-VAN-THAC-SI
%  Compiler: XeLaTeX
% =============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uth-thesis}[2026/05/27 UTH Master Thesis Class]

% Base class: report, 13pt, A4
\LoadClass[13pt,a4paper,oneside]{report}

% ---- Geometry: lề trái 3cm, phải 2cm, trên 2.5cm, dưới 2.5cm ---------------
\RequirePackage[a4paper,left=3cm,right=2cm,top=2.5cm,bottom=2.5cm,
                headheight=14pt,headsep=10pt,footskip=10pt]{geometry}

% ---- Vietnamese support via polyglossia (XeLaTeX) --------------------------
\RequirePackage{polyglossia}
\setmainlanguage{vietnamese}
\setotherlanguage{english}

% ---- Math packages (PHẢI load TRƯỚC fontspec) ------------------------------
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{bm}
\RequirePackage{dsfont}     % cho \mathds{1}
\RequirePackage{mathtools}  % mở rộng amsmath

% ---- Font: Times New Roman (XeLaTeX) ---------------------------------------
\RequirePackage{fontspec}
\setmainfont{Times New Roman}[
  Ligatures=TeX,
  Scale=1.0
]
% Math: dùng Latin Modern Math mặc định (không dùng unicode-math vì xung đột với amssymb)
% Times New Roman cho text, math giữ Computer Modern là chấp nhận được trong luận văn

% ---- Spacing: 1.5 line spacing ---------------------------------------------
\RequirePackage{setspace}
\onehalfspacing

% No paragraph spacing
\setlength{\parskip}{0pt}
\setlength{\parindent}{1.27cm}  % First-line indent (chuẩn VN)

% ---- Common packages -------------------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage{xcolor}
\RequirePackage{xspace}
\RequirePackage{enumitem}
% csquotes bỏ - không hỗ trợ Vietnamese, không cần thiết cho luận văn này
\RequirePackage{float}              % cần cho algorithm[H]
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% ---- Hyperlinks (tối ưu: tắt bookmarks numbering để compile nhanh) ---------
\RequirePackage[
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=blue,
  pdfencoding=auto,
  bookmarks=false  % tắt bookmark để compile nhanh hơn trên Overleaf free
]{hyperref}

% ---- Caption: bảng trên trái đậm, hình dưới giữa đậm -----------------------
\RequirePackage[font=normalsize,labelfont=bf]{caption}
\captionsetup[table]{
  position=above,
  justification=raggedright,
  singlelinecheck=false,
  skip=6pt,
  labelsep=period
}
\captionsetup[figure]{
  position=below,
  justification=centering,
  singlelinecheck=false,
  skip=6pt,
  labelsep=period
}

% ---- Section title formatting ----------------------------------------------
\RequirePackage{titlesec}

% Chapter: CHƯƠNG 1. TÊN — font 14, IN HOA, đậm, căn giữa
\titleformat{\chapter}[hang]
  {\centering\bfseries\fontsize{14}{17}\selectfont\MakeUppercase}
  {CHƯƠNG \thechapter.}{1ex}{}
\titlespacing*{\chapter}{0pt}{-20pt}{20pt}

% Section: 1.1. — font 13, đậm, căn trái
\titleformat{\section}
  {\bfseries\fontsize{13}{16}\selectfont}
  {\thesection.}{0.7ex}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}

% Subsection: 1.1.1. — font 13, in nghiêng, căn trái
\titleformat{\subsection}
  {\itshape\fontsize{13}{16}\selectfont}
  {\thesubsection.}{0.7ex}{}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

% Subsubsection: a. — font 13, in thường, căn trái
\titleformat{\subsubsection}
  {\fontsize{13}{16}\selectfont}
  {\alph{subsubsection}.}{0.7ex}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

% ---- Page numbering: căn giữa trên -----------------------------------------
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Plain style (for chapter-start pages) — also center top
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% ---- Front matter style: KHÔNG đánh số trang (bìa, cam đoan, ToC, ...) -----
\newcommand{\frontmatterstyle}{
  \pagestyle{empty}
  \thispagestyle{empty}
}

% ---- Commands cho trang bìa ------------------------------------------------
\newcommand{\thesisuniversity}[1]{\def\@thesisuniversity{#1}}
\newcommand{\thesisfaculty}[1]{\def\@thesisfaculty{#1}}
\newcommand{\thesistitle}[1]{\def\@thesistitle{#1}}
\newcommand{\thesisauthor}[1]{\def\@thesisauthor{#1}}
\newcommand{\thesisadvisor}[1]{\def\@thesisadvisor{#1}}
\newcommand{\thesismajor}[1]{\def\@thesismajor{#1}}
\newcommand{\thesissubmajor}[1]{\def\@thesissubmajor{#1}}
\newcommand{\thesismajorcode}[1]{\def\@thesismajorcode{#1}}
\newcommand{\thesisyear}[1]{\def\@thesisyear{#1}}
\newcommand{\thesiscity}[1]{\def\@thesiscity{#1}}
% Mặc định chuyên ngành = ngành nếu không khai báo
\def\@thesissubmajor{}

% Trang bìa chính (giống trang bìa phụ, đều có GVHD)
\newcommand{\makecoverpage}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    {\fontsize{14}{17}\bfseries\MakeUppercase \@thesisuniversity \par}
    \vspace{0.5cm}
    {\fontsize{13}{16}\selectfont \@thesisfaculty \par}
    \vspace{0.8cm}

    \includegraphics[width=4.5cm]{figures/logo_uth.png}\par
    \vspace{0.6cm}

    {\fontsize{16}{20}\bfseries\MakeUppercase \@thesisauthor \par}
    \vspace{1cm}

    {\fontsize{20}{24}\bfseries\MakeUppercase \@thesistitle \par}
    \vspace{1.2cm}

    {\fontsize{16}{20}\selectfont LUẬN VĂN THẠC SĨ \par}
    \vspace{0.3cm}
    {\fontsize{14}{17}\selectfont Ngành \quad\quad : \@thesismajor \par}
    \vspace{0.15cm}
    {\fontsize{14}{17}\selectfont Chuyên ngành: \@thesissubmajor \par}
    \vspace{0.15cm}
    {\fontsize{14}{17}\selectfont Mã số \quad\quad : \@thesismajorcode \par}

    \vspace{1cm}
    \begin{flushleft}
    \hspace{2cm}{\fontsize{14}{17}\bfseries NGƯỜI HƯỚNG DẪN:\par}
    \vspace{0.3cm}
    \hspace{2cm}\begin{minipage}{0.7\textwidth}
    {\fontsize{14}{17}\bfseries \@thesisadvisor \par}
    \end{minipage}
    \end{flushleft}

    \vfill
    {\fontsize{13}{16}\selectfont \@thesiscity, năm \@thesisyear \par}
  \end{center}
  \end{titlepage}
}

% Trang bìa phụ (giống bìa chính + thêm người hướng dẫn)
\newcommand{\makeinnercoverpage}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    {\fontsize{14}{17}\bfseries\MakeUppercase \@thesisuniversity \par}
    \vspace{0.5cm}
    {\fontsize{13}{16}\selectfont \@thesisfaculty \par}
    \vspace{0.8cm}

    \includegraphics[width=4.5cm]{figures/logo_uth.png}\par
    \vspace{0.6cm}

    {\fontsize{16}{20}\bfseries\MakeUppercase \@thesisauthor \par}
    \vspace{1cm}

    {\fontsize{20}{24}\bfseries\MakeUppercase \@thesistitle \par}
    \vspace{1.2cm}

    {\fontsize{16}{20}\selectfont LUẬN VĂN THẠC SĨ \par}
    \vspace{0.3cm}
    {\fontsize{14}{17}\selectfont Ngành \quad\quad : \@thesismajor \par}
    \vspace{0.15cm}
    {\fontsize{14}{17}\selectfont Chuyên ngành: \@thesissubmajor \par}
    \vspace{0.15cm}
    {\fontsize{14}{17}\selectfont Mã số \quad\quad : \@thesismajorcode \par}

    \vspace{1cm}
    \begin{flushleft}
    \hspace{2cm}{\fontsize{14}{17}\bfseries NGƯỜI HƯỚNG DẪN:\par}
    \vspace{0.3cm}
    \hspace{2cm}\begin{minipage}{0.7\textwidth}
    {\fontsize{14}{17}\bfseries \@thesisadvisor \par}
    \end{minipage}
    \end{flushleft}

    \vfill
    {\fontsize{13}{16}\selectfont \@thesiscity, năm \@thesisyear \par}
  \end{center}
  \end{titlepage}
}

% ---- TOC depth: hiện đến cấp 3 (1.1.1) ------------------------------------
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}

% ---- Theorem-like environments --------------------------------------------
\newtheorem{definition}{Định nghĩa}[chapter]
\newtheorem{theorem}{Định lý}[chapter]
\newtheorem{lemma}{Bổ đề}[chapter]
\newtheorem{remark}{Nhận xét}[chapter]

% ---- Default math symbols -------------------------------------------------
\newcommand{\E}{\mathbb{E}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bh}{\mathbf{h}}
\newcommand{\bg}{\mathbf{g}}
\newcommand{\bG}{\mathbf{G}}
\newcommand{\bPhi}{\boldsymbol{\Phi}}

\endinput
