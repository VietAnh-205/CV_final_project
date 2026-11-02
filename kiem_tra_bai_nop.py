#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra tệp zip nộp bài cuối theo điều kiện của vòng thi OLP AI, và chấm các tệp zip cho ban tổ chức.

Thí sinh kiểm tra tệp zip của mình bằng lệnh dưới đây. Thư mục public/ là thư mục tạo ra khi giải nén input.zip của
bài, và có tệp test.csv.
    python3 public/kiem_tra_bai_nop.py bai1_<ten_doi>.zip --du-lieu public

Ban tổ chức kiểm tra và chấm mọi tệp zip trong một thư mục bằng lệnh dưới đây, với script nằm ở thư mục gốc của bộ đề.
    python3 kiem_tra_bai_nop.py <thư mục chứa các tệp zip> --cham

Tên tệp zip phải chứa bai1, bai2 hoặc bai3 để script biết đó là bài nào. Nếu tên tệp không có, hãy dùng tuỳ chọn --bai.
Một tệp zip hợp lệ chứa đúng một tệp .csv là kết quả trên tập test, và một tệp .ipynb là notebook đã tạo ra tệp kết
quả đó.

Bài nộp không hợp lệ khi gặp một trong các lỗi sau.
  - Tệp zip không chứa đúng một tệp .csv và một tệp .ipynb.
  - Tệp .csv có dòng tiêu đề sai, có dòng không đúng hai cột, có mã bị lặp, hoặc thiếu hay thừa mã so với tập test.
  - Notebook có ô code chưa chạy, hoặc số thứ tự chạy của các ô code không tăng đều 1, 2, 3, ... từ trên xuống dưới.
  - Notebook cài thêm thư viện, import thư viện ngoài danh sách, hoặc tải model pretrained dạng Transformer.

Script còn in các dòng "cần xem lại" khi notebook có đường link, tải tài nguyên pretrained cơ bản (fastText, gensim,
NLTK), dùng trọng số pretrained của torchvision ở Bài 1 hoặc Bài 2, vẽ chữ lên ảnh ở Bài 3, không nhắc tới tên tệp kết
quả trong code, hoặc không có output nào. Script cũng in các dòng này khi tệp .csv có giá trị mà trình chấm sẽ tính 0
điểm. Các dòng "cần xem lại" không làm bài nộp mất hiệu lực, nhưng cần một người đọc lại phần tương ứng.
"""
import argparse
import ast
import csv
import io
import json
import math
import os
import re
import sys
import zipfile

# Hai cột của tệp kết quả ở từng bài. Cột đầu tiên là mã của dòng.
CAC_COT = {1: ["customer_id", "spend_90d"], 2: ["id", "order"], 3: ["id", "plate"]}
# Thư mục của từng bài trong bộ đề.
THU_MUC_BAI = {1: "task1_spend", 2: "task2_order", 3: "task3_plate"}
# Tên module cấp ngoài cùng của các thư viện trong Bảng 1 của hướng dẫn vòng thi. Module scorer là trình chấm đi kèm đề.
# Các module thuộc thư viện chuẩn của Python cũng được phép, và được kiểm tra riêng trong hàm kiem_tra_notebook.
THU_VIEN_DUOC_PHEP = {
    "torch", "tensorflow", "keras", "tf_keras", "sklearn", "xgboost", "catboost", "transformers", "spacy", "nltk",
    "gensim", "fasttext", "pandas", "numpy", "scipy", "csv", "json", "pickle", "cv2", "PIL", "torchvision", "skimage",
    "matplotlib", "seaborn", "plotly", "autoviz", "joblib", "datasets", "evaluate", "os", "sys", "re", "itertools",
    "collections", "time", "pdb", "pytorch_lightning", "lightning", "tensorboard", "tqdm", "IPython", "__future__",
    "scorer",
}
# Các mẫu code làm bài nộp không hợp lệ, kèm lời giải thích được in ra.
MAU_BI_CAM = [
    (re.compile(r"^\s*[!%]\s*(pip|pip3|conda|mamba|apt|apt-get)\b|\b(pip|conda)\s+install\b", re.M),
     "có lệnh cài thêm thư viện"),
    (re.compile(r"\b(hf_hub_download|snapshot_download|huggingface_hub|sentence_transformers)\b"),
     "tải model hoặc dữ liệu từ Hugging Face"),
    (re.compile(r"\bpipeline\s*\(\s*[\"']"),
     "dùng pipeline pretrained của transformers"),
]
# Các mẫu code không làm bài nộp mất hiệu lực nhưng cần một người đọc lại, kèm lời giải thích được in ra.
MAU_CAN_XEM_LAI = [
    (re.compile(r"https?://"), "có đường link trong code"),
    (re.compile(r"\b(wget|curl)\b"), "có lệnh tải tệp (wget hoặc curl)"),
    (re.compile(r"fasttext\.util\.download_model|gensim\.downloader|api\.load\s*\(|nltk\.download|torch\.hub"),
     "tải tài nguyên pretrained cơ bản, cần xem đó có phải loại được phép không"),
]
# Lệnh from_pretrained cùng với đối số đầu tiên của nó.
MAU_FROM_PRETRAINED = re.compile(r"from_pretrained\s*\(\s*([^,)]*)")
# Tên một model trên Hugging Face, có dạng "tên_model" hoặc "tác_giả/tên_model".
MAU_TEN_MODEL = re.compile(r"[\w.-]+/[\w.-]+|[\w.-]+")
# Đối số weights khi tạo model của torchvision, ví dụ weights="DEFAULT" hoặc weights=ResNet18_Weights.DEFAULT.
# Bài 3 (thị giác máy tính) được dùng trọng số pretrained cơ bản của torchvision, nên script chỉ nhắc ở Bài 1 và Bài 2.
MAU_TRONG_SO = re.compile(r"weights\s*=\s*[\"'A-Z]")
# Bài 3 không cho tự vẽ biển số mới bằng font chữ, nên mọi chỗ vẽ chữ lên ảnh cần một người đọc lại.
MAU_VE_CHU = re.compile(r"\bImageFont\b|\bputText\b|\.(ttf|otf)\b")


def ma_nguon(o):
    """Trả về mã nguồn của một ô notebook. Tệp .ipynb lưu mã nguồn của mỗi ô thành danh sách các dòng."""
    return "".join(o.get("source", []))


def doc_cac_o_code(noi_dung):
    """Đọc notebook từ nội dung của tệp .ipynb và trả về các ô code không trống."""
    notebook = json.loads(noi_dung.decode("utf-8"))
    cac_o_code = []
    for o in notebook.get("cells", []):
        if o.get("cell_type") == "code" and ma_nguon(o).strip():
            cac_o_code.append(o)
    return cac_o_code


def tim_cac_module_duoc_import(cac_o_code):
    """Tìm tên module cấp ngoài cùng trong mọi lệnh import, ví dụ sklearn trong lệnh from sklearn.svm import SVC."""
    cac_module = set()
    for o in cac_o_code:
        # Bỏ các dòng lệnh đặc biệt của Jupyter bắt đầu bằng % hoặc !, vì Python không đọc được các dòng này.
        cac_dong_python = []
        for dong in ma_nguon(o).split("\n"):
            if not dong.lstrip().startswith(("%", "!")):
                cac_dong_python.append(dong)
        code = "\n".join(cac_dong_python)

        try:
            cay_cu_phap = ast.parse(code)
        except SyntaxError:
            # Ô code có lỗi cú pháp thì tìm tên module trong các dòng bắt đầu bằng import hoặc from.
            cac_module |= set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", code, re.M))
            continue
        for nut in ast.walk(cay_cu_phap):
            if isinstance(nut, ast.Import):
                for ten in nut.names:
                    cac_module.add(ten.name.split(".")[0])
            elif isinstance(nut, ast.ImportFrom) and nut.module and not nut.level:
                # Lệnh import tương đối như from . import x có level lớn hơn 0, và không nhắc tới thư viện nào.
                cac_module.add(nut.module.split(".")[0])
    return cac_module


def kiem_tra_notebook(noi_dung, ten_tep_csv, so_bai):
    """Kiểm tra notebook trong tệp zip. Hàm trả về danh sách lỗi và danh sách điều cần xem lại."""
    cac_loi = []
    can_xem_lai = []
    try:
        cac_o_code = doc_cac_o_code(noi_dung)
    except Exception as e:
        return [f"không đọc được notebook ({e})"], []
    if len(cac_o_code) == 0:
        return ["notebook không có ô code nào"], []

    # Notebook phải được chạy một lượt từ trên xuống, nên số thứ tự chạy của các ô code phải là 1, 2, 3, ...
    cac_so_thu_tu = []
    so_o_chua_chay = 0
    for o in cac_o_code:
        so_thu_tu = o.get("execution_count")
        cac_so_thu_tu.append(so_thu_tu)
        if so_thu_tu is None:
            so_o_chua_chay += 1
    if so_o_chua_chay > 0:
        cac_loi.append(f"có {so_o_chua_chay} ô code chưa được chạy")
    elif cac_so_thu_tu != list(range(1, len(cac_so_thu_tu) + 1)):
        dang_la = str(cac_so_thu_tu[:8])
        if len(cac_so_thu_tu) > 8:
            dang_la += "..."
        cac_loi.append(f"số thứ tự chạy của các ô code không phải 1, 2, 3, ... từ trên xuống dưới (đang là {dang_la}). "
                       "Hãy chọn Restart & Run All rồi lưu lại notebook")

    # Notebook cần còn output để người chấm xem được quá trình chạy.
    co_output = False
    for o in cac_o_code:
        if o.get("outputs"):
            co_output = True
    if not co_output:
        can_xem_lai.append("notebook không có output nào, nên không xem được quá trình chạy")

    # Ghép mã nguồn của mọi ô code để tìm các mẫu code bị cấm và các mẫu code cần xem lại.
    cac_ma_nguon = []
    for o in cac_o_code:
        cac_ma_nguon.append(ma_nguon(o))
    toan_bo_code = "\n".join(cac_ma_nguon)

    for mau, loi in MAU_BI_CAM:
        if mau.search(toan_bo_code):
            cac_loi.append(loi)

    # Gọi from_pretrained với một chuỗi giống tên model trên Hugging Face nghĩa là tải model pretrained từ Internet.
    # Chuỗi bắt đầu bằng ".", "/" hoặc "~" là đường dẫn tới một thư mục trên máy, thường chứa model do đội tự train.
    # Nếu đối số là một biến thì script không biết giá trị của biến, nên cần một người đọc lại.
    for ket_qua_tim in MAU_FROM_PRETRAINED.finditer(toan_bo_code):
        doi_so = ket_qua_tim.group(1).strip()
        la_chuoi = len(doi_so) >= 2 and doi_so[0] in "\"'" and doi_so[-1] == doi_so[0]
        if not la_chuoi:
            can_xem_lai.append(f"có lệnh from_pretrained({doi_so}), cần xem đối số là thư mục model tự train "
                               "hay tên model tải về")
            continue
        chuoi = doi_so[1:-1]
        la_duong_dan = chuoi.startswith((".", "/", "~"))
        if not la_duong_dan and MAU_TEN_MODEL.fullmatch(chuoi):
            cac_loi.append(f"from_pretrained({doi_so}) có dạng tên model trên Hugging Face")

    for mau, dieu_can_xem in MAU_CAN_XEM_LAI:
        if mau.search(toan_bo_code):
            can_xem_lai.append(dieu_can_xem)
    if so_bai != 3 and MAU_TRONG_SO.search(toan_bo_code):
        can_xem_lai.append("dùng trọng số pretrained của torchvision trong bài không phải thị giác máy tính")
    if so_bai == 3 and MAU_VE_CHU.search(toan_bo_code):
        can_xem_lai.append("có code vẽ chữ lên ảnh (ImageFont, putText hoặc tệp font), "
                           "cần xem notebook có tự vẽ biển số mới không")

    # Mọi thư viện được import phải nằm trong danh sách được phép hoặc thuộc thư viện chuẩn của Python.
    # Từ Python 3.10, sys.stdlib_module_names chứa tên các module của thư viện chuẩn.
    thu_vien_chuan = getattr(sys, "stdlib_module_names", set())
    cac_module_ngoai = []
    for module in sorted(tim_cac_module_duoc_import(cac_o_code)):
        if module not in THU_VIEN_DUOC_PHEP and module not in thu_vien_chuan:
            cac_module_ngoai.append(module)
    if cac_module_ngoai:
        cac_loi.append("có import thư viện ngoài danh sách, gồm " + ", ".join(cac_module_ngoai))

    # Tệp kết quả trong zip phải do notebook ghi ra, nên tên tệp thường xuất hiện trong code.
    if ten_tep_csv:
        ten_tep = os.path.basename(ten_tep_csv)
        if ten_tep not in toan_bo_code:
            can_xem_lai.append(f"không thấy tên tệp {ten_tep} trong code, cần chắc chắn tệp này do notebook tạo ra")
    return cac_loi, can_xem_lai


def kiem_tra_csv(noi_dung, so_bai, cac_ma_test, nhan_cau_cua_doan):
    """Kiểm tra tệp kết quả .csv trong tệp zip.

    Hàm trả về danh sách lỗi, danh sách điều cần xem lại, và một từ điển từ mã của mỗi dòng tới giá trị dự đoán.
    Từ điển này là None khi tệp không đọc được hoặc sai định dạng.
    """
    cac_loi = []
    can_xem_lai = []
    cac_cot = CAC_COT[so_bai]
    try:
        # Bảng mã utf-8-sig bỏ qua ký tự BOM ở đầu tệp, nếu có.
        cac_dong = list(csv.reader(io.StringIO(noi_dung.decode("utf-8-sig"))))
    except Exception as e:
        return [f"không đọc được tệp csv ({e})"], [], None

    if len(cac_dong) == 0:
        return [f"dòng tiêu đề phải là {','.join(cac_cot)}, nhưng tệp đang trống"], [], None
    tieu_de = []
    for ten_cot in cac_dong[0]:
        tieu_de.append(ten_cot.strip())
    if tieu_de != cac_cot:
        return [f"dòng tiêu đề phải là {','.join(cac_cot)}, nhưng đang là {','.join(cac_dong[0])}"], [], None

    # Bỏ các dòng trống, rồi kiểm tra mỗi dòng còn lại có đúng hai cột.
    cac_dong_du_lieu = []
    for dong in cac_dong[1:]:
        if dong:
            cac_dong_du_lieu.append(dong)
    cac_dong_sai_so_cot = []
    for so_dong, dong in enumerate(cac_dong_du_lieu, start=2):
        if len(dong) != len(cac_cot):
            cac_dong_sai_so_cot.append(so_dong)
    if cac_dong_sai_so_cot:
        cac_loi.append(f"{len(cac_dong_sai_so_cot)} dòng không có đúng {len(cac_cot)} cột "
                       f"(ví dụ dòng {cac_dong_sai_so_cot[0]})")
        return cac_loi, can_xem_lai, None

    # Ghi lại giá trị dự đoán của từng mã. Nếu một mã xuất hiện nhiều lần thì giữ dòng đầu tiên, giống trình chấm.
    du_doan = {}
    so_dong_lap = 0
    for ma, gia_tri in cac_dong_du_lieu:
        if ma in du_doan:
            so_dong_lap += 1
        else:
            du_doan[ma] = gia_tri
    if so_dong_lap > 0:
        cac_loi.append(f"{so_dong_lap} dòng có mã trùng với một dòng phía trên")

    # So các mã trong tệp với các mã của tập test.
    if cac_ma_test is not None:
        tap_ma_test = set(cac_ma_test)
        cac_ma_thieu = []
        for ma in cac_ma_test:
            if ma not in du_doan:
                cac_ma_thieu.append(ma)
        cac_ma_thua = []
        for ma in du_doan:
            if ma not in tap_ma_test:
                cac_ma_thua.append(ma)
        if cac_ma_thieu:
            cac_loi.append(f"thiếu {len(cac_ma_thieu)} mã của tập test (ví dụ {cac_ma_thieu[0]})")
        if cac_ma_thua:
            cac_loi.append(f"có {len(cac_ma_thua)} mã không thuộc tập test (ví dụ {cac_ma_thua[0]})")

    if so_bai == 1:
        # Đếm các giá trị mà trình chấm sẽ tính là dự đoán 0. Giá trị âm vô cực được đếm ở cả hai nhóm.
        so_dong_khong_phai_so = 0
        so_dong_am = 0
        for gia_tri in du_doan.values():
            try:
                so = float(gia_tri)
            except ValueError:
                so = float("nan")
            if not math.isfinite(so):
                so_dong_khong_phai_so += 1
            if so < 0:
                so_dong_am += 1
        if so_dong_khong_phai_so > 0:
            can_xem_lai.append(f"{so_dong_khong_phai_so} dòng có spend_90d không phải số, và trình chấm tính các dòng "
                               "này là dự đoán 0")
        if so_dong_am > 0:
            can_xem_lai.append(f"{so_dong_am} dòng có spend_90d âm, và trình chấm tính các dòng này là dự đoán 0")

    if so_bai == 2 and nhan_cau_cua_doan:
        # Đếm các đoạn mà thứ tự dự đoán không gồm đúng các nhãn câu của đoạn. Trình chấm cho các đoạn này 0 điểm.
        so_doan_sai_nhan = 0
        for ma, gia_tri in du_doan.items():
            if ma not in nhan_cau_cua_doan:
                continue
            cac_nhan = []
            for ky_tu in gia_tri.upper():
                if "A" <= ky_tu <= "Z":
                    cac_nhan.append(ky_tu)
            if sorted(cac_nhan) != nhan_cau_cua_doan[ma]:
                so_doan_sai_nhan += 1
        if so_doan_sai_nhan > 0:
            can_xem_lai.append(f"{so_doan_sai_nhan} đoạn có thứ tự không gồm đúng các nhãn câu của đoạn, mỗi nhãn một "
                               "lần, nên các đoạn này được 0 điểm")

    if so_bai == 3:
        # Đếm các ảnh có ô dự đoán không chứa chữ cái hay chữ số nào.
        so_anh_trong = 0
        for gia_tri in du_doan.values():
            co_ky_tu = False
            for ky_tu in gia_tri:
                if ky_tu.isascii() and ky_tu.isalnum():
                    co_ky_tu = True
            if not co_ky_tu:
                so_anh_trong += 1
        if so_anh_trong > 0:
            can_xem_lai.append(f"{so_anh_trong} ảnh không có biển số dự đoán")
    return cac_loi, can_xem_lai, du_doan


def doc_ma_tu_test(thu_muc_du_lieu, so_bai):
    """Đọc mã của tập test từ tệp test.csv trong thư mục dữ liệu của bài.

    Ở Bài 2, hàm đọc thêm nhãn các câu của từng đoạn. Nếu thư mục không có test.csv thì hàm trả về None.
    """
    duong_dan = os.path.join(thu_muc_du_lieu, "test.csv")
    if not os.path.exists(duong_dan):
        return None, None
    with open(duong_dan, encoding="utf-8", newline="") as tep:
        cac_dong = list(csv.DictReader(tep))
    cot_ma = CAC_COT[so_bai][0]
    if len(cac_dong) == 0 or cot_ma not in cac_dong[0]:
        sys.exit(f"Tệp {duong_dan} không có cột {cot_ma}. "
                 f"Hãy kiểm tra xem --du-lieu có phải thư mục dữ liệu của Bài {so_bai} không.")

    # Ở Bài 2, mỗi đoạn có nhiều dòng trong test.csv. Mỗi dòng là một câu, và nhãn của câu nằm ở cột sentence_id.
    nhan_cau_cua_doan = {}
    if so_bai == 2:
        for dong in cac_dong:
            ma = dong["id"]
            if ma not in nhan_cau_cua_doan:
                nhan_cau_cua_doan[ma] = []
            nhan_cau_cua_doan[ma].append(dong["sentence_id"])
        for ma in nhan_cau_cua_doan:
            nhan_cau_cua_doan[ma] = sorted(nhan_cau_cua_doan[ma])

    # Mỗi mã chỉ lấy một lần, theo thứ tự xuất hiện trong test.csv.
    cac_ma = []
    da_gap = set()
    for dong in cac_dong:
        ma = dong[cot_ma]
        if ma not in da_gap:
            da_gap.add(ma)
            cac_ma.append(ma)
    return cac_ma, nhan_cau_cua_doan


def doc_ma_tu_dap_an(thu_muc_goc, so_bai):
    """Đọc mã của tập test từ đáp án trong bộ đề. Ở Bài 2, hàm đọc thêm nhãn các câu của từng đoạn từ cột order."""
    duong_dan = os.path.join(thu_muc_goc, THU_MUC_BAI[so_bai], "private", "ground_truth.csv")
    with open(duong_dan, encoding="utf-8", newline="") as tep:
        cac_dong = list(csv.DictReader(tep))
    cot_ma = CAC_COT[so_bai][0]
    cac_ma = []
    for dong in cac_dong:
        cac_ma.append(dong[cot_ma])
    nhan_cau_cua_doan = {}
    if so_bai == 2:
        for dong in cac_dong:
            nhan_cau_cua_doan[dong["id"]] = sorted(dong["order"].split())
    return cac_ma, nhan_cau_cua_doan


def kiem_tra_zip(duong_dan, so_bai, cac_ma_test, nhan_cau_cua_doan):
    """Kiểm tra một tệp zip nộp bài. Hàm trả về danh sách lỗi, danh sách điều cần xem lại và các giá trị dự đoán."""
    try:
        tep_zip = zipfile.ZipFile(duong_dan)
    except Exception as e:
        return [f"không mở được tệp zip ({e})"], [], None

    with tep_zip:
        # Bỏ qua các thư mục, thư mục __MACOSX do máy Mac tự thêm, và các tệp ẩn có tên bắt đầu bằng dấu chấm.
        cac_tep = []
        for ten in tep_zip.namelist():
            if ten.endswith("/") or ten.startswith("__MACOSX/") or os.path.basename(ten).startswith("."):
                continue
            cac_tep.append(ten)
        cac_tep_csv = []
        cac_notebook = []
        for ten in cac_tep:
            if ten.lower().endswith(".csv"):
                cac_tep_csv.append(ten)
            if ten.lower().endswith(".ipynb"):
                cac_notebook.append(ten)

        cac_loi = []
        can_xem_lai = []
        if len(cac_tep_csv) != 1 or len(cac_notebook) != 1 or len(cac_tep) != 2:
            if cac_tep:
                cac_loi.append("tệp zip phải chứa đúng một tệp .csv và một tệp .ipynb, nhưng đang chứa "
                               + ", ".join(cac_tep))
            else:
                cac_loi.append("tệp zip phải chứa đúng một tệp .csv và một tệp .ipynb, nhưng đang trống")

        # Nếu zip có nhiều tệp .csv hay nhiều notebook, script vẫn kiểm tra tệp đầu tiên của mỗi loại để báo thêm lỗi.
        du_doan = None
        if cac_tep_csv:
            noi_dung_csv = tep_zip.read(cac_tep_csv[0])
            loi_csv, xem_lai_csv, du_doan = kiem_tra_csv(noi_dung_csv, so_bai, cac_ma_test, nhan_cau_cua_doan)
            cac_loi += loi_csv
            can_xem_lai += xem_lai_csv
        if cac_notebook:
            ten_tep_csv = None
            if cac_tep_csv:
                ten_tep_csv = cac_tep_csv[0]
            loi_notebook, xem_lai_notebook = kiem_tra_notebook(tep_zip.read(cac_notebook[0]), ten_tep_csv, so_bai)
            cac_loi += loi_notebook
            can_xem_lai += xem_lai_notebook
    return cac_loi, can_xem_lai, du_doan


def cham_diem(so_bai, du_doan, thu_muc_goc):
    """Chấm các giá trị dự đoán trên toàn bộ tập test, bằng đúng metric của nền tảng và đáp án trong bộ đề."""
    # Thí sinh chạy script này mà không cần pandas, vì pandas chỉ được dùng khi ban tổ chức chấm điểm.
    import importlib.util
    import pandas as pd

    thu_muc_bai = os.path.join(thu_muc_goc, THU_MUC_BAI[so_bai])
    dac_ta = importlib.util.spec_from_file_location("metric", os.path.join(thu_muc_bai, "platform", "metric.py"))
    metric = importlib.util.module_from_spec(dac_ta)
    dac_ta.loader.exec_module(metric)

    dap_an = pd.read_csv(os.path.join(thu_muc_bai, "private", "ground_truth.csv"), dtype=str, keep_default_na=False)
    cot_ma, cot_gia_tri = CAC_COT[so_bai]
    bai_nop = pd.DataFrame({cot_ma: list(du_doan.keys()), cot_gia_tri: list(du_doan.values())})
    return metric.score(dap_an, bai_nop, cot_ma)


def tong_diem_ba_bai(doi_va_diem):
    """Trả về tổng điểm ba bài của một đội, dùng để xếp hạng các đội."""
    ten_doi, diem_tung_bai = doi_va_diem
    return sum(diem_tung_bai.values())


def main():
    trinh_doc_tham_so = argparse.ArgumentParser(description="Kiểm tra tệp zip nộp bài cuối và chấm điểm")
    trinh_doc_tham_so.add_argument("paths", nargs="+", help="tệp zip hoặc thư mục chứa tệp zip")
    trinh_doc_tham_so.add_argument("--bai", type=int, choices=[1, 2, 3],
                                   help="số thứ tự của bài, dùng khi tên tệp không chứa bai1, bai2 hay bai3")
    trinh_doc_tham_so.add_argument("--du-lieu",
                                   help="thư mục dữ liệu của bài, có tệp test.csv, để kiểm tra mã của tập test")
    trinh_doc_tham_so.add_argument("--cham", action="store_true",
                                   help="dành cho ban tổ chức, chấm điểm bằng đáp án trong bộ đề")
    trinh_doc_tham_so.add_argument("--ket-qua", default="ket_qua.csv", help="tệp ghi bảng kết quả khi dùng --cham")
    tham_so = trinh_doc_tham_so.parse_args()

    # Thư mục gốc của bộ đề là thư mục chứa script này.
    thu_muc_goc = os.path.dirname(os.path.abspath(__file__))

    # Lập danh sách tệp zip cần kiểm tra. Với một thư mục, lấy mọi tệp .zip trong thư mục đó theo thứ tự tên.
    cac_tep_zip = []
    for duong_dan in tham_so.paths:
        if os.path.isdir(duong_dan):
            cac_tep_trong_thu_muc = []
            for ten in os.listdir(duong_dan):
                if ten.lower().endswith(".zip"):
                    cac_tep_trong_thu_muc.append(os.path.join(duong_dan, ten))
            cac_tep_zip += sorted(cac_tep_trong_thu_muc)
        else:
            cac_tep_zip.append(duong_dan)

    cac_dong_ket_qua = []
    tat_ca_hop_le = True
    for duong_dan in cac_tep_zip:
        ten_tep = os.path.basename(duong_dan)

        # Tuỳ chọn --bai được ưu tiên. Nếu không có, tìm số thứ tự bài trong tên tệp, ví dụ bai2 trong bai2_doi07.zip.
        so_bai = tham_so.bai
        if so_bai is None:
            ket_qua_tim = re.search(r"bai\s*([123])", ten_tep, re.I)
            if ket_qua_tim:
                so_bai = int(ket_qua_tim.group(1))
        if so_bai is None:
            print()
            print(f"{ten_tep}: không biết đây là bài nào. "
                  "Hãy đặt tên tệp có chứa bai1, bai2 hoặc bai3, hoặc dùng tuỳ chọn --bai.")
            tat_ca_hop_le = False
            continue

        # Ban tổ chức lấy mã của tập test từ đáp án, còn thí sinh lấy từ test.csv trong thư mục dữ liệu.
        cac_ma_test = None
        nhan_cau_cua_doan = None
        if tham_so.cham:
            cac_ma_test, nhan_cau_cua_doan = doc_ma_tu_dap_an(thu_muc_goc, so_bai)
        elif tham_so.du_lieu:
            cac_ma_test, nhan_cau_cua_doan = doc_ma_tu_test(tham_so.du_lieu, so_bai)

        cac_loi, can_xem_lai, du_doan = kiem_tra_zip(duong_dan, so_bai, cac_ma_test, nhan_cau_cua_doan)
        hop_le = len(cac_loi) == 0
        if not hop_le:
            tat_ca_hop_le = False

        # Chỉ chấm điểm bài nộp hợp lệ.
        diem = None
        if tham_so.cham and du_doan is not None and hop_le:
            diem = cham_diem(so_bai, du_doan, thu_muc_goc)

        dong_tom_tat = f"{ten_tep} (Bài {so_bai}): "
        if hop_le:
            dong_tom_tat += "HỢP LỆ"
        else:
            dong_tom_tat += "KHÔNG HỢP LỆ"
        if diem is not None:
            dong_tom_tat += f", điểm = {diem:.4f}"
        print()
        print(dong_tom_tat)
        for loi in cac_loi:
            print("  lỗi:", loi)
        for dieu_can_xem in can_xem_lai:
            print("  cần xem lại:", dieu_can_xem)
        if cac_ma_test is None and not tham_so.cham:
            print("  (Chưa kiểm tra mã của tập test. Hãy dùng --du-lieu với thư mục chứa test.csv của bài.)")

        # Tên đội là tên tệp sau khi bỏ phần đuôi và phần bai1, bai2 hoặc bai3, ví dụ doi07 trong bai2_doi07.zip.
        ten_khong_duoi = os.path.splitext(ten_tep)[0]
        ten_doi = re.sub(r"(?i)[_\s-]*bai\s*[123][_\s-]*", "_", ten_khong_duoi).strip("_")
        if ten_doi == "":
            ten_doi = ten_tep
        diem_ghi_ra = ""
        if diem is not None:
            diem_ghi_ra = round(diem, 4)
        cac_dong_ket_qua.append({
            "tep": ten_tep,
            "doi": ten_doi,
            "bai": so_bai,
            "hop_le": hop_le,
            "diem": diem_ghi_ra,
            "loi": " | ".join(cac_loi),
            "can_xem_lai": " | ".join(can_xem_lai),
        })

    if tham_so.cham and cac_dong_ket_qua:
        with open(tham_so.ket_qua, "w", encoding="utf-8", newline="") as tep:
            trinh_ghi = csv.DictWriter(tep, fieldnames=list(cac_dong_ket_qua[0].keys()))
            trinh_ghi.writeheader()
            trinh_ghi.writerows(cac_dong_ket_qua)

        # Cộng điểm ba bài của mỗi đội. Bài không nộp hoặc không hợp lệ được 0 điểm.
        diem_cua_doi = {}
        for dong in cac_dong_ket_qua:
            if dong["doi"] not in diem_cua_doi:
                diem_cua_doi[dong["doi"]] = {1: 0.0, 2: 0.0, 3: 0.0}
            diem_bai = dong["diem"]
            if diem_bai == "":
                diem_bai = 0.0
            diem_cua_doi[dong["doi"]][dong["bai"]] = diem_bai

        print()
        print(f"Đã ghi {tham_so.ket_qua}. Tổng điểm (bài không hợp lệ tính 0 điểm):")
        # Xếp các đội theo tổng điểm từ cao xuống thấp. Các đội bằng điểm giữ thứ tự xuất hiện.
        for ten_doi, diem_tung_bai in sorted(diem_cua_doi.items(), key=tong_diem_ba_bai, reverse=True):
            tong_diem = sum(diem_tung_bai.values())
            print(f"  {ten_doi}: {tong_diem:.4f}  (Bài 1 {diem_tung_bai[1]:.4f}, Bài 2 {diem_tung_bai[2]:.4f}, "
                  f"Bài 3 {diem_tung_bai[3]:.4f})")

    # Mã thoát 1 báo rằng có ít nhất một tệp zip không hợp lệ.
    sys.exit(0 if tat_ca_hop_le else 1)


if __name__ == "__main__":
    main()
