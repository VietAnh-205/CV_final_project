#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trình chấm Bài 3 (Đọc biển số xe) để thí sinh chạy trên máy của mình.

Hàm score() tính điểm giống hệt metric của SkillPixel. Điểm là tỉ lệ ảnh đọc đúng hoàn toàn biển số. Trước khi so sánh,
cả đáp án và dự đoán được đổi sang chữ hoa và bỏ mọi ký tự không phải chữ cái từ A đến Z hay chữ số, nên "51F-123.45",
"51f12345" và "51F 123 45" được coi là như nhau.

Cách dùng:
    python3 scorer.py du_doan.csv --gt dap_an.csv
    python3 scorer.py du_doan.csv --gt dap_an.csv --show 10     # in thêm 10 ảnh bị đọc sai

Tệp đáp án và tệp dự đoán đều cần hai cột id và plate. Nếu tệp đáp án có thêm cột lines, tức số dòng chữ trên biển như
trong train.csv, script in thêm tỉ lệ đọc đúng riêng cho biển một dòng và biển hai dòng.
"""
import argparse
import sys

import pandas as pd


def _ten_cot_sach(ten):
    """Bỏ ký tự BOM và khoảng trắng thừa trong một tên cột. Tệp CSV lưu bằng Excel thường có ký tự BOM ở đầu tệp."""
    return str(ten).replace("\ufeff", "").strip()


def _chuan_hoa_ten_cot(bang):
    """Làm sạch mọi tên cột của bảng."""
    return bang.rename(columns=_ten_cot_sach)


def _chuan_hoa_bien_so(gia_tri):
    """Đổi biển số sang chữ hoa rồi chỉ giữ chữ cái từ A đến Z và chữ số, nên "51F-123.45" và "51f12345" như nhau."""
    if pd.isna(gia_tri):
        return ""
    ket_qua = ""
    for ky_tu in str(gia_tri).upper():
        if "A" <= ky_tu <= "Z" or "0" <= ky_tu <= "9":
            ket_qua += ky_tu
    return ket_qua


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    try:
        cot_id = row_id_column_name
        solution = _chuan_hoa_ten_cot(solution)
        submission = _chuan_hoa_ten_cot(submission)

        dap_an = pd.DataFrame({
            cot_id: solution[cot_id].astype(str).str.strip(),
            "dung": solution["plate"].map(_chuan_hoa_bien_so),
        })

        # Bài nộp thiếu cột id hoặc thiếu cột plate được 0 điểm.
        if cot_id not in submission.columns or "plate" not in submission.columns:
            return 0.0
        du_doan = pd.DataFrame({
            cot_id: submission[cot_id].astype(str).str.strip(),
            "du_doan": submission["plate"].map(_chuan_hoa_bien_so),
        })
        # Nếu một ảnh có nhiều dòng dự đoán thì chỉ lấy dòng đầu tiên.
        du_doan = du_doan.drop_duplicates(cot_id, keep="first")

        # Ghép dự đoán vào đáp án theo id. Ảnh không có dự đoán nhận chuỗi rỗng, tức là bị tính đọc sai.
        bang = dap_an.merge(du_doan, on=cot_id, how="left")
        doc_dung = (bang["dung"] == bang["du_doan"].fillna("")) & (bang["dung"] != "")
        if len(bang) == 0:
            return 0.0
        return float(doc_dung.mean())
    except Exception:
        # Mọi lỗi khác khi đọc bài nộp đều cho 0 điểm, để việc chấm không bị dừng giữa chừng.
        return 0.0


def khoang_cach_levenshtein(chuoi_a, chuoi_b):
    """Đếm số phép thêm, xoá hoặc thay một ký tự ít nhất để biến chuỗi a thành chuỗi b.

    Con số này gọi là khoảng cách Levenshtein. Hàm tính theo quy hoạch động và chỉ giữ hai hàng của bảng. Ô thứ j của hàng
    truoc là khoảng cách giữa i - 1 ký tự đầu của a và j ký tự đầu của b, còn hàng hien ứng với i ký tự đầu của a.
    """
    truoc = list(range(len(chuoi_b) + 1))
    for i in range(1, len(chuoi_a) + 1):
        hien = [i]
        for j in range(1, len(chuoi_b) + 1):
            # Cách thứ nhất là xoá ký tự thứ i của a.
            sau_khi_xoa = truoc[j] + 1
            # Cách thứ hai là thêm ký tự thứ j của b.
            sau_khi_them = hien[j - 1] + 1
            # Cách thứ ba là thay ký tự thứ i của a bằng ký tự thứ j của b. Phép thay không tốn gì nếu hai ký tự giống nhau.
            if chuoi_a[i - 1] == chuoi_b[j - 1]:
                sau_khi_thay = truoc[j - 1]
            else:
                sau_khi_thay = truoc[j - 1] + 1
            hien.append(min(sau_khi_xoa, sau_khi_them, sau_khi_thay))
        truoc = hien
    return truoc[-1]


def main():
    trinh_doc_tham_so = argparse.ArgumentParser(description="Chấm Bài 3 bằng tỉ lệ đọc đúng hoàn toàn")
    trinh_doc_tham_so.add_argument("pred", help="tệp CSV dự đoán")
    trinh_doc_tham_so.add_argument("--gt", required=True, help="tệp CSV đáp án")
    trinh_doc_tham_so.add_argument("--show", type=int, default=0, help="số ảnh bị đọc sai cần in thêm")
    tham_so = trinh_doc_tham_so.parse_args()

    # Đọc mọi cột dưới dạng chuỗi, rồi làm sạch tên cột.
    dap_an = _chuan_hoa_ten_cot(pd.read_csv(tham_so.gt, dtype=str))
    du_doan = _chuan_hoa_ten_cot(pd.read_csv(tham_so.pred, dtype=str))
    if "id" not in du_doan.columns or "plate" not in du_doan.columns:
        sys.exit(f"Tệp dự đoán phải có hai cột id và plate. Các cột đang có là {list(du_doan.columns)}.")

    diem = score(dap_an, du_doan, "id")

    # Ghép dự đoán vào đáp án để tính thêm vài con số giúp đội tìm chỗ mô hình đọc sai.
    # Sau khi ghép, cột plate là biển số đúng còn cột plate_pred là biển số dự đoán.
    bang = dap_an.merge(du_doan.drop_duplicates("id"), on="id", how="left", suffixes=("", "_pred"))
    bang["bien_dung"] = bang["plate"].map(_chuan_hoa_bien_so)
    bang["bien_du_doan"] = bang["plate_pred"].map(_chuan_hoa_bien_so)
    bang["doc_dung"] = bang["bien_dung"] == bang["bien_du_doan"]

    # Đếm số ký tự sai của từng ảnh bằng khoảng cách Levenshtein.
    cac_so_ky_tu_sai = []
    for bien_dung, bien_du_doan in zip(bang["bien_dung"], bang["bien_du_doan"]):
        cac_so_ky_tu_sai.append(khoang_cach_levenshtein(bien_dung, bien_du_doan))
    bang["so_ky_tu_sai"] = cac_so_ky_tu_sai

    so_anh_thieu = int(bang["plate_pred"].isna().sum())
    # Tỉ lệ ký tự sai bằng tổng số ký tự sai chia cho tổng số ký tự của các biển số đúng.
    tong_so_ky_tu = max(1, bang["bien_dung"].str.len().sum())
    ti_le_ky_tu_sai = bang["so_ky_tu_sai"].sum() / tong_so_ky_tu
    so_anh_sai_mot_ky_tu = int((bang["so_ky_tu_sai"] == 1).sum())

    print(f"Tỉ lệ đọc đúng hoàn toàn = {diem:.4f}   "
          f"(số ảnh chấm = {len(bang)}, ảnh không có dự đoán = {so_anh_thieu})")
    print(f"Tỉ lệ ký tự sai = {ti_le_ky_tu_sai:.4f}, ảnh sai đúng một ký tự = {so_anh_sai_mot_ky_tu}")

    # Nếu tệp đáp án có cột lines thì in tỉ lệ đọc đúng riêng cho biển một dòng và biển hai dòng.
    if "lines" in bang.columns:
        for so_dong, nhom in bang.groupby("lines"):
            print(f"- biển {so_dong} dòng: {len(nhom)} ảnh, đọc đúng {nhom['doc_dung'].mean():.4f}")

    if tham_so.show:
        anh_doc_sai = bang[~bang["doc_dung"]].head(tham_so.show)
        print()
        print(f"{tham_so.show} ảnh đầu tiên bị đọc sai:")
        for _, dong in anh_doc_sai.iterrows():
            bien_du_doan = dong["bien_du_doan"]
            if bien_du_doan == "":
                bien_du_doan = "(trống)"
            print(f"- {dong['id']}: đúng là {dong['bien_dung']}, dự đoán là {bien_du_doan}")


if __name__ == "__main__":
    main()
