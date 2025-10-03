#!/usr/bin/env python3
"""
generate_attendance_report.py

Usage:
    python generate_attendance_report.py /path/to/attendancemain.csv \
        --out report_output_folder --defaulter-threshold 75

Function:
    - Reads attendance CSV exported from OCR of scanned attendance sheets.
    - Normalizes many attendance markers (Present/Absent and OCR variants).
    - Calculates attendance percentage per student.
    - Flags defaulters (threshold configurable).
    - Detects anomalies (unrecognized/blank entries, >100% presence, duplicate dates).
    - Outputs:
        - attendance_report.csv : per-student details and flags
        - summary.txt : overall summary
        - attendance_distribution.png : bar chart of attendance %
        - attendance_report.pdf : polished PDF report
"""

import os
import re
import sys
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Helpers / Normalization
# ---------------------------
PRESENT_VARIANTS = {"present", "p", "pr", "prs", "1", "yes", "y", "att", "attd", "here"}
ABSENT_VARIANTS = {"absent", "a", "ab", "0", "no", "n", "nan", "leave"}
COMMON_OCR_FIXES = {
    "ab-": "ab", "ab_s": "ab", "abS": "ab", "aB": "ab",
    "pbsent": "present", "prescnt": "present", "presevt": "present", "peat": "present"
}

def normalize_token(tok: str):
    """Map raw cell strings to 1 (present), 0 (absent) or None for unknown."""
    if pd.isna(tok):
        return None
    s = str(tok).strip().lower()
    s = re.sub(r"[^\w]", "", s)  # clean punctuation
    if s in COMMON_OCR_FIXES:
        s = COMMON_OCR_FIXES[s]
    if s.isdigit():
        if s == "1": return 1
        if s == "0": return 0
        return None
    if s in PRESENT_VARIANTS: return 1
    if s in ABSENT_VARIANTS: return 0
    if s.startswith("ab"): return 0
    if s.startswith("pr"): return 1
    return None

# ---------------------------
# Core processing
# ---------------------------

def find_attendance_columns(df: pd.DataFrame):
    date_like = []
    date_re = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{2,4}$")
    for col in df.columns:
        if date_re.match(str(col).strip()):
            date_like.append(col)
    if date_like:
        return date_like
    if len(df.columns) > 3:
        return list(df.columns[3:])
    return list(df.columns[2:])

def collapse_duplicate_date_cols(df: pd.DataFrame, attn_cols):
    base_name_map = defaultdict(list)
    for col in attn_cols:
        base = re.sub(r"\.\d+$", "", str(col))
        base_name_map[base].append(col)
    merged = pd.DataFrame(index=df.index)
    new_cols = []
    for base, cols in base_name_map.items():
        if len(cols) == 1:
            merged[base] = df[cols[0]]
        else:
            merged_vals = []
            for i in df.index:
                tokens = [df.at[i, c] for c in cols]
                norm = [normalize_token(t) for t in tokens]
                if any(v == 1 for v in norm):
                    merged_vals.append(1)
                elif all(v == 0 for v in norm if v is not None) and any(v == 0 for v in norm):
                    merged_vals.append(0)
                else:
                    val = next((v for v in norm if v is not None), None)
                    merged_vals.append(val)
            merged[base] = merged_vals
        new_cols.append(base)
    return merged, new_cols

def compute_attendance(df_raw: pd.DataFrame, defaulter_threshold: float=75.0):
    attn_cols = find_attendance_columns(df_raw)
    merged_attn_df, final_day_cols = collapse_duplicate_date_cols(df_raw, attn_cols)
    norm_matrix = merged_attn_df.applymap(normalize_token)

    total_classes = len(final_day_cols)
    presents = norm_matrix.apply(lambda r: sum(1 for v in r if v == 1), axis=1)
    absents = norm_matrix.apply(lambda r: sum(1 for v in r if v == 0), axis=1)
    unknowns = norm_matrix.apply(lambda r: sum(1 for v in r if v is None), axis=1)

    attendance_pct = (presents / total_classes) * 100

    anomalies = []
    for idx in norm_matrix.index:
        reasons = []
        if unknowns.at[idx] > 0:
            reasons.append(f"{unknowns.at[idx]} unknown entries (possible OCR issues)")
        if attendance_pct.at[idx] > 100:
            reasons.append("Attendance > 100%")
        if presents.at[idx] + absents.at[idx] > total_classes:
            reasons.append("More marks than classes")
        anomalies.append("; ".join(reasons) if reasons else "")

    defaulter_flags = attendance_pct < defaulter_threshold

    cols_lower = [c.lower() for c in df_raw.columns]
    def find_col_like(words):
        for w in words:
            for i, c in enumerate(cols_lower):
                if w in c: return df_raw.columns[i]
        return None

    roll_col = find_col_like(["roll"])
    id_col = find_col_like(["student id", "id"])
    name_col = find_col_like(["name", "student"])

    report = pd.DataFrame({
        "Roll": df_raw[roll_col] if roll_col is not None else range(1, len(df_raw)+1),
        "StudentId": df_raw[id_col] if id_col is not None else [None]*len(df_raw),
        "Name": df_raw[name_col] if name_col is not None else [None]*len(df_raw),
        "TotalClasses": total_classes,
        "TotalPresents": presents,
        "TotalAbsents": absents,
        "UnknownEntries": unknowns,
        "AttendancePercent": attendance_pct.round(2),
        "Defaulter": defaulter_flags,
        "Anomalies": anomalies
    })

    summary = {
        "num_students": len(report),
        "total_classes": total_classes,
        "num_defaulters": int(report["Defaulter"].sum()),
        "highest_attendance": float(report["AttendancePercent"].max()),
        "lowest_attendance": float(report["AttendancePercent"].min()),
        "avg_attendance": float(report["AttendancePercent"].mean()),
        "defaulter_threshold": defaulter_threshold
    }

    return report, summary

# ---------------------------
# Save functions
# ---------------------------

def save_report(report_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "attendance_report.csv")
    report_df.to_csv(csv_path, index=False)
    return csv_path

def save_summary(summary, report_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("Attendance Report Summary\n")
        f.write("=========================\n\n")
        f.write(f"Total students: {summary['num_students']}\n")
        f.write(f"Total classes: {summary['total_classes']}\n")
        f.write(f"Average attendance %: {summary['avg_attendance']:.2f}\n")
        f.write(f"Highest attendance %: {summary['highest_attendance']:.2f}\n")
        f.write(f"Lowest attendance %: {summary['lowest_attendance']:.2f}\n")
        f.write(f"Defaulter threshold: {summary['defaulter_threshold']}%\n")
        f.write(f"Number of defaulters: {summary['num_defaulters']}\n\n")
    return path

def plot_distribution(report_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    plot_df = report_df.sort_values("AttendancePercent")
    ax.bar(range(len(plot_df)), plot_df["AttendancePercent"])
    ax.set_xlabel("Students (sorted)")
    ax.set_ylabel("Attendance %")
    ax.set_title("Attendance Distribution")
    ax.axhline(y=plot_df["AttendancePercent"].mean(), linestyle="--", label="Average")
    ax.legend()
    path = os.path.join(out_dir, "attendance_distribution.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

# ---------------------------
# PDF Export
# ---------------------------

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def save_pdf_report(report_df, summary, out_dir, chart_path=None):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "attendance_report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("📊 Attendance Summary Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    summary_data = [
        ["Total Students", summary["num_students"]],
        ["Total Classes", summary["total_classes"]],
        ["Average Attendance %", f"{summary['avg_attendance']:.2f}"],
        ["Highest Attendance %", f"{summary['highest_attendance']:.2f}"],
        ["Lowest Attendance %", f"{summary['lowest_attendance']:.2f}"],
        ["Defaulter Threshold", f"{summary['defaulter_threshold']}%"],
        ["Number of Defaulters", summary["num_defaulters"]],
    ]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    if chart_path and os.path.exists(chart_path):
        elements.append(Paragraph("Attendance Distribution", styles["Heading2"]))
        elements.append(Image(chart_path, width=500, height=250))
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("🚨 Defaulter Students", styles["Heading2"]))
    defaulters = report_df[report_df["Defaulter"]]
    if defaulters.empty:
        elements.append(Paragraph("No defaulters 🎉", styles["Normal"]))
    else:
        def_table_data = [["Roll", "Name", "Attendance %", "Anomalies"]]
        for _, row in defaulters.iterrows():
            def_table_data.append([
                row["Roll"],
                row["Name"],
                f"{row['AttendancePercent']:.2f}",
                row["Anomalies"] if row["Anomalies"] else "-"
            ])
        def_table = Table(def_table_data, colWidths=[60, 200, 100, 200])
        def_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.red),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ffe6e6")),
        ]))
        elements.append(def_table)

    doc.build(elements)
    return pdf_path

# ---------------------------
# CLI
# ---------------------------

def main(argv):
    parser = argparse.ArgumentParser(description="Generate attendance report from CSV exported from scanned sheets.")
    parser.add_argument("csvfile", help="Path to attendance CSV file")
    parser.add_argument("--out", default="./attendance_report_output", help="Output directory")
    parser.add_argument("--defaulter-threshold", type=float, default=75.0, help="Threshold % for defaulters")
    args = parser.parse_args(argv)

    if not os.path.exists(args.csvfile):
        print("ERROR: CSV file not found:", args.csvfile)
        return 2

    try:
        df = pd.read_csv(args.csvfile)
    except Exception:
        df = pd.read_csv(args.csvfile, encoding="latin1")

    report_df, summary = compute_attendance(df, defaulter_threshold=args.defaulter_threshold)

    csv_out = save_report(report_df, args.out)
    sum_out = save_summary(summary, report_df, args.out)
    plot_out = plot_distribution(report_df, args.out)
    pdf_out = save_pdf_report(report_df, summary, args.out, chart_path=plot_out)

    print("Report generated:")
    print(" - CSV:", csv_out)
    print(" - Summary:", sum_out)
    print(" - Chart:", plot_out)
    print(" - PDF:", pdf_out)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))