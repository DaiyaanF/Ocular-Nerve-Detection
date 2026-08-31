"""
deltas.py — Pre/Post Flight Delta Calculator
=============================================
GUI tool to:
  1. Select one or more detection_results.csv (fundus/DICOM) or
     oct_metrics.csv (E2E OCT) files — auto-detected
  2. Auto-parse subject ID (EXP-XXXX) and condition (Pre/Post-flight)
     from filenames or folder paths
  3. Compute per-subject mean metrics for each condition
  4. Display a colour-coded delta table (Post − Pre)
  5. Save an HTML report matching the PDF format + a summary CSV

Run:
    python3 deltas.py
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import subprocess


# ──────────────────────────────────────────────────────────────
# WSL-AWARE BROWSER OPENER
# ──────────────────────────────────────────────────────────────

def _open_in_browser(path):
    """Open an HTML file in a browser, handling WSL → Windows handoff."""
    import shutil, platform
    abs_path = os.path.abspath(path)

    # WSL: convert Linux path to Windows path and use cmd.exe start
    if "microsoft" in platform.uname().release.lower() or shutil.which("wslpath"):
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", abs_path], text=True
            ).strip()
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
            return
        except Exception:
            pass

    # Native Linux with a browser
    for browser in ["xdg-open", "firefox", "chromium-browser", "google-chrome"]:
        if shutil.which(browser):
            subprocess.Popen([browser, abs_path])
            return

    # Absolute fallback
    import webbrowser
    webbrowser.open(f"file://{abs_path}")


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

METRICS = [
    ("ONSD_mm",                 "ONSD (mm)"),
    ("CRAE_mm",                 "CRAE (mm)"),
    ("CRVE_mm",                 "CRVE (mm)"),
    ("AVR",                     "AVR"),
    ("Optic_Disc_Radius_mm",    "Optic Disc Radius (mm)"),
    ("Posterior_Globe_Area_mm2","Posterior Globe Area (mm²)"),
    ("Radius_of_Curvature_mm",  "Radius of Curvature (mm)"),
]

PRE_KEYWORDS  = ["pre-flight", "pre_flight", "preflight", "pre flight",
                 "training", "pre"]
POST_KEYWORDS = ["post-flight", "post_flight", "postflight", "post flight",
                 "flight", "post"]

# Mapping from oct_metrics.csv column names → shared METRICS keys
OCT_COLUMN_MAP = {
    'onsd_mm':        'ONSD_mm',
    'crae_mm':        'CRAE_mm',
    'crve_mm':        'CRVE_mm',
    'avr':            'AVR',
    'disc_radius_mm': 'Optic_Disc_Radius_mm',
    'globe_area_mm2': 'Posterior_Globe_Area_mm2',
    'roc_mm':         'Radius_of_Curvature_mm',
}

# ──────────────────────────────────────────────────────────────
# PARSING HELPERS
# ──────────────────────────────────────────────────────────────

def _extract_subject_condition(text):
    """
    Scan a string (filename or folder path) for a subject ID and condition.
    Returns (subject, condition) — either may be None if not found.

    Recognised condition patterns (in priority order):
      Post-flight : post-flight / postflight / flight / _R<N> (return day)
      Pre-flight  : pre-flight / preflight / training / _L<N> (launch day)
    """
    subject = None
    condition = None

    m = re.search(r'(EXP-\d+|TRISH-\d+)', text, re.IGNORECASE)
    if m:
        subject = m.group(1).upper()

    lower = text.lower()

    # Explicit keyword matches first
    if any(kw in lower for kw in ["post-flight", "post_flight", "postflight", "post flight"]):
        condition = "Post-flight"
    elif any(kw in lower for kw in ["pre-flight", "pre_flight", "preflight", "pre flight"]):
        condition = "Pre-flight"
    elif "training" in lower:
        condition = "Pre-flight"
    # E2E filename convention: _R<N> = Return (post-flight), _L<N> = Launch (pre-flight)
    elif re.search(r'_r\d+', lower):
        condition = "Post-flight"
    elif re.search(r'_l\d+', lower):
        condition = "Pre-flight"
    elif "flight" in lower:
        condition = "Post-flight"

    return subject, condition


def parse_subject(filename):
    subject, _ = _extract_subject_condition(filename)
    return subject


def parse_condition(filename):
    _, condition = _extract_subject_condition(filename)
    return condition


def load_and_annotate(csv_path):
    """
    Load a detection_results.csv and add _subject / _condition columns.

    Priority order for each row:
      1. Parse subject + condition from the Filename value itself
      2. If either is missing, scan the full CSV folder path as a fallback
         (handles DICOM UID filenames where the folder encodes subject/condition,
          e.g. Training/TRISH-4449/... or Flight/TRISH-4449/...)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Filename" not in df.columns:
        raise ValueError(f"No 'Filename' column in {csv_path}")

    # Fallback values from the CSV's own folder path
    folder_subject, folder_condition = _extract_subject_condition(
        os.path.abspath(csv_path).replace("\\", "/")
    )

    def _subject(fname):
        s, _ = _extract_subject_condition(str(fname))
        return s or folder_subject

    def _condition(fname):
        _, c = _extract_subject_condition(str(fname))
        return c or folder_condition

    df["_subject"]   = df["Filename"].apply(_subject)
    df["_condition"] = df["Filename"].apply(_condition)
    df["_modality"]  = "Fundus/DICOM"

    df = df[df["Status"] == "SUCCESS"].copy()
    df = df[df["_subject"].notna() & df["_condition"].notna()].copy()

    return df


def _detect_csv_type(csv_path):
    """Return 'oct' or 'fundus' based on the columns present in the CSV."""
    df_head = pd.read_csv(csv_path, nrows=0)
    cols = [c.strip() for c in df_head.columns]
    if "series" in cols or "scan_index" in cols:
        return "oct"
    return "fundus"


def load_and_annotate_oct(csv_path):
    """
    Load an oct_metrics.csv produced by e2e.py and normalise column names to
    match the shared METRICS keys.

    Subject and condition are derived from the CSV's folder path, e.g.:
        .../Flight/EXP-6405_L60/series_00_L/oct_metrics.csv  → Post-flight
        .../Training/TRISH-4449/series_00_L/oct_metrics.csv  → Pre-flight
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Rename to the shared METRICS column names
    df = df.rename(columns=OCT_COLUMN_MAP)

    # Derive subject + condition entirely from the folder path
    folder_path = os.path.abspath(csv_path).replace("\\", "/")
    subject, condition = _extract_subject_condition(folder_path)

    if subject is None:
        raise ValueError(
            f"Could not find a subject ID (EXP-XXXX / TRISH-XXXX) in path:\n{folder_path}"
        )
    if condition is None:
        raise ValueError(
            f"Could not determine Pre/Post-flight condition from path:\n{folder_path}\n"
            "Expected 'Training' (pre) or 'Flight' (post) somewhere in the path."
        )

    df["_subject"]   = subject
    df["_condition"] = condition
    df["_modality"]  = "OCT"
    df["_source"]    = os.path.basename(csv_path)

    # Drop rows where all metric columns are NaN (e.g. failed scans)
    metric_cols = list(OCT_COLUMN_MAP.values())
    available   = [c for c in metric_cols if c in df.columns]
    df = df.dropna(subset=available, how="all").copy()

    return df


# ──────────────────────────────────────────────────────────────
# DELTA CALCULATION
# ──────────────────────────────────────────────────────────────

def compute_deltas(df):
    """
    Returns a DataFrame with one row per subject and columns:
        subject, <metric>_pre, <metric>_post, <metric>_delta
    for each metric in METRICS.
    """
    results = []
    subjects = sorted(df["_subject"].unique())

    for subj in subjects:
        subj_df = df[df["_subject"] == subj]
        pre_df  = subj_df[subj_df["_condition"] == "Pre-flight"]
        post_df = subj_df[subj_df["_condition"] == "Post-flight"]

        row = {"subject": subj,
               "n_pre":  len(pre_df),
               "n_post": len(post_df)}

        for col, _ in METRICS:
            if col not in df.columns:
                row[f"{col}_pre"]   = np.nan
                row[f"{col}_post"]  = np.nan
                row[f"{col}_delta"] = np.nan
                continue

            pre_vals  = pd.to_numeric(pre_df[col],  errors="coerce").dropna()
            post_vals = pd.to_numeric(post_df[col], errors="coerce").dropna()

            pre_mean  = float(pre_vals.mean())  if len(pre_vals)  > 0 else np.nan
            post_mean = float(post_vals.mean()) if len(post_vals) > 0 else np.nan
            delta     = (post_mean - pre_mean)  if not (np.isnan(pre_mean) or np.isnan(post_mean)) else np.nan

            row[f"{col}_pre"]   = pre_mean
            row[f"{col}_post"]  = post_mean
            row[f"{col}_delta"] = delta

        results.append(row)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# HTML REPORT  (matches PDF layout)
# ──────────────────────────────────────────────────────────────

def _fmt(value, dp=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{dp}f}"


def _delta_cell(delta, dp=4):
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return "<td>N/A</td>"
    sign  = "+" if delta >= 0 else ""
    color = "#2e7d32" if delta >= 0 else "#c62828"   # green / red
    return f'<td style="color:{color};font-weight:bold">{sign}{delta:.{dp}f}</td>'


def build_html(delta_df, csv_names, save_path, mod_label="Fundus/DICOM"):
    # Metric-specific decimal places
    dp_map = {
        "ONSD_mm": 4, "CRAE_mm": 4, "CRVE_mm": 4,
        "AVR": 4, "Optic_Disc_Radius_mm": 4,
        "Posterior_Globe_Area_mm2": 2, "Radius_of_Curvature_mm": 4,
    }

    table_blocks = []
    for col, label in METRICS:
        dp = dp_map.get(col, 4)
        rows_html = ""
        all_na = True
        for _, r in delta_df.iterrows():
            pre   = r.get(f"{col}_pre",   np.nan)
            post  = r.get(f"{col}_post",  np.nan)
            delta = r.get(f"{col}_delta", np.nan)

            if not (np.isnan(pre) and np.isnan(post)):
                all_na = False

            rows_html += (
                f"<tr><td>{r['subject']}</td>"
                f"<td>{_fmt(pre, dp)}</td>"
                f"<td>{_fmt(post, dp)}</td>"
                f"{_delta_cell(delta, dp)}</tr>\n"
            )

        note = ""
        if all_na:
            note = ' <span style="color:#888;font-size:0.85em">(not measured for this image type)</span>'

        table_blocks.append(f"""
        <div class="metric-block">
          <table>
            <thead>
              <tr>
                <th class="metric-label">{label}{note}</th>
                <th>Pre-flight</th>
                <th>Post-flight</th>
                <th>Delta (Δ)</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """)

    source_line = ", ".join(csv_names)
    mod_str     = f" [{mod_label}]" if mod_label else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pre/Post Flight Deltas{mod_str}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1   {{ font-size: 1.5em; margin-bottom: 4px; }}
  .source {{ font-size: 0.82em; color: #555; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .metric-block {{ }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  thead tr {{ background: #1a3a5c; color: white; }}
  th  {{ padding: 8px 12px; text-align: left; }}
  td  {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  tbody tr:nth-child(even) {{ background: #f5f5f5; }}
  .metric-label {{ font-weight: bold; }}
  .note {{ margin-top: 32px; font-size: 0.82em; color: #555; font-style: italic; }}
</style>
</head>
<body>
<h1>Detection Results — Pre/Post Flight Deltas{mod_str}</h1>
<div class="source">Source: {source_line} | Delta = Post-flight mean − Pre-flight mean (per subject, averaged across scans)</div>

<div class="grid">
  {''.join(table_blocks)}
</div>

<p class="note">
  * Posterior Globe Area and Radius of Curvature are only measurable from B-scan / OCT images.
    For fundus photographs these columns will show N/A.<br>
  * AVR is physiologically constrained to &lt; 1.0 (arteries always narrower than veins).
    Any result outside this range is excluded as a misdetection.
</p>
</body>
</html>
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    return save_path


# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────

class DeltaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pre/Post Flight Delta Calculator")
        self.root.geometry("900x680")
        self.root.resizable(True, True)

        self.csv_paths  = []
        self.delta_df   = None
        self.html_path  = None

        self._build_ui()

    # ── UI layout ──────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # Top bar
        top = tk.Frame(root, pady=8, padx=10)
        top.pack(fill=tk.X)

        tk.Button(top, text="Select CSV file(s)", command=self._select_csvs,
                  bg="#1a3a5c", fg="white", padx=12, pady=4,
                  relief=tk.FLAT, font=("Arial", 10, "bold")).pack(side=tk.LEFT)

        self.file_label = tk.Label(top, text="No file selected",
                                   fg="#555", font=("Arial", 9))
        self.file_label.pack(side=tk.LEFT, padx=12)

        tk.Button(top, text="Open HTML report", command=self._open_html,
                  padx=10, pady=4, relief=tk.FLAT).pack(side=tk.RIGHT)
        tk.Button(top, text="Save summary CSV", command=self._save_csv,
                  padx=10, pady=4, relief=tk.FLAT).pack(side=tk.RIGHT, padx=6)

        # Status bar
        self.status_var = tk.StringVar(value="Select a detection_results.csv to begin.")
        tk.Label(root, textvariable=self.status_var, fg="#333",
                 font=("Arial", 9), anchor="w", padx=10).pack(fill=tk.X)

        # Notebook with one tab per metric
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

    # ── File selection ─────────────────────────────────────────

    def _select_csvs(self):
        paths = filedialog.askopenfilenames(
            title="Select detection_results CSV file(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not paths:
            return

        self.csv_paths = list(paths)
        names = [os.path.basename(p) for p in self.csv_paths]
        self.file_label.config(text=", ".join(names))
        self._run(names)

    # ── Core logic ─────────────────────────────────────────────

    def _run(self, csv_names):
        try:
            frames = []
            for p in self.csv_paths:
                csv_type = _detect_csv_type(p)
                if csv_type == "oct":
                    df = load_and_annotate_oct(p)
                else:
                    df = load_and_annotate(p)
                df["_source"] = os.path.basename(p)
                frames.append(df)

            if not frames:
                messagebox.showerror("Error", "No valid data loaded.")
                return

            combined   = pd.concat(frames, ignore_index=True)
            n_subjects = combined["_subject"].nunique()
            n_rows     = len(combined)

            modalities = combined["_modality"].unique().tolist()
            mod_label  = " + ".join(sorted(modalities))

            self.delta_df = compute_deltas(combined)

            # Build tabs
            for tab in self.nb.tabs():
                self.nb.forget(tab)

            self._build_overview_tab(mod_label)
            for col, label in METRICS:
                self._build_metric_tab(col, label)

            # Build HTML
            import tempfile as _tf
            tmp_f = _tf.NamedTemporaryFile(suffix=".html", delete=False)
            tmp   = tmp_f.name
            tmp_f.close()
            self.html_path = build_html(self.delta_df, csv_names, tmp, mod_label)

            self.status_var.set(
                f"[{mod_label}] Loaded {n_rows} scan(s) from {n_subjects} subject(s) "
                f"across {len(self.csv_paths)} file(s). "
                f"HTML report ready — click 'Open HTML report'."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

    # ── Tab builders ───────────────────────────────────────────

    def _make_tree(self, parent, columns, col_labels, col_widths):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tree = ttk.Treeview(frame, columns=columns, show="headings",
                             selectmode="browse")
        for col, lbl, w in zip(columns, col_labels, col_widths):
            tree.heading(col, text=lbl)
            tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        return tree

    def _build_overview_tab(self, mod_label=""):
        frame = ttk.Frame(self.nb)
        tab_text = f"Overview ({mod_label})" if mod_label else "Overview"
        self.nb.add(frame, text=tab_text)

        cols    = ["subject", "n_pre", "n_post"] + \
                  [col for col, _ in METRICS]
        labels  = ["Subject", "# Pre", "# Post"] + \
                  [lbl for _, lbl in METRICS]
        widths  = [100, 55, 55] + [120] * len(METRICS)

        tree = self._make_tree(frame, cols, labels, widths)

        dp_map = {"Posterior_Globe_Area_mm2": 2, "Radius_of_Curvature_mm": 3}

        for _, r in self.delta_df.iterrows():
            vals = [r["subject"], int(r["n_pre"]), int(r["n_post"])]
            for col, _ in METRICS:
                dp = dp_map.get(col, 4)
                d = r.get(f"{col}_delta", np.nan)
                if np.isnan(d):
                    vals.append("N/A")
                else:
                    sign = "+" if d >= 0 else ""
                    vals.append(f"{sign}{d:.{dp}f}")
            tree.insert("", "end", values=vals)

    def _build_metric_tab(self, col, label):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=label.split("(")[0].strip())

        dp = 2 if col in ("Posterior_Globe_Area_mm2",) else 4

        cols   = ["subject", "pre", "post", "delta"]
        labels = ["Subject", "Pre-flight", "Post-flight", "Delta (Δ)"]
        widths = [120, 150, 150, 150]

        tree = self._make_tree(frame, cols, labels, widths)

        # Tag colours
        tree.tag_configure("pos",  foreground="#2e7d32")
        tree.tag_configure("neg",  foreground="#c62828")
        tree.tag_configure("na",   foreground="#888888")

        for _, r in self.delta_df.iterrows():
            pre   = r.get(f"{col}_pre",   np.nan)
            post  = r.get(f"{col}_post",  np.nan)
            delta = r.get(f"{col}_delta", np.nan)

            pre_s  = _fmt(pre,  dp)
            post_s = _fmt(post, dp)

            if np.isnan(delta):
                delta_s = "N/A"
                tag = "na"
            else:
                sign    = "+" if delta >= 0 else ""
                delta_s = f"{sign}{delta:.{dp}f}"
                tag     = "pos" if delta >= 0 else "neg"

            tree.insert("", "end",
                        values=[r["subject"], pre_s, post_s, delta_s],
                        tags=(tag,))

    # ── Export ─────────────────────────────────────────────────

    def _open_html(self):
        if not self.html_path:
            messagebox.showinfo("No report", "Load a CSV first.")
            return
        _open_in_browser(self.html_path)

    def _save_csv(self):
        if self.delta_df is None:
            messagebox.showinfo("No data", "Load a CSV first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="flight_deltas.csv"
        )
        if not path:
            return

        rows = []
        for _, r in self.delta_df.iterrows():
            row = {"Subject": r["subject"],
                   "N_Pre_Images": int(r["n_pre"]),
                   "N_Post_Images": int(r["n_post"])}
            for col, label in METRICS:
                base = label.replace(" (mm)", "").replace(" (mm²)", "").replace(" ", "_")
                row[f"{base}_Pre"]   = r.get(f"{col}_pre",   np.nan)
                row[f"{base}_Post"]  = r.get(f"{col}_post",  np.nan)
                row[f"{base}_Delta"] = r.get(f"{col}_delta", np.nan)
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
        messagebox.showinfo("Saved", f"Summary CSV saved to:\n{path}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # CLI mode: python3 deltas.py path/to/detection_results.csv [output.html]
    if len(sys.argv) >= 2:
        csv_path  = sys.argv[1]
        out_html  = sys.argv[2] if len(sys.argv) >= 3 else \
                    os.path.splitext(csv_path)[0] + "_deltas.html"
        out_csv   = os.path.splitext(out_html)[0] + ".csv"

        df       = load_and_annotate(csv_path)
        delta_df = compute_deltas(df)

        build_html(delta_df, [os.path.basename(csv_path)], out_html)

        rows = []
        for _, r in delta_df.iterrows():
            row = {"Subject": r["subject"]}
            for col, _ in METRICS:
                row[f"{col}_Pre"]   = r.get(f"{col}_pre",   np.nan)
                row[f"{col}_Post"]  = r.get(f"{col}_post",  np.nan)
                row[f"{col}_Delta"] = r.get(f"{col}_delta", np.nan)
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.4f")

        print(f"HTML report : {out_html}")
        print(f"Summary CSV : {out_csv}")
        print()
        print(delta_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    else:
        root = tk.Tk()
        app  = DeltaApp(root)
        root.mainloop()
