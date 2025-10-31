from typing import NamedTuple, Optional
import argparse
import os
import pandas as pd
import numpy as np
import re
import sys
import string
import unicodedata
import difflib
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm 
    tqdm.pandas()
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  Install tqdm for progress bars: pip install tqdm")

# CONFIGURATION
class ParsedArgs(NamedTuple):
    input_path: str
    output_path: str
    legend_path: Optional[str]

def _parse_args() -> ParsedArgs:
    p = argparse.ArgumentParser(
        description="Advanced cleaning + feature engineering (CLI). "
                    "Usage: Data_Cleaning_CLI.py <input> [output] [legend] "
                    "or use flags: -i/--input, -o/--output, -l/--legend"
    )

    # Positional args
    p.add_argument("input_pos", nargs="?", help="Input file (.csv, .gz, .parquet, .xlsx, .xls)")
    p.add_argument("output_pos", nargs="?", default=None, help="Output file (csv/parquet)")
    p.add_argument("legend_pos", nargs="?", default=None, help="Legend CSV path")

    # Optional flags
    p.add_argument("-i", "--input", dest="input_opt", help="Input file path")
    p.add_argument("-o", "--output", dest="output_opt", help="Output file path (csv/parquet)")
    p.add_argument("-l", "--legend", dest="legend_opt", help="Legend CSV path")

    args = p.parse_args()

    input_path = args.input_opt or args.input_pos
    if not input_path:
        p.error("No input file provided. Provide <input> positionally or via -i/--input.")

    output_path = args.output_opt or args.output_pos or "cleaned_output.csv"
    legend_path = args.legend_opt or args.legend_pos or (os.path.splitext(output_path)[0] + "_legend.csv")

    return ParsedArgs(input_path=input_path, output_path=output_path, legend_path=legend_path)

# File I/O Helpers
def read_input_file(path: str) -> pd.DataFrame:
    # Read CSV, GZ, Parquet, XLSX, or XLS into a DataFrame
    ext = os.path.splitext(path)[1].lower()
    print(f"[Cleaner] Detected file extension: {ext}")

    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext == ".gz":
        return pd.read_csv(path, compression="gzip", low_memory=False)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_output_file(df: pd.DataFrame, path: str):
    # Save DataFrame to CSV or Parquet depending on extension
    ext = os.path.splitext(path)[1].lower()
    dirpath = os.path.dirname(path)
    if dirpath:  # Only create dirs if path includes a directory
        os.makedirs(dirpath, exist_ok=True)

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"[Cleaner] Saved cleaned file to: {path}")

# Performance settings
USE_SPACY = False 
USE_PARALLEL = True 
N_CORES = max(1, cpu_count() - 1)

# Columns to drop
DROP_COLS = ["PATIENT_NAME", "CASE_NUMBER", "BOOKING_DATE", "PATIENT_CODE_OLD"]

PLANNED_COLS = [
    "PLANNED_PATIENT_CALL_TIME", "PLANNED_PATIENT_FETCH_TIME",
    "PLANNED_RECEPTION_IN_TIME", "PLANNED_ENTER_OR_TIME",
    "PLANNED_ANAESTHESIA_INDUCTION", "PLANNED_SURGERY_PREP_TIME",
    "PLANNED_KNIFE_TO_SKIN_TIME", "PLANNED_SKIN_CLOSURE",
    "PLANNED_PATIENT_REVERSAL_TIME", "PLANNED_EXIT_OR_TIME",
    "PLANNED_OR_CLEANUP_TIME", "PLANNED_EXIT_RECOVERY_TIME",
]

ACTUAL_COLS = [
    "PATIENT_CALL_TIME", "PATIENT_FETCH_TIME",
    "ACTUAL_RECEPTION_IN_TIME", "ACTUAL_ENTER_OR_TIME",
    "ACTUAL_ANAESTHESIA_INDUCTION", "ACTUAL_SURGERY_PREP_TIME",
    "ACTUAL_KNIFE_TO_SKIN_TIME", "ACTUAL_SKIN_CLOSURE",
    "ACTUAL_PATIENT_REVERSAL_TIME", "ACTUAL_EXIT_OR_TIME",
    "ACTUAL_OR_CLEANUP_TIME", "ACTUAL_EXIT_RECOVERY_TIME",
]

# Optional: spaCy for advanced text normalization
if USE_SPACY:
    try:
        import spacy # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except Exception:
            nlp = spacy.blank("en")
    except Exception:
        nlp = None
        USE_SPACY = False
        print("⚠️  spaCy not available, using basic normalization")
else:
    nlp = None

# HELPER FUNCTIONS: TEXT NORMALIZATION (VECTORIZED)
_punct_tbl = str.maketrans("", "", string.punctuation)

def normalize_text_series(s: pd.Series) -> pd.Series:
    # text normalization
    s = s.astype(str).str.normalize('NFKC')
    s = s.str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"\s*/\s*", " / ", regex=True)
    s = s.str.replace(r"\s*,\s*", ", ", regex=True)
    s = s.str.strip(" ,")
    return s

def remove_trailing_code_in_parens(name_s: pd.Series, code_s: pd.Series) -> pd.Series:
    # removal of trailing codes in parentheses
    code_up = code_s.astype(str).str.strip().str.upper()
    
    # Vectorized regex replacement
    def remove_code(row):
        name, code = row
        if pd.isna(code) or code == "":
            return name
        pattern = rf"\(\s*{re.escape(code)}\s*\)\s*$"
        return re.sub(pattern, "", str(name), flags=re.IGNORECASE).strip(" ,")
    
    result = pd.DataFrame({'name': name_s, 'code': code_up}).apply(remove_code, axis=1)
    return result

def choose_canonical_name(name_series: pd.Series) -> str:
    # Choose the most frequent and longest name from a series
    s = name_series.dropna().astype(str)
    s = s[s.str.strip().ne("").values]
    s = s[s.str.strip().ne("unknown").values]
    if s.empty:
        return "0"
    vc = s.value_counts()
    top_freq = vc.iloc[0]
    candidates = vc[vc.eq(top_freq)].index.tolist()
    return max(candidates, key=len)

def build_operation_legend(df, code_col="SURGICAL_CODE", nature_col="NATURE", 
                           unknown_tokens=("0", "na", "n/a", "-", "null", "nan")):
    # Build operation legend
    work = pd.DataFrame({
        "operation_code": df[code_col].astype(str).str.strip().str.upper(),
        "operation_name_raw": df[nature_col]
    })
    
    name_norm = normalize_text_series(work["operation_name_raw"])
    name_norm = name_norm.replace(list(unknown_tokens), "unknown")
    name_clean = remove_trailing_code_in_parens(name_norm, work["operation_code"])
    
    tmp = pd.DataFrame({
        "operation_code": work["operation_code"], 
        "operation_name": name_clean
    })
    tmp = tmp[tmp["operation_code"].str.len() > 0]
    
    legend = (
        tmp.groupby("operation_code", as_index=False)["operation_name"]
        .apply(choose_canonical_name)
        .rename(columns={"operation_name": "operation_name"})
    )
    
    df_out = df.drop(columns=[nature_col], errors="ignore")
    return df_out, legend

def clean_equipment_vectorized(df, col="EQUIPMENT", sep=";",
                               tags_to_strip=(r"#nuh",),
                               unknown_vals=("0", "na", "n/a", "-", "null", "nan", "")):
    # equipment cleaning
    pattern = r"|".join(fr"{re.escape(tag)}[_-]?" for tag in tags_to_strip)
    
    s = df[col].astype(str)
    s = s.str.replace(pattern, "", regex=True, case=False)
    s = s.str.lower().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"[-_/]", " ", regex=True)
    s = s.replace(list(unknown_vals), "unknown")
    
    # Sort items within each cell
    def sort_items(cell):
        if cell == "unknown":
            return "unknown"
        items = [x.strip() for x in cell.split(sep) if x.strip() and x.strip() != "unknown"]
        if not items:
            return "unknown"
        return f"{sep} ".join(sorted(set(items)))
    
    df[col] = s.apply(sort_items)
    return df

def clean_implant_vectorized(df, col="IMPLANT", unknown_vals=("0", "na", "n/a", "-", "null", "nan", "", "nil", "nil.")):
    # implant cleaning
    s = df[col].astype(str).str.lower().str.strip()
    s = s.replace(list(unknown_vals), "0")
    s = s.str.strip(" ;,.-")
    s = s.str.replace(r"\bx\d+\b", "", regex=True)
    s = s.str.replace(r"\byes\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    s = s.str.replace("&", "and", regex=False)
    df[col] = s
    return df

def clean_diagnosis_vectorized(df, col="DIAGNOSIS", unknown_vals=("0", "na", "n/a", "-", "null", "nan", "", "nil")):
    # diagnosis cleaning
    s = df[col].astype(str).str.lower().str.strip()
    s = s.replace(list(unknown_vals), "not recorded")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    df[col] = s
    return df

# HELPER FUNCTIONS: DATE/TIME IMPUTATION

def enforce_ordering_vectorized(data, col_pairs):
    # enforcement of time ordering
    for before, after in col_pairs:
        if before in data.columns and after in data.columns:
            mask = (data[before].notna()) & (data[after].notna()) & (data[after] < data[before])
            data.loc[mask, after] = data.loc[mask, before]
    return data

def impute_planned_from_knife_vectorized(data):
    # mputation of planned times from knife-to-skin.
    knife = data["PLANNED_KNIFE_TO_SKIN_TIME"]
    closure = data["PLANNED_SKIN_CLOSURE"]
    
    # Fill induction
    if "PLANNED_ANAESTHESIA_INDUCTION" in data.columns:
        mask = data["PLANNED_ANAESTHESIA_INDUCTION"].isna() & knife.notna()
        data.loc[mask, "PLANNED_ANAESTHESIA_INDUCTION"] = knife[mask]
    
    # Fill prep
    if "PLANNED_SURGERY_PREP_TIME" in data.columns:
        mask = data["PLANNED_SURGERY_PREP_TIME"].isna() & knife.notna()
        data.loc[mask, "PLANNED_SURGERY_PREP_TIME"] = knife[mask]
    
    # Fill reversal
    if "PLANNED_PATIENT_REVERSAL_TIME" in data.columns:
        mask = data["PLANNED_PATIENT_REVERSAL_TIME"].isna() & closure.notna()
        data.loc[mask, "PLANNED_PATIENT_REVERSAL_TIME"] = closure[mask]
    
    return data

def impute_patient_fetch_time_vectorized(data):
    # imputation of patient fetch time as midpoint
    call = pd.to_datetime(data.get("PATIENT_CALL_TIME", pd.Series(dtype='datetime64[ns]')), errors='coerce')
    reception = pd.to_datetime(data.get("ACTUAL_RECEPTION_IN_TIME", pd.Series(dtype='datetime64[ns]')), errors='coerce')
    fetch = pd.to_datetime(data.get("PATIENT_FETCH_TIME", pd.Series(dtype='datetime64[ns]')), errors='coerce')
    
    # Case 1: Both call and reception exist
    mask1 = fetch.isna() & call.notna() & reception.notna()
    data.loc[mask1, "PATIENT_FETCH_TIME"] = (call[mask1] + (reception[mask1] - call[mask1]) / 2).dt.floor("min")
    
    # Case 2: Only call exists
    mask2 = fetch.isna() & call.notna() & reception.isna()
    data.loc[mask2, "PATIENT_FETCH_TIME"] = call[mask2].dt.floor("min")
    
    # Case 3: Only reception exists
    mask3 = fetch.isna() & call.isna() & reception.notna()
    data.loc[mask3, "PATIENT_FETCH_TIME"] = reception[mask3].dt.floor("min")
    
    return data

def compute_statistical_marks(data):
    # Compute average 'marks' for imputing missing times
    marks = {}
    
    # Case A: induction & prep relative to enter/knife
    mask = (
        data["ACTUAL_ENTER_OR_TIME"].notna()
        & data["ACTUAL_ANAESTHESIA_INDUCTION"].notna()
        & data["ACTUAL_SURGERY_PREP_TIME"].notna()
        & data["ACTUAL_KNIFE_TO_SKIN_TIME"].notna()
    )
    clean = data.loc[mask].copy()
    clean = clean[
        (clean["ACTUAL_ENTER_OR_TIME"] <= clean["ACTUAL_ANAESTHESIA_INDUCTION"])
        & (clean["ACTUAL_ANAESTHESIA_INDUCTION"] <= clean["ACTUAL_SURGERY_PREP_TIME"])
        & (clean["ACTUAL_SURGERY_PREP_TIME"] <= clean["ACTUAL_KNIFE_TO_SKIN_TIME"])
    ]
    if not clean.empty:
        total = (clean["ACTUAL_KNIFE_TO_SKIN_TIME"] - clean["ACTUAL_ENTER_OR_TIME"]).dt.total_seconds()
        marks["induction"] = ((clean["ACTUAL_ANAESTHESIA_INDUCTION"] - clean["ACTUAL_ENTER_OR_TIME"]).dt.total_seconds() / total).mean(skipna=True)
        marks["prep"] = ((clean["ACTUAL_SURGERY_PREP_TIME"] - clean["ACTUAL_ENTER_OR_TIME"]).dt.total_seconds() / total).mean(skipna=True)
    
    # Case B: prep relative to induction/knife
    mask = (
        data["ACTUAL_ANAESTHESIA_INDUCTION"].notna()
        & data["ACTUAL_SURGERY_PREP_TIME"].notna()
        & data["ACTUAL_KNIFE_TO_SKIN_TIME"].notna()
    )
    clean = data.loc[mask].copy()
    clean = clean[
        (clean["ACTUAL_ANAESTHESIA_INDUCTION"] <= clean["ACTUAL_SURGERY_PREP_TIME"])
        & (clean["ACTUAL_SURGERY_PREP_TIME"] <= clean["ACTUAL_KNIFE_TO_SKIN_TIME"])
    ]
    if not clean.empty:
        total = (clean["ACTUAL_KNIFE_TO_SKIN_TIME"] - clean["ACTUAL_ANAESTHESIA_INDUCTION"]).dt.total_seconds()
        marks["prep_from_induction"] = ((clean["ACTUAL_SURGERY_PREP_TIME"] - clean["ACTUAL_ANAESTHESIA_INDUCTION"]).dt.total_seconds() / total).mean(skipna=True)
    
    # Case C: induction relative to enter/prep
    mask = (
        data["ACTUAL_ENTER_OR_TIME"].notna()
        & data["ACTUAL_ANAESTHESIA_INDUCTION"].notna()
        & data["ACTUAL_SURGERY_PREP_TIME"].notna()
    )
    clean = data.loc[mask].copy()
    clean = clean[
        (clean["ACTUAL_ENTER_OR_TIME"] <= clean["ACTUAL_ANAESTHESIA_INDUCTION"])
        & (clean["ACTUAL_ANAESTHESIA_INDUCTION"] <= clean["ACTUAL_SURGERY_PREP_TIME"])
    ]
    if not clean.empty:
        total = (clean["ACTUAL_SURGERY_PREP_TIME"] - clean["ACTUAL_ENTER_OR_TIME"]).dt.total_seconds()
        marks["induction_from_enter"] = ((clean["ACTUAL_ANAESTHESIA_INDUCTION"] - clean["ACTUAL_ENTER_OR_TIME"]).dt.total_seconds() / total).mean(skipna=True)
    
    # Case D: reversal relative to closure/exit
    mask = (
        data["ACTUAL_SKIN_CLOSURE"].notna()
        & data["ACTUAL_PATIENT_REVERSAL_TIME"].notna()
        & data["ACTUAL_EXIT_OR_TIME"].notna()
    )
    clean = data.loc[mask].copy()
    clean = clean[
        (clean["ACTUAL_SKIN_CLOSURE"] <= clean["ACTUAL_PATIENT_REVERSAL_TIME"])
        & (clean["ACTUAL_PATIENT_REVERSAL_TIME"] <= clean["ACTUAL_EXIT_OR_TIME"])
    ]
    if not clean.empty:
        total = (clean["ACTUAL_EXIT_OR_TIME"] - clean["ACTUAL_SKIN_CLOSURE"]).dt.total_seconds()
        marks["reversal"] = ((clean["ACTUAL_PATIENT_REVERSAL_TIME"] - clean["ACTUAL_SKIN_CLOSURE"]).dt.total_seconds() / total).mean(skipna=True)
    
    # Case E: cleanup offset from exit
    mask = (
        data["ACTUAL_EXIT_OR_TIME"].notna()
        & data["ACTUAL_OR_CLEANUP_TIME"].notna()
    )
    clean = data.loc[mask].copy()
    valid = (clean["ACTUAL_OR_CLEANUP_TIME"] >= clean["ACTUAL_EXIT_OR_TIME"]) & (
        (clean["ACTUAL_OR_CLEANUP_TIME"] - clean["ACTUAL_EXIT_OR_TIME"]) <= pd.Timedelta(hours=12)
    )
    clean = clean[valid]
    if not clean.empty:
        diffs = (clean["ACTUAL_OR_CLEANUP_TIME"] - clean["ACTUAL_EXIT_OR_TIME"]).dt.total_seconds()
        marks["cleanup_offset"] = round(diffs.mean(skipna=True) / 60.0)
    
    return marks

def impute_induction_prep_reversal_cleanup_vectorized(data, marks):
    # imputation using statistical marks
    enter = data["ACTUAL_ENTER_OR_TIME"]
    induction = data["ACTUAL_ANAESTHESIA_INDUCTION"]
    prep = data["ACTUAL_SURGERY_PREP_TIME"]
    knife = data["ACTUAL_KNIFE_TO_SKIN_TIME"]
    closure = data["ACTUAL_SKIN_CLOSURE"]
    reversal = data["ACTUAL_PATIENT_REVERSAL_TIME"]
    exit_ = data["ACTUAL_EXIT_OR_TIME"]
    cleanup = data["ACTUAL_OR_CLEANUP_TIME"]
    
    # Case A: both missing induction & prep
    if "induction" in marks and "prep" in marks:
        mask = enter.notna() & induction.isna() & prep.isna() & knife.notna()
        total = knife[mask] - enter[mask]
        data.loc[mask, "ACTUAL_ANAESTHESIA_INDUCTION"] = (enter[mask] + total * marks["induction"]).dt.round("min")
        data.loc[mask, "ACTUAL_SURGERY_PREP_TIME"] = (enter[mask] + total * marks["prep"]).dt.round("min")
    
    # Case B: missing prep only
    if "prep_from_induction" in marks:
        mask = induction.notna() & prep.isna() & knife.notna()
        total = knife[mask] - induction[mask]
        data.loc[mask, "ACTUAL_SURGERY_PREP_TIME"] = (induction[mask] + total * marks["prep_from_induction"]).dt.round("min")
    
    # Case C: missing induction only
    if "induction_from_enter" in marks:
        mask = enter.notna() & induction.isna() & prep.notna()
        total = prep[mask] - enter[mask]
        data.loc[mask, "ACTUAL_ANAESTHESIA_INDUCTION"] = (enter[mask] + total * marks["induction_from_enter"]).dt.round("min")
    
    # Case D: missing reversal
    if "reversal" in marks:
        mask = closure.notna() & reversal.isna() & exit_.notna()
        total = exit_[mask] - closure[mask]
        data.loc[mask, "ACTUAL_PATIENT_REVERSAL_TIME"] = (closure[mask] + total * marks["reversal"]).dt.round("min")
    
    # Case E: missing cleanup
    if "cleanup_offset" in marks:
        mask = exit_.notna() & cleanup.isna()
        data.loc[mask, "ACTUAL_OR_CLEANUP_TIME"] = (exit_[mask] + pd.Timedelta(minutes=marks["cleanup_offset"])).dt.round("min")
    
    return data

# ============================================================================
# HELPER FUNCTIONS: VALIDATION (VECTORIZED)
# ============================================================================

def validate_datetime_order_vectorized(data, cols):
    # validation of datetime ordering. Returns boolean mask of valid rows.
    times = data[cols]
    valid = pd.Series(True, index=data.index)
    
    # Check systematic ordering
    for i in range(len(cols) - 4):
        mask = times.iloc[:, i].notna() & times.iloc[:, i + 1].notna()
        invalid = mask & (times.iloc[:, i] > times.iloc[:, i + 1])
        valid &= ~invalid
    
    # Check anchor relationships
    anchor_idx = -3
    mask = times.iloc[:, anchor_idx].notna() & times.iloc[:, -2].notna()
    invalid = mask & (times.iloc[:, anchor_idx] > times.iloc[:, -2])
    valid &= ~invalid
    
    mask = times.iloc[:, anchor_idx].notna() & times.iloc[:, -1].notna()
    invalid = mask & (times.iloc[:, anchor_idx] > times.iloc[:, -1])
    valid &= ~invalid
    
    return valid

def validate_duration_vectorized(data, cols, max_hours=72):
    # validation of duration limits. Returns boolean mask of valid rows.
    times = data[cols]
    valid = pd.Series(True, index=data.index)
    max_td = pd.Timedelta(hours=max_hours)
    
    # Check systematic durations
    for i in range(len(cols) - 4):
        mask = times.iloc[:, i].notna() & times.iloc[:, i + 1].notna()
        invalid = mask & ((times.iloc[:, i + 1] - times.iloc[:, i]) > max_td)
        valid &= ~invalid
    
    # Check anchor relationships
    anchor_idx = -3
    mask = times.iloc[:, anchor_idx].notna() & times.iloc[:, -2].notna()
    invalid = mask & ((times.iloc[:, -2] - times.iloc[:, anchor_idx]) > max_td)
    valid &= ~invalid
    
    mask = times.iloc[:, anchor_idx].notna() & times.iloc[:, -1].notna()
    invalid = mask & ((times.iloc[:, -1] - times.iloc[:, anchor_idx]) > max_td)
    valid &= ~invalid
    
    return valid

# HELPER FUNCTIONS: DELAY REASON TAXONOMY

def normalize_delay_text_basic(s: str) -> str:
    # Basic text normalization with abbreviation expansion.
    s = str(s).lower()
    s = s.translate(_punct_tbl)
    s = re.sub(r"\s+", " ", s).strip()
    
    # Expand abbreviations
    abbreviations = [
        (r"\bo\.t\b", "operating theater"),
        (r"\bot\b", "operating theater"),
        (r"\bo\.r\b", "operating room"),
        (r"\banaesth\b", "anaesthesia"),
        (r"\banesth\b", "anaesthesia"),
        (r"\bpt\b", "patient"),
        (r"\bprev\b", "previous"),
        (r"\bdr\b", "doctor"),
        (r"\bpre ?med\b", "premedication"),
        (r"\baoh\b", "after office hours"),
        (r"\bem(er|urg)\w*\b", "emergency"),
    ]
    
    for pattern, replacement in abbreviations:
        s = re.sub(pattern, replacement, s)
    
    return s

# Simplified taxonomy
TAXONOMY_DATA_FAST = [
    {"category": "Priority Case/Emergency", "patterns": [
        r"emergen", r"\be\s?case\b", r"lscs", r"icu", r"\bp\s?[01]\b", r"\bec\b"
    ]},
    {"category": "Scheduling/Venue Changes", "patterns": [
        r"^\s*from\b", r"transfer", r"add.*case", r"reschedul", r"swap", r"change"
    ]},
    {"category": "Anaesthetic", "patterns": [
        r"anaesth", r"anesth", r"pacu", r"spinal", r"block"
    ]},
    {"category": "Administration", "patterns": [
        r"pre.*op", r"assessment", r"checklist", r"registration", r"marking", r"consent"
    ]},
    {"category": "Imaging/Labs", "patterns": [
        r"blood", r"lab", r"ecg", r"scan", r"radiology", r"x.*ray", r"ct\b", r"mri"
    ]},
    {"category": "Pharmacy/Medication", "patterns": [
        r"pharmacy", r"medication", r"pre.*med", r"mitomycin"
    ]},
    {"category": "Equipment", "patterns": [
        r"machine", r"equipment", r"scope", r"table", r"robot", r"pacemaker"
    ]},
    {"category": "Room/Facilities", "patterns": [
        r"^\s*o[rt]\b", r"theater", r"theatre", r"not ready", r"cleaning", r"power", r"temperature"
    ]},
    {"category": "Preparation", "patterns": [
        r"turn.*over", r"wash", r"prepar", r"warming", r"bed"
    ]},
    {"category": "Transport/Portering", "patterns": [
        r"porter", r"transport", r"fetch"
    ]},
    {"category": "Patient Factors", "patterns": [
        r"patient", r"\bpt\b", r"fasting", r"parent", r"baby", r"bed.*not", r"interpreter", r"dilat"
    ]},
    {"category": "Other Case Issues", "patterns": [
        r"\d(?:st|nd|rd|th)\s*case", r"previous", r"am.*list", r"pm.*list", r"case.*late", r"finish.*late"
    ]},
    {"category": "Timing", "patterns": [
        r"schedule.*at", r"case.*at.*\d{3,4}", r"op.*at.*\d{3,4}", r"list.*at"
    ]},
    {"category": "Surgeon/Staff", "patterns": [
        r"surgeon", r"team", r"staff", r"nurse"
    ]},
]

COMPILED_TAXONOMY_FAST = [
    (entry["category"], [re.compile(p, re.IGNORECASE) for p in entry.get("patterns", [])])
    for entry in TAXONOMY_DATA_FAST
]

def classify_late_reason_fast(text_norm: str) -> str:
    # Fast classification using simplified taxonomy
    for cat, pats in COMPILED_TAXONOMY_FAST:
        for pat in pats:
            if pat.search(text_norm or ""):
                return cat
    return "Unspecified (late)"

def process_delay_reasons_vectorized(data):
    # delay reason processing
    if "Delay_Reason" not in data.columns:
        return data
    
    # Normalize text 
    data["_Delay_norm"] = data["Delay_Reason"].astype(str).fillna("").apply(normalize_delay_text_basic)
    
    # Compute Is_Late
    planned = pd.to_datetime(data.get("PLANNED_KNIFE_TO_SKIN_TIME"), errors='coerce')
    actual = pd.to_datetime(data.get("ACTUAL_KNIFE_TO_SKIN_TIME"), errors='coerce')
    delta_min = (actual - planned).dt.total_seconds() / 60.0
    data["Is_Late"] = ((delta_min > 0) & planned.notna() & actual.notna()).astype(int)
    
    # Flag Reason_Is_Late
    not_late_phrases = ["no delay", "not delay", "not delayed", "not late", "na", "0", "null", "nan"]
    pattern = r"(?:" + "|".join(rf"(?<!\w){re.escape(p)}(?!\w)" for p in not_late_phrases) + r")"
    regex = re.compile(pattern, flags=re.IGNORECASE)
    
    clean_reason = data["Delay_Reason"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    only_punct = clean_reason.str.match(r'^(?=.*\S)(?!.*[A-Za-z]).*$', na=False)
    phrase_hit = clean_reason.str.contains(regex, na=False)
    
    data["Reason_Is_Late"] = np.where(only_punct | phrase_hit, 0, 1)
    
    # Classify delay categories
    data["Delay_Category"] = "No Delay"
    late_mask = data["Is_Late"] == 1
    
    if USE_PARALLEL and late_mask.sum() > 1000:
        # Parallel classification for large datasets
        late_texts = data.loc[late_mask, "_Delay_norm"].tolist()
        with Pool(N_CORES) as pool:
            categories = pool.map(classify_late_reason_fast, late_texts)
        data.loc[late_mask, "Delay_Category"] = categories
    else:
        # Serial classification
        if HAS_TQDM:
            tqdm.pandas(desc="Classifying delays")
            data.loc[late_mask, "Delay_Category"] = data.loc[late_mask, "_Delay_norm"].progress_apply(classify_late_reason_fast)
        else:
            data.loc[late_mask, "Delay_Category"] = data.loc[late_mask, "_Delay_norm"].apply(classify_late_reason_fast)
    
    # Drop temporary column
    data = data.drop(columns=["_Delay_norm"], errors="ignore")
    
    return data

# HELPER FUNCTIONS: DATA TYPE CONVERSION
def convert_to_correct_dtypes(data):
    # Convert all columns to their correct data types for analysis. Returns the dataframe with corrected dtypes.   
    # 1. DATETIME COLUMNS 
    datetime_cols = [
        "PLANNED_PATIENT_CALL_TIME", "PLANNED_PATIENT_FETCH_TIME",
        "PLANNED_RECEPTION_IN_TIME", "PLANNED_ENTER_OR_TIME",
        "PLANNED_ANAESTHESIA_INDUCTION", "PLANNED_SURGERY_PREP_TIME",
        "PLANNED_KNIFE_TO_SKIN_TIME", "PLANNED_SKIN_CLOSURE",
        "PLANNED_PATIENT_REVERSAL_TIME", "PLANNED_EXIT_OR_TIME",
        "PLANNED_OR_CLEANUP_TIME", "PLANNED_EXIT_RECOVERY_TIME",
        "PATIENT_CALL_TIME", "PATIENT_FETCH_TIME",
        "ACTUAL_RECEPTION_IN_TIME", "ACTUAL_ENTER_OR_TIME",
        "ACTUAL_ANAESTHESIA_INDUCTION", "ACTUAL_SURGERY_PREP_TIME",
        "ACTUAL_KNIFE_TO_SKIN_TIME", "ACTUAL_SKIN_CLOSURE",
        "ACTUAL_PATIENT_REVERSAL_TIME", "ACTUAL_EXIT_OR_TIME",
        "ACTUAL_OR_CLEANUP_TIME", "ACTUAL_EXIT_RECOVERY_TIME",
    ]
    for col in datetime_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    print(f"Converted {len([c for c in datetime_cols if c in data.columns])} datetime columns")
    
    # 2. NUMERIC COLUMNS 
    numeric_cols = [
        "ACTUAL_SURGERY_DURATION", "PLANNED_SURGERY_DURATION", "DIFF_SURGERY_DURATION",
        "ACTUAL_USAGE_DURATION", "PLANNED_USAGE_DURATION", "DIFF_USAGE_DURATION",
        "ENTER_START_DELAY", "KNIFE_START_DELAY", "EXIT_OR_DELAY"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    print(f"Converted {len([c for c in numeric_cols if c in data.columns])} numeric columns")
    
    # 3. CATEGORICAL COLUMNS
    categorical_cols = {
        "LOCATION": "category",
        "ROOM": "category",
        "CASE_STATUS": "category",
        "OPERATION_TYPE": "category",
        "EMERGENCY_PRIORITY": "category",
        "SURGICAL_CODE": "category",
        "DISCIPLINE": "category",
        "SURGEON": "category",
        "ANAESTHETIST_TEAM": "category",
        "ANESTHESIA": "category",
        "ADMISSION_STATUS": "category",
        "ADMISSION_CLASS_TYPE": "category",
        "ADMISSION_TYPE": "category",
        "ADMISSION_WARD": "category",
        "ADMISSION_BED": "category",
        "Delay_Category": "category",
    }
    for col, dtype in categorical_cols.items():
        if col in data.columns:
            data[col] = data[col].astype(dtype)
    print(f"Converted {len([c for c in categorical_cols if c in data.columns])} categorical columns")
    
    # 4. BOOLEAN COLUMNS
    boolean_cols = ["AOH", "CANCER_INDICATOR", "TRAUMA_INDICATOR"]
    for col in boolean_cols:
        if col in data.columns:
            # Convert "true"/"false" strings to boolean
            data[col] = data[col].astype(str).str.lower().map({'true': True, 'false': False})
    print(f"Converted {len([c for c in boolean_cols if c in data.columns])} boolean columns")
    
    # 5. INTEGER COLUMNS
    integer_cols = ["Is_Late", "Reason_Is_Late"]
    for col in integer_cols:
        if col in data.columns:
            data[col] = data[col].astype('int8')  # Use int8 for 0/1 flags
    print(f"Converted {len([c for c in integer_cols if c in data.columns])} integer columns")
    
    # 6. STRING COLUMNS (keep as object for flexibility)
    string_cols = [
        "OPERATION_ID", "PATIENT_CODE", "ANAESTHETIST_MCR_NO",
        "EQUIPMENT", "IMPLANT", "DIAGNOSIS", "BLOOD",
        "Delay_Reason", "Remarks"
    ]
    for col in string_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)
    print(f"Ensured {len([c for c in string_cols if c in data.columns])} string columns")
    
    return data

# MAIN PROCESSING
def main():
    args = _parse_args()
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    NATURE_LEGEND_FILE = args.legend_path
    ...

    import time
    start_time = time.time()
    
    print("=" * 80)
    print("OPTIMIZED PRODUCTION PIPELINE - File 2".center(80))
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - USE_SPACY: {USE_SPACY}")
    print(f"  - USE_PARALLEL: {USE_PARALLEL}")
    print(f"  - N_CORES: {N_CORES}")
    print("=" * 80)
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    step_start = time.time()
    data = read_input_file(INPUT_PATH)
    print(f"Loaded {len(data):,} rows from {INPUT_PATH}")
    
    # Parse datetime columns
    for col in PLANNED_COLS + ACTUAL_COLS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    print(f"Parsed datetime columns")
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DROP UNNECESSARY COLUMNS")
    print("=" * 80)
    step_start = time.time()
    
    # Drop index column if exists
    if "Unnamed: 0" in data.columns:
        data = data.drop("Unnamed: 0", axis=1)
        print("Dropped index column")
    
    # Drop unnecessary columns
    existing_drop_cols = [c for c in DROP_COLS if c in data.columns]
    if existing_drop_cols:
        data = data.drop(columns=existing_drop_cols)
        print(f"Dropped columns: {', '.join(existing_drop_cols)}")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: BUILD OPERATION LEGEND (NATURE → SURGICAL_CODE)")
    print("=" * 80)
    step_start = time.time()
    
    if "NATURE" in data.columns and "SURGICAL_CODE" in data.columns:
        data, nature_legend = build_operation_legend(data, code_col="SURGICAL_CODE", nature_col="NATURE")
        nature_legend.to_csv(NATURE_LEGEND_FILE, index=False)
        print(f"Built operation legend with {len(nature_legend):,} unique codes")
        print(f"Saved legend to {NATURE_LEGEND_FILE}")
        print(f"Dropped NATURE column")
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: CLEAN TEXT COLUMNS (VECTORIZED)")
    print("=" * 80)
    step_start = time.time()
    
    # Equipment
    if "EQUIPMENT" in data.columns:
        data = clean_equipment_vectorized(data, col="EQUIPMENT")
        print("Cleaned EQUIPMENT column")
    
    # Implant
    if "IMPLANT" in data.columns:
        data = clean_implant_vectorized(data, col="IMPLANT")
        print("Cleaned IMPLANT column")
    
    # Diagnosis
    if "DIAGNOSIS" in data.columns:
        data = clean_diagnosis_vectorized(data, col="DIAGNOSIS")
        print("Cleaned DIAGNOSIS column")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: IMPUTE MISSING TIMES (PLANNED) - VECTORIZED")
    print("=" * 80)
    step_start = time.time()
    
    # Enforce ordering constraints (vectorized)
    col_pairs = [
        ("PLANNED_KNIFE_TO_SKIN_TIME", "PLANNED_SKIN_CLOSURE"),
        ("PLANNED_SKIN_CLOSURE", "PLANNED_PATIENT_REVERSAL_TIME"),
        ("PLANNED_PATIENT_REVERSAL_TIME", "PLANNED_EXIT_OR_TIME"),
        ("PLANNED_EXIT_OR_TIME", "PLANNED_EXIT_RECOVERY_TIME"),
        ("PLANNED_EXIT_RECOVERY_TIME", "PLANNED_OR_CLEANUP_TIME"),
    ]
    data = enforce_ordering_vectorized(data, col_pairs)
    print("Enforced planned timeline ordering")
    
    # Fill missing times from knife/closure (vectorized)
    data = impute_planned_from_knife_vectorized(data)
    print("Filled missing PLANNED times from knife-to-skin/closure anchors")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: IMPUTE MISSING TIMES (ACTUAL) - VECTORIZED")
    print("=" * 80)
    step_start = time.time()
    
    # Impute patient fetch time (vectorized)
    data = impute_patient_fetch_time_vectorized(data)
    print("Imputed PATIENT_FETCH_TIME as midpoint")
    
    # Compute statistical marks
    print("Computing statistical marks for imputation...")
    marks = compute_statistical_marks(data)
    print(f"Computed {len(marks)} statistical marks:")
    for key, val in marks.items():
        if isinstance(val, float):
            print(f"  - {key}: {val:.4f}")
        else:
            print(f"  - {key}: {val}")
    
    # Impute missing times using marks (vectorized)
    data = impute_induction_prep_reversal_cleanup_vectorized(data, marks)
    print("Imputed missing ACTUAL times using statistical marks")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: VALIDATE TIMELINE INTEGRITY (VECTORIZED)")
    print("=" * 80)
    step_start = time.time()
    
    # Validate non-decreasing order (vectorized)
    planned_valid_order = validate_datetime_order_vectorized(data, PLANNED_COLS)
    actual_valid_order = validate_datetime_order_vectorized(data, ACTUAL_COLS)
    
    order_valid_mask = planned_valid_order & actual_valid_order
    num_invalid_order = (~order_valid_mask).sum()
    data = data[order_valid_mask].copy()
    print(f"Removed {num_invalid_order:,} rows with non-decreasing validation failures")
    
    # Validate duration (72 hours max) (vectorized)
    planned_valid_duration = validate_duration_vectorized(data, PLANNED_COLS, max_hours=72)
    actual_valid_duration = validate_duration_vectorized(data, ACTUAL_COLS, max_hours=72)
    
    duration_valid_mask = planned_valid_duration & actual_valid_duration
    num_invalid_duration = (~duration_valid_mask).sum()
    data = data[duration_valid_mask].copy()
    print(f"Removed {num_invalid_duration:,} rows exceeding 72-hour duration")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: REMOVE OUT-OF-SCOPE ROWS")
    print("=" * 80)
    step_start = time.time()
    
    # Remove "OUT OF OT ROOMS" (not relevant for OR analysis)
    if "LOCATION" in data.columns:
        before = len(data)
        data = data[data["LOCATION"] != "OUT OF OT ROOMS"].copy()
        after = len(data)
        print(f"Removed {before - after:,} rows with LOCATION = 'OUT OF OT ROOMS'")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: REMOVE DUPLICATES")
    print("=" * 80)
    step_start = time.time()
    before = len(data)
    data = data.drop_duplicates()
    after = len(data)
    if before != after:
        print(f"Removed {before - after} duplicate rows")
    else:
        print("No duplicate rows found")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: PROCESS DELAY REASONS (OPTIMIZED)")
    print("=" * 80)
    step_start = time.time()
    
    data = process_delay_reasons_vectorized(data)
    print("Processed delay reasons")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 11: CREATE FEATURE VARIABLES (VECTORIZED)")
    print("=" * 80)
    step_start = time.time()
    
    # Surgery duration (knife → closure)
    data["ACTUAL_SURGERY_DURATION"] = data["ACTUAL_SKIN_CLOSURE"] - data["ACTUAL_KNIFE_TO_SKIN_TIME"]
    data["PLANNED_SURGERY_DURATION"] = data["PLANNED_SKIN_CLOSURE"] - data["PLANNED_KNIFE_TO_SKIN_TIME"]
    data["DIFF_SURGERY_DURATION"] = data["ACTUAL_SURGERY_DURATION"] - data["PLANNED_SURGERY_DURATION"]
    
    # OR usage duration (enter OR → exit OR)
    data["ACTUAL_USAGE_DURATION"] = data["ACTUAL_EXIT_OR_TIME"] - data["ACTUAL_ENTER_OR_TIME"]
    data["PLANNED_USAGE_DURATION"] = data["PLANNED_EXIT_OR_TIME"] - data["PLANNED_ENTER_OR_TIME"]
    data["DIFF_USAGE_DURATION"] = data["ACTUAL_USAGE_DURATION"] - data["PLANNED_USAGE_DURATION"]
    
    # Start delays
    data["ENTER_START_DELAY"] = data["ACTUAL_ENTER_OR_TIME"] - data["PLANNED_ENTER_OR_TIME"]
    data["KNIFE_START_DELAY"] = data["ACTUAL_KNIFE_TO_SKIN_TIME"] - data["PLANNED_KNIFE_TO_SKIN_TIME"]
    data["EXIT_OR_DELAY"] = data["ACTUAL_EXIT_OR_TIME"] - data["PLANNED_EXIT_OR_TIME"]
    
    print("Created duration and delay features")
    
    # Convert timedelta features to minutes (vectorized)
    duration_cols = [
        "ACTUAL_SURGERY_DURATION", "PLANNED_SURGERY_DURATION", "DIFF_SURGERY_DURATION",
        "ACTUAL_USAGE_DURATION", "PLANNED_USAGE_DURATION", "DIFF_USAGE_DURATION",
        "ENTER_START_DELAY", "KNIFE_START_DELAY", "EXIT_OR_DELAY"
    ]
    for col in duration_cols:
        if col in data.columns:
            data[col] = data[col].dt.total_seconds() / 60
    print("Converted duration/delay features to minutes")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 12: DROP ROWS WITH MISSING CRITICAL DATA")
    print("=" * 80)
    step_start = time.time()
    
    # Define critical columns that MUST have values
    critical_cols = [
        "OPERATION_ID",
        "LOCATION", 
        "ROOM",
        "CASE_STATUS",
        "OPERATION_TYPE",
        "SURGICAL_CODE",
        "DISCIPLINE",
        "ANESTHESIA",
        "AOH",
        "BLOOD",
        "CANCER_INDICATOR",
        "TRAUMA_INDICATOR"
    ]
    
    # Only check critical columns that actually exist
    existing_critical = [c for c in critical_cols if c in data.columns]
    
    before = len(data)
    # Drop rows where ANY critical column is missing
    data = data.dropna(subset=existing_critical)
    after = len(data)
    
    print(f"Checked {len(existing_critical)} critical columns for missing values")
    print(f"Removed {before - after:,} rows with missing critical data")
    print(f"Retained {after:,} rows ({(after/before)*100:.1f}% of data)")
    
    # Show remaining missing values in non-critical columns (for information only)
    remaining_missing = data.isna().sum()
    if remaining_missing.sum() > 0:
        print(f"\nRemaining missing values in non-critical columns:")
        for col, count in remaining_missing[remaining_missing > 0].head(10).items():
            pct = (count / len(data)) * 100
            print(f"     - {col}: {count:,} ({pct:.1f}%)")
        if len(remaining_missing[remaining_missing > 0]) > 10:
            print(f"     ... and {len(remaining_missing[remaining_missing > 0]) - 10} more columns")
    
    data = data.dropna()
    print(f"Drop NA values in non critical columns")

    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 13: CONVERT TO CORRECT DATA TYPES")
    print("=" * 80)
    step_start = time.time()
    
    data = convert_to_correct_dtypes(data)
    print("All columns converted to correct data types")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 14: FINAL DATA QUALITY CHECKS")
    print("=" * 80)
    step_start = time.time()
    
    # Summary statistics
    print(f"Final dataset shape: {data.shape}")
    print(f"Total records: {len(data):,}")
    
    # Print data type summary
    print(f"\n✓ Data type distribution:")
    dtype_counts = data.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  - {dtype}: {count} columns")
    
    if "ACTUAL_KNIFE_TO_SKIN_TIME" in data.columns:
        min_date = data["ACTUAL_KNIFE_TO_SKIN_TIME"].min()
        max_date = data["ACTUAL_KNIFE_TO_SKIN_TIME"].max()
        print(f"\n✓ Date range: {min_date.date() if pd.notna(min_date) else 'N/A'} to {max_date.date() if pd.notna(max_date) else 'N/A'}")
    
    if "Is_Late" in data.columns:
        late_count = (data["Is_Late"] == 1).sum()
        late_pct = (late_count / len(data)) * 100
        print(f"✓ Late cases: {late_count:,} ({late_pct:.1f}%)")
    
    if "Delay_Category" in data.columns:
        print(f"\n✓ Delay category distribution (top 10):")
        cat_counts = data["Delay_Category"].value_counts()
        for cat, count in cat_counts.head(10).items():
            pct = (count / len(data)) * 100
            print(f"  - {cat}: {count:,} ({pct:.1f}%)")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 15: SAVE FINAL DATASET")
    print("=" * 80)
    step_start = time.time()
    save_output_file(data, OUTPUT_PATH)
    print(f"Saved {len(data):,} rows to {OUTPUT_PATH}")
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("ADVANCED CLEANING COMPLETE")
    print("=" * 80)
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Legend file: {NATURE_LEGEND_FILE}")
    print(f"Final row count: {len(data):,}")
    print(f"Final column count: {len(data.columns)}")
    print(f"\nTOTAL RUNTIME: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print("=" * 80)

if __name__ == "__main__":
    main()