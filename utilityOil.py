import pandas as pd
import numpy as np
import re
import math
import streamlit as st


# Define the function to extract and clean
def extract_from_dilution(text):
    ratio = None
    oil_pct = None
    tube_volume = None
    start_time = None
    cleaned_dilution = text
    if pd.notna(text):
        original_text = text = str(text)
        # Extract ratio
        # match_ratio = re.search(r"\((\d:\d)\)\s*ratio", text)
        # if match_ratio:
            # ratio = match_ratio.group(1)
        # Extract oil
        match_oil = re.search(r"(\d+)%\s*oil", text, re.IGNORECASE)
        if match_oil:
            oil_pct = f"{match_oil.group(1)}%"
        # Extract mL
        match_ml = re.search(r"(\d+)\s*mL", text, re.IGNORECASE)
        if match_ml:
            tube_volume = match_ml.group(1)
        # Extract start time
        match_time = re.search(r"\b(\d{1,2}:\d{2})\b", text)
        if match_time:
            start_time = match_time.group(1)
        # Now remove all matched patterns from text
        cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}:\d{2}", "", cleaned_dilution)  # Remove time
        cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}-[A-Za-z]{3}", "", cleaned_dilution)  # Remove date
        #cleaned_dilution = re.sub(r"\s*\(?\d:\d\)?\s*ratio", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove (X:Y) ratio
        cleaned_dilution = re.sub(r"\s*\d+% oil", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove oil %
        cleaned_dilution = re.sub(r"\s*\d+\s*mL", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove mL
        # Remove leftover " - " at ends or doubled spaces
        cleaned_dilution = re.sub(r"\s*-\s*", " - ", cleaned_dilution).strip(" -")
        cleaned_dilution = re.sub(r"\s{2,}", " ", cleaned_dilution).strip()
    return pd.Series([ratio, oil_pct, tube_volume, start_time, cleaned_dilution])

# Parse Extracted Formulation
def parse_formulation(text):
    data = {
        "SampleID": re.search(r"\((.*?)\)", text).group(1).strip() if re.search(r"\((.*?)\)", text) else None
    }
    seen_keys_lower = set()
    parts = re.split(r'[,-]', text)
    for p in parts:
        p = p.strip()
        # Handle "12% Component"
        percent_match = re.match(r"(\d+\.?\d*)%\s*(.*)", p)
        if percent_match:
            val, chem = percent_match.groups()
            chem = re.sub(r"\(.*?\)", "", chem).strip()
            chem_key = f"{chem} (%)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())
            continue

        # Handle "Component-12%" (new format)
        reverse_match = re.match(r"(.*?)-(\d+\.?\d*)%", p)
        if reverse_match:
            chem, val = reverse_match.groups()
            chem = re.sub(r"\(.*?\)", "", chem).strip()
            chem_key = f"{chem} (%)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())

    return data

def extract_all_samples_with_corrected_stability(df):
    samples = []
    row = 0
    last_formulation = {}
    last_sample_id = None
    is_stable = np.nan
    last_dilution_text = None
    last_date = None
    while row < df.shape[0]:
        cell = str(df.iat[row, 0]).strip()
        # Detect formulation row
        if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
            last_formulation = parse_formulation(cell)
            last_sample_id = last_formulation.get("SampleID")
            # Check up to 10 columns from column 1 to 10 (inclusive)
            is_stable = np.nan
            for col_offset in range(1, min(11, df.shape[1])):  # 1 to 10 inclusive
                try:
                    col_val = str(df.iat[row, col_offset]).lower().strip()
                    if "unstable" in col_val:
                        is_stable = False
                        break
                    elif "stable" in col_val and "unstable" not in col_val:
                        is_stable = True
                except:
                    continue
            row += 1
            next_cell = str(df.iat[row, 0]).lower() if row < df.shape[0] else ""
            if "dilution" not in next_cell:
                samples.append({
                    "SampleID": last_sample_id,
                    "Is_stable": is_stable,
                    "Dilution": np.nan,
                    **{k: v for k, v in last_formulation.items() if k != "SampleID"}
                })
            continue
        # Detect dilution row
        if "dilution" in cell.lower():
            dilution_row = df.iloc[row].fillna("").astype(str).tolist()
            last_dilution_text = " - ".join([x.strip() for x in dilution_row if x.strip()])
            last_date = df.iat[row, 2] if df.shape[1] > 2 else None
            row += 1
            if row >= df.shape[0]:
                break
            header_row = df.iloc[row].fillna("").astype(str).tolist()
            header_map = {name.lower().strip(): idx for idx, name in enumerate(header_row)}
            row += 1
            while row < df.shape[0]:
                time_val = str(df.iat[row, 0]).strip()
                if time_val.lower().startswith("time") or time_val == "":
                    row += 1
                    continue
                if any(sym in time_val.lower() for sym in ["%", "ppm"]) and "(" in time_val:
                    break
                if "dilution" in time_val.lower():
                    break
                if not re.match(r"^\d+(\.\d+)?[hdm]?$", time_val, re.IGNORECASE):
                    row += 1
                    continue
                sample = {
                    "SampleID": last_sample_id,
                    "Dilution": last_dilution_text,
                    "Time (min)": time_val,
                    "Date": last_date,
                    "Baseline": df.iat[row, header_map.get("baseline", -1)] if "baseline" in header_map else None,
                    "Foam Layer (cc)": df.iat[row, header_map.get("foam layer (cc)", -1)] if "foam layer (cc)" in header_map else None,
                    "Foam Texture": df.iat[row, header_map.get("foam texture", -1)] if "foam texture" in header_map else None,
                    "Is_stable": is_stable,
                }
                sample.update({k: v for k, v in last_formulation.items() if k != "SampleID"})
                samples.append(sample)
                row += 1
            continue
        else:
            row += 1
    return pd.DataFrame(samples)

def convert_to_minutes(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    
    if re.fullmatch(r"\d+(\.\d+)?", value):  # already in minutes
        return float(value)
    elif value.endswith("h"):
        return float(value[:-1]) * 60
    elif value.endswith("d"):
        return float(value[:-1]) * 24 * 60
    else:
        return None  # or return 0 or keep original

def sort_time_columns_in_df(df):
    time_cols = [col for col in df.columns if re.search(r"Time \([\d.]+\)", col)]
    other_cols = [col for col in df.columns if col not in time_cols]

    def extract_number(col):
        match = re.search(r"\(([\d.]+)\)", col)
        return float(match.group(1)) if match else float('inf')

    sorted_time_cols = sorted(time_cols, key=extract_number)
    
    # Reorder the DataFrame
    df = df[other_cols + sorted_time_cols]
    return df


def convert_time_columns_to_float_hour(df):
    new_columns = {}

    for col in df.columns:
        match = re.search(r"Time \(([\d.]+)\) - (.+)", col)
        if match:
            minutes = float(match.group(1))
            if minutes >= 1000: 
                hours = round(minutes / 60, 2) 
                hour_label = f"{hours}h"
                new_col = f"Time ({hour_label}) - {match.group(2)}"
                new_columns[col] = new_col

    df = df.rename(columns=new_columns)
    return df

def extract_ratio_from_dilution(df):
    if 'Ratio' not in df.columns:
        df['Ratio'] = ""
    else:
        df['Ratio'] = df['Ratio'].astype(str)
    # Function to extract ratio and clean dilution
    def process_dilution_text(text):
        if pd.isna(text):
            return "", text
        match = re.search(r"\((\d+:\d+)\)\s*ratio", text)
        ratio = match.group(1) if match else np.nan
        new_text = re.sub(r"\s*\(\d+:\d+\)\s*ratio", "", text).strip()
        new_text = new_text.replace("-", "").strip()  # Remove hyphens
        return ratio, new_text
    # Apply the function to each row
    df[['ExtractedRatio', 'CleanDilution']] = df['Dilution'].apply(
        lambda x: pd.Series(process_dilution_text(x))
    )
    # Update the Ratio and Dilution columns
    df['Ratio'] = df['ExtractedRatio'].where(df['ExtractedRatio'] != "", df['Ratio'])
    df['Dilution'] = df['CleanDilution']
    # Drop temporary columns
    df.drop(columns=['ExtractedRatio', 'CleanDilution'], inplace=True)
    return df

def check_password():
    def password_entered():
        if st.session_state["password"] == "2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Optional: clear password after check
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Ask for password
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("Incorrect password")
        return False
    else:
        return True
