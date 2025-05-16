# utility.py
import re
import numpy as np
import pandas as pd
import streamlit as st

def extract_samples_complete_fixed(df):
    samples = []
    formulations = {}
    row = 0
    last_formulation = None
    last_dilution = None
    last_tube_volume = None
    column_map = {}
    current_dilution_rows = []

    concentrate_has_RT = np.nan
    concentrate_has_8c = np.nan
    concentrate_has_4c = np.nan
    concentrate_has_HT = np.nan
    dilution_stability = np.nan

    def flush_current_dilution():
        for row_data in current_dilution_rows:
            row_data["Concentrate Stability (8C)"] = concentrate_has_8c
            row_data["Concentrate Stability (4C)"] = concentrate_has_4c
            row_data["Concentrate Stability (RT)"] = concentrate_has_RT
            row_data["Concentrate Stability (HT)"] = concentrate_has_HT
            row_data["Dilution Stability"] = dilution_stability
            row_data["Tube Volume (mL)"] = last_tube_volume
            samples.append(row_data)
        current_dilution_rows.clear()

    def parse_formulation(text):
        data = {
            "SampleID": re.search(r"\((.*?)\)", text).group(1).strip() if re.search(r"\((.*?)\)", text) else None
        }
        seen_keys_lower = set()
        for p in re.split(r'[,-]', text):
            p = p.strip()
            ppm_match = re.match(r"(\d+\.?\d*)\s*ppm\s*(.*)", p, re.IGNORECASE)
            if ppm_match:
                val, chem = ppm_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                chem_key = f"{chem} (ppm)"
                if chem_key.lower() not in seen_keys_lower:
                    data[chem_key] = float(val)
                    seen_keys_lower.add(chem_key.lower())
                continue
            percent_match = re.match(r"(\d+\.?\d*)%\s*(.*)", p, re.IGNORECASE)
            if percent_match:
                val, chem = percent_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                chem_key = f"{chem} (%)"
                if chem_key.lower() not in seen_keys_lower:
                    data[chem_key] = float(val)
                    seen_keys_lower.add(chem_key.lower())
        return data

    while row < df.shape[0]:
        cell = str(df.iat[row, 0]).strip()

        # --- Formulation row detection ---
        if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
            flush_current_dilution()
            formulation_text = cell.lower()

            # Detect stability
            s8, s4, RT, HT, ds= np.nan, np.nan, True, np.nan, np.nan
            if "unstable concentrate" in formulation_text:
                s8 = False
                s4 = False
                #RT = False
                #HT = False

            values_to_check = [
                str(df.iat[row, col]).lower().strip()
                for col in range(1, min(11, df.shape[1]))
                if pd.notna(df.iat[row, col])
            ]
            
            for val in values_to_check:
                if (("unstable" in val.lower() or "cloudy" in val.lower()) and "concentrate" in val.lower() and not re.search(r"\b\d{1,2}C\b", val, flags=re.IGNORECASE)):
                    s8 = False
                    #s4 = False
                    RT = False
                    HT = False
                if (("unstable" in val.lower() or "cloudy" in val.lower()) and "concentrate" in val.lower() and  re.search(r"8\s*[cC]", val)):
                    s8 = False
                if (("unstable" in val.lower() or "cloudy" in val.lower()) and "concentrate" in val.lower() and  re.search(r"4\s*[cC]", val)):
                    s4 = False
                if ("unstable" in val.lower() or "cloudy" in val.lower()) and "Dilution" in val.lower() :
                    ds = False
                     
                if re.search(r"\bunstable\b", val, flags=re.IGNORECASE) and re.search(r"4\s*[cC]\b", val, flags=re.IGNORECASE):
                    s4 = False
                elif re.search(r"\bstable\b", val, flags=re.IGNORECASE) and re.search(r"4\s*[cC]\b", val, flags=re.IGNORECASE):
                    if s4 is not False:
                        s4 = True
                #if re.search(r"unstable.*8\s*[cC]", val, flags=re.IGNORECASE): 
                if re.search(r"\bunstable\b", val, flags=re.IGNORECASE) and re.search(r"8\s*[cC]\b", val, flags=re.IGNORECASE):
                    s8 = False
                #elif re.search(r"stable.*8\s*[cC]", val, flags=re.IGNORECASE):
                elif re.search(r"\bstable\b", val, flags=re.IGNORECASE) and re.search(r"8\s*[cC]\b", val, flags=re.IGNORECASE):
                    if s8 is not False:
                        s8 = True
                if re.search(r"\bunstable\b", val, flags=re.IGNORECASE) and re.search(r"\bRT\b", val, flags=re.IGNORECASE):
                    RT = False
                elif re.search(r"\bstable\b", val, flags=re.IGNORECASE) and re.search(r"\bRT\b", val, flags=re.IGNORECASE):
                    if RT is not False:
                        RT = True

            concentrate_has_8c = s8
            concentrate_has_4c = s4
            dilution_stability = ds
            concentrate_has_RT = RT
            concentrate_has_HT = HT

            last_formulation = parse_formulation(cell)
            if not last_formulation.get("SampleID"):
                fallback_id = f"Sample_{len(formulations)+1}"
                last_formulation["SampleID"] = fallback_id
            formulations[last_formulation["SampleID"]] = last_formulation

            # 🔍 Check next row for dilution
            next_row_text = ",".join(df.iloc[row + 1].fillna("").astype(str)).lower() if row + 1 < df.shape[0] else ""
            if not re.search(r"\d+\s*x", next_row_text):
                # If no dilution row follows, treat it as a single-row sample
                samples.append({
                    "SampleID": last_formulation["SampleID"],
                    "Dilution Ratio": None,
                    "Day": None,
                    "Foam (cc)": None,
                    "Foam Texture": None,
                    "Water (cc)": None,
                    "Zeta": None,
                    "Conductivity": None,
                    "Size": None,
                    "PI": None,
                    "Baseline": None,
                    "Date": None,
                    "Concentrate Stability (8C)": s8,
                    "Concentrate Stability (4C)": s4,
                    "Concentrate Stability (RT)": RT,
                    "Concentrate Stability (HT)": HT,
                    "Tube Volume (mL)": None
                })

            row += 1
            continue

        if re.match(r"Day\s*\d+", cell, re.IGNORECASE):
            row_data = {"SampleID": last_formulation["SampleID"]} if last_formulation else {}
            row_data["Dilution Ratio"] = last_dilution
            row_data["Day"] = cell.strip()
            stars = ["*" for i in range(column_map.get("Foam Texture", 0) + 1, df.shape[1]) if "*" in str(df.iat[row, i])]
            row_data["Baseline"] = ", ".join(stars) if stars else None
            for offset, label in enumerate(["Date", "Foam (cc)", "Foam Texture", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]):
                col_idx = column_map.get(label)
                if label == "Date" and col_idx is None:
                    col_idx = 1
                val = str(df.iat[row, col_idx]).strip() if col_idx is not None and col_idx < df.shape[1] else None
                if not val or val.lower() == "nan":
                    row_data[label] = None
                else:
                    if label in ["Foam (cc)", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]:
                        num = re.search(r"[-+]?\d+\.?\d*", val)
                        row_data[label] = float(num.group()) if num else None
                    else:
                        row_data[label] = val

            if row + 1 < df.shape[0]:
                next_row = df.iloc[row + 1]
                non_empty = next_row.dropna()
                if len(non_empty) == 1 and column_map.get("Foam Texture") in non_empty.index:
                    extra_texture = str(non_empty.values[0]).strip()
                    if extra_texture:
                        existing = row_data.get("Foam Texture", "")
                        row_data["Foam Texture"] = f"{existing}, {extra_texture}".strip(", ")
                    row += 1

            current_dilution_rows.append(row_data)
            row += 1
            continue

        row_text_combined = ",".join(df.iloc[row].fillna("").astype(str)).lower()
        dilution_search = re.search(r"(\d+\s*X)", row_text_combined, re.IGNORECASE)
        if dilution_search:
            flush_current_dilution()
            base_dilution = dilution_search.group(1).replace(" ", "").upper()
            extra_label = []
            tube_volume = ""

            for col in range(1, 12):
                if col < df.shape[1]:
                    val = str(df.iat[row, col]).strip()
                    if val and val.lower() != "nan":
                        if "ml" in val.lower():
                            tube_volume = val
                        else:
                            extra_label.append(val)

            last_dilution = base_dilution + (" " + " ".join(extra_label) if extra_label else "")
            last_tube_volume = tube_volume

            if "foam" in row_text_combined:
                header_row = df.iloc[row]
                row += 1
            elif row + 1 < df.shape[0] and "foam" in ",".join(df.iloc[row + 1].fillna("").astype(str)).lower():
                header_row = df.iloc[row + 1]
                row += 2
            else:
                row += 1
                continue

            column_map = {}
            for i, val in header_row.items():
                val = str(val).strip().lower()
                if "foam amount" in val or ("foam" in val and "cc" in val):
                    column_map["Foam (cc)"] = i
                elif "foam texture" in val or "texture" in val:
                    column_map["Foam Texture"] = i
                elif "zeta" in val:
                    column_map["Zeta"] = i
                elif "pi" in val:
                    column_map["PI"] = i
                elif "conductivity" in val:
                    column_map["Conductivity"] = i
                elif "size" in val:
                    column_map["Size"] = i
                elif "Water (cc)" in val:
                    column_map["Water (cc)"] = i
                elif "date" in val:
                    column_map["Date"] = i
            continue

        row += 1

    flush_current_dilution()
    return samples, formulations

import numpy as np
import pandas as pd
import re

def process_dilution(dilution, date):
    date=date
    pilot = np.nan
    temp_foam = np.nan
    temp_dilu = np.nan
    ini_foam = "5cc"
    ratio = np.nan
    sonic = np.nan
    sampleDescription = np.nan
    #dilution_ratio = "Only Formulation - No dilution Data"  # Default, if no pattern is found

    if pd.isna(dilution):
        cleaned_text_1 = "Only Formulation - No Dilution Data"
        return pilot, temp_foam, ini_foam, cleaned_text_1, ratio, sonic, sampleDescription

    text = str(dilution)

    # Extract AFC
    if "AFC" in text:
        pilot = "AFC"
        # text = re.sub(r"\bAFC\b", "", text, flags=re.IGNORECASE)

    # Extract sonicated
    if re.search(r"\b(no|not)\s*\w*sonic", text, flags=re.IGNORECASE):
        sonic = False
    elif re.search(r"\bsonic\S*", text, flags=re.IGNORECASE):
        sonic = True

    # Extract ratio
    match_ratio = re.search(r"\(?(\d):(\d)\)?\s*ratio", text, flags=re.IGNORECASE)
    if match_ratio:
        ratio = f"{match_ratio.group(1)}::{match_ratio.group(2)}"
        #text = re.sub(r"\(?(\d):(\d)\)?\s*ratio", "", text, flags=re.IGNORECASE)

    # Extract and remove Xcc (e.g. 5cc)
    xcc_match = re.search(r"(\d+)\s*cc", text, flags=re.IGNORECASE)
    if xcc_match:
        ini_foam = xcc_match.group(1).strip()
        #text = re.sub(r"(\d+)\s*cc", "", text, flags=re.IGNORECASE)

    # Extract and remove Xc (e.g. 45c, not 45cc)
    xc_match = re.search(r"(\d+)\s*c(?!c)", text, flags=re.IGNORECASE)
    if xc_match:
        temp_foam = int(xc_match.group(1))
        #text = re.sub(r"(\d+)\s*c(?!c)", "", text, flags=re.IGNORECASE)

    # Extract and remove "X stream" and set as Dilution Ratio
    match_stream = re.search(r"\b(\d+)\s*stream\b", text, flags=re.IGNORECASE)
    if match_stream:
        stream_value = match_stream.group(1)
        ratio = f"{stream_value} stream"
        #text = re.sub(r"\b\d+\s*stream\b", "", text, flags=re.IGNORECASE)

    # Extract and remove "RT"
    match_RT = re.search(r"RT", text, flags=re.IGNORECASE)
    if match_RT:
        temp_foam = "RT"
        #text = re.sub(r"RT", "", text, flags=re.IGNORECASE)

    # Clean up leftover text to put in Sample Description
    cleaned_text = text.strip(" ,;-").strip()
    #if cleaned_text and cleaned_text.lower() != "nan":
        #sampleDescription = cleaned_text

    return pilot, temp_foam, ini_foam, cleaned_text, ratio, sonic, sampleDescription, date 

def clean_dilution(df):
    df = df[df["Day"].notna()].copy()

    # Convert Day column to numeric index
    df["Day_Num"] = df["Day"].str.extract(r'(\d+)').astype(int)

    max_day = df["Day_Num"].max()

    formulation_cols = [
        col for col in df.columns
        if col not in ["SampleID", "Day", "Day_Num", "Foam (cc)", "Foam Texture", "Date", "Baseline", "Pilot"]
    ]

    output_rows = []

    for (sample_id, dilution), group in df.groupby(["SampleID", "Dilution Ratio"]):
        row = {"SampleID": sample_id, "Dilution Ratio": dilution}

        # Copy formulation columns
        for col in formulation_cols:
            if col in group.columns:
                row[col] = group[col].dropna().iloc[0] if not group[col].dropna().empty else np.nan

        # Assign Date from Day 0
        day0_row = group[group["Day_Num"] == 0]
        row["Date"] = day0_row["Date"].iloc[0] if not day0_row.empty else np.nan

        # Assign Pilot
        pilot_val = group["Pilot"].dropna()
        row["Pilot"] = pilot_val.iloc[0] if not pilot_val.empty else ""

        # 🆕 Create a mapping: Day_Num → Baseline has * (True/False)
        baseline_marker = {}
        for i, r in group.iterrows():
            if "*" in str(r.get("Baseline", "")):
                baseline_marker[r["Day_Num"]] = True
                row["Baseline"] = "*"
            else:
                baseline_marker[r["Day_Num"]] = False
                 


        # Iterate over all days
        for day in range(max_day + 1):
            day_row = group[group["Day_Num"] == day]

            if not day_row.empty:
                # Get Foam (cc)
                row[f"Day {day} - Foam (cc)"] = day_row["Foam (cc)"].values[0]

                # Get Foam Texture
                foam_texture = day_row["Foam Texture"].values[0] if not pd.isna(day_row["Foam Texture"].values[0]) else ""

                # ✨ Add "*" to Foam Texture if baseline for this day has "*"
                if baseline_marker.get(day, False):
                    if foam_texture:
                        foam_texture = f"{foam_texture}, *"
                    else:
                        foam_texture = "*"

                row[f"Day {day} - Foam Texture"] = foam_texture

            else:
                row[f"Day {day} - Foam (cc)"] = np.nan
                row[f"Day {day} - Foam Texture"] = ""

        output_rows.append(row)

    return output_rows

def check_password():
    st.title("Foam Parser - No Oil")
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
def assign_pilot_column(df):
    df["Pilot"] = df["SampleID"].apply(lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None)
    df["Pilot"] = df["Dilution Ratio"].apply(lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None)

    return df  

def update_sampleid_with_sonicated_status(df):
    df = df.copy()

    def update_sampleid(row):
        sampleid = str(row["SampleID"]).strip() if pd.notna(row["SampleID"]) else ""
        if pd.isna(row["Sonicated"]):
            return sampleid  # No change
        if row["Sonicated"] == False and "not sonicated" not in sampleid.lower():
            return f"{sampleid} - Not Sonicated"
        if row["Sonicated"] == True and "sonicated" not in sampleid.lower():
            return f"{sampleid} - Sonicated"
        return sampleid  # already handled

    df["SampleID"] = df.apply(update_sampleid, axis=1)

    return df

def sort_columns_custom(df):
    # Step 1: Columns containing '%'
    percent_cols = [col for col in df.columns if ('%' in col) or ('ppm' in col)]
    # Step 2: Specific fixed columns
    fixed_order = ['SampleID', 'Date', 'concentrate manufacturing method (Ratio)', 'Dilution Ratio', 'Brine Type']
    # Step 3: Columns containing "Day"
    day_cols = [col for col in df.columns if 'Day' in col]
    # Step 4: Performance-related columns
    performance_cols = ['Concentrate Stability (RT)','Concentrate Stability (8C)', 'Concentrate Stability (4C)','Concentrate Stability (HT)', 'Dilution Stability', 'Zeta','Conductivity',
        'Size', 'PI', 'Initial Foam Temp (dilution Temp)', 'Temp Foam Monitoring', 'Sonicated', 'Baseline', 'Sample Description']
    # Combine and keep only those that exist in df
    all_desired_order = (percent_cols + fixed_order + day_cols + performance_cols)
    existing_cols = [col for col in all_desired_order if col in df.columns]
    # Add the rest of columns not yet included
    used = set(existing_cols)
    remaining_cols = [col for col in df.columns if col not in used]
    # Final sorted column list
    sorted_cols = existing_cols + remaining_cols
    return df[sorted_cols]


import pandas as pd
import re

def clean_dilution_ratio(df, dilution_col="Dilution Ratio", desc_col="Sample Description"):
    # Ensure the description column exists
    if desc_col not in df.columns:
        df[desc_col] = None

    # Extract valid YX and the rest
    extracted = df[dilution_col].astype(str).str.extract(r"^(?P<valid>\d+[A-Z])\s*(?P<rest>.*)", expand=True)

    # Keep only the YX value in "Dilution Ratio"
    df[dilution_col] = extracted["valid"]

    # Append extra text to "Sample Description" only if it's not empty
    new_desc = extracted["rest"].fillna("").str.strip()
    old_desc = df[desc_col].fillna("").astype(str).str.strip()

    # Combine old and new descriptions, avoiding leading/trailing spaces and "nan"
    df[desc_col] = (old_desc + " " + new_desc).str.strip()
    df[desc_col].replace(["", "nan", "None"], pd.NA, inplace=True)

    # Optional: if you really want to keep it as empty string instead of NaN:
    df[desc_col] = df[desc_col].fillna("")

    # If "Dilution Ratio" ends up empty, set to NaN
    df[dilution_col].replace("", pd.NA, inplace=True)

    return df

def make_sampleid_unique(df):
    df = df.copy()
    df["SampleID"] = df["SampleID"].astype(str)
    
    # Ensure consistent sorting
    df.sort_values(by=["SampleID", "Dilution Ratio", "Date", "Day"], inplace=True)
    
    # Group by SampleID and Dilution Ratio
    updated_rows = []

    for (sample_id, dilution), group in df.groupby(["SampleID", "Dilution Ratio"]):
        suffix = 0
        current_suffix = ""
        for idx, row in group.iterrows():
            if pd.notna(row["Day"]) and row["Day"].strip().lower() == "day 0":
                if suffix == 0:
                    current_suffix = ""  # original
                else:
                    current_suffix = f"-{suffix}"
                suffix += 1
            new_row = row.copy()
            new_row["SampleID"] = f"{sample_id}{current_suffix}"
            updated_rows.append(new_row)
    
    return pd.DataFrame(updated_rows)

