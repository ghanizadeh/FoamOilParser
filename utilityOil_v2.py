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
    temp_foam = np.nan
    sonic = np.nan
    brine_type=np.nan
    if pd.notna(text):
        # Extract start time
        match_time = re.search(r"\b(\d{1,2}:\d{2})\b", text)
        if match_time:
            start_time = match_time.group(1)
            text = re.sub(r"\b(\d{1,2}:\d{2})\b", "", text, flags=re.IGNORECASE)
            cleaned_dilution = re.sub(r"\b(\d{1,2}:\d{2})\b", "", text, flags=re.IGNORECASE)
        # Extract ratio
        #match_ratio = re.search(r'\b(\d:0\d)\b(?:\s*ratio)?', text, flags=re.IGNORECASE)
        match_ratio = re.search(r'\(?(\d+):0*(\d+)\)?(?:\s*ratio)?', text, flags=re.IGNORECASE)
        if match_ratio:
            #ratio = match_ratio.group(1)
            ratio = f"{match_ratio.group(1)}:{match_ratio.group(2)}"
            cleaned_dilution = re.sub(r"\(?\b(\d):0*(\d{1,2})\b\)?(?:\s*ratio)?", "", text, flags=re.IGNORECASE)
        # Extract and remove "X stream"  
        match_stream = re.search(r"\b(\d+)\s*stream\b", text, flags=re.IGNORECASE)
        if match_stream:
            stream_value = match_stream.group(1)
            ratio = f"{stream_value} stream"
            cleaned_dilution = re.sub(r"\b\d+\s*stream\b", "", text, flags=re.IGNORECASE)
        # Extract oil
        match_oil = re.search(r"(\d+)%\s*oil", text, re.IGNORECASE)
        if match_oil:
            oil_pct = f"{match_oil.group(1)}%"
            cleaned_dilution = re.sub(r"(\d+)%\s*oil", "", text, flags=re.IGNORECASE)
        # Extract mL
        match_ml = re.search(r"(\d+)\s*mL", text, re.IGNORECASE)
        if match_ml:
            tube_volume = match_ml.group(1)
            cleaned_dilution = re.sub(r"(\d+)\s*mL", "", text, flags=re.IGNORECASE)

        # Extract sonicated
        if re.search(r"[-\s]{0,10}(no|not)[-\s]{0,10}\w*sonic[\w-]*", text, flags=re.IGNORECASE):
            sonic = False
            cleaned_dilution = re.sub(r"[-\s]{0,10}(no|not)[-\s]{0,10}\w*sonic[\w-]*", "", text, flags=re.IGNORECASE)
        elif re.search(r"[-\s]{0,10}sonic[\w-]*", text, flags=re.IGNORECASE):
            sonic = True
            cleaned_dilution = re.sub(r"[-\s]{0,10}sonic[\w-]*", "", text, flags=re.IGNORECASE)
        # Extract and remove Xc (e.g. 45c, not 45cc)
        xc_match = re.search(r"(\d+)\s*c(?!c)", text, flags=re.IGNORECASE)
        if xc_match:
            temp_foam = int(xc_match.group(1))
            cleaned_dilution = re.sub(r"(\d+)\s*c(?!c)", "", text, flags=re.IGNORECASE)

        # Extract and remove Xcc (e.g. 5cc)
        xcc_match = re.search(r"(\d+)\s*cc", text, flags=re.IGNORECASE)
        if xcc_match:
            ini_foam = xcc_match.group(1).strip()
            cleaned_dilution = re.sub(r"(\d+)\s*cc", "", text, flags=re.IGNORECASE)

        # Extract and remove Xcc (e.g. 5cc)
        brine_match = re.search(r"synthetic brine", text, flags=re.IGNORECASE)
        if brine_match:
            brine_type = "Synthetic"
            cleaned_dilution = re.sub(r"synthetic brine", "", text, flags=re.IGNORECASE)
        #text = re.sub(r"(\d+)\s*cc", "", text, flags=re.IGNORECASE)
        # Now remove all matched patterns from text
        #cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}:\d{2}", "", cleaned_dilution)  # Remove time
        cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}-[A-Za-z]{3}", "", cleaned_dilution)  # Remove date
        #cleaned_dilution = re.sub(r"\b(\d:\d)\b(?:\s*ratio)?", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove (X:Y) ratio
        cleaned_dilution = re.sub(r"\(?\b(\d):0*(\d{1,2})\b\)?(?:\s*ratio)?", "", cleaned_dilution, flags=re.IGNORECASE)

        cleaned_dilution = re.sub(r"\s*\d+% oil", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove oil %
        #cleaned_dilution = re.sub(r"\s*\d+\s*mL", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove mL
         #Remove leftover " - " at ends or doubled spaces
        cleaned_dilution = re.sub(r"-", " ", cleaned_dilution).strip(" ")
        #cleaned_dilution = re.sub(r"\s*-\s*", " -", cleaned_dilution).strip(" -")
        #cleaned_dilution = re.sub(r"\s{2,}", " ", cleaned_dilution).strip()
    return pd.Series([ratio, oil_pct, tube_volume, start_time, cleaned_dilution,temp_foam,sonic,brine_type])

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
    concentrate_has_RT = np.nan
    concentrate_has_8c = np.nan
    concentrate_has_4c = np.nan
    concentrate_has_HT = np.nan
    dilution_stability = np.nan

    while row < df.shape[0]:
        cell = str(df.iat[row, 0]).strip()
        # Detect formulation row
        if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
            last_formulation = parse_formulation(cell)
            last_sample_id = last_formulation.get("SampleID")
            # Check up to 10 columns from column 1 to 10 (inclusive)
            is_stable = np.nan
            concentrate_has_RT = True
            concentrate_has_8c = np.nan
            concentrate_has_4c = np.nan
            concentrate_has_HT = np.nan
            for col_offset in range(1, min(11, df.shape[1])):  # 1 to 10 inclusive
                try:
                    col_val = str(df.iat[row, col_offset]).lower().strip()
                    #if "unstable" in col_val:
                    if (("unstable" in col_val.lower() or "cloudy" in col_val.lower()) and "concentrate" in col_val.lower() and not re.search(r"\b\d{1,2}C\b", col_val.lower(), flags=re.IGNORECASE)):
                        is_stable = False
                        concentrate_has_RT = False
                        concentrate_has_8c = False
                        concentrate_has_4c = False
                        concentrate_has_HT = False
                        break
                    if (("unstable dilution" in col_val.lower() or "cloudy dilution" in col_val.lower())) :
                        dilution_stability = False
                    else:
                        dilution_stability = True
                    if (("unstable" in col_val.lower() or "cloudy" in col_val.lower()) and  re.search(r"8\s*[cC]", col_val.lower())):
                        concentrate_has_8c = False
                    elif ("stable" in col_val.lower() and  re.search(r"8\s*[cC]", col_val.lower())):
                        concentrate_has_8c = True
                    if (("unstable" in col_val.lower() or "cloudy" in col_val.lower()) and  re.search(r"4\s*[cC]", col_val.lower())):
                        concentrate_has_4c = False
                    elif ("stable" in col_val.lower() and  re.search(r"4\s*[cC]", col_val.lower())):
                        concentrate_has_4c = True
                    if (("unstable" in col_val.lower() or "cloudy" in col_val.lower()) and  re.search(r"rt", col_val.lower())):
                        concentrate_has_RT = False
                    elif ("stable" in col_val.lower() and  re.search(r"rt", col_val.lower())):
                        concentrate_has_RT = True
                #elif "stable" in col_val and "unstable" not in col_val:
                #        is_stable = True
                except:
                    continue
            row += 1
            next_cell = str(df.iat[row, 0]).lower() if row < df.shape[0] else ""
            if 'dilution' not in next_cell:
                samples.append({
                    "SampleID": last_sample_id,
                    "Is_stable": is_stable,
                    "Concentrate Stability (RT)": concentrate_has_RT,
                    "Concentrate Stability (8C)": concentrate_has_8c,
                    "Concentrate Stability (4C)": concentrate_has_4c,
                    "Concentrate Stability (HT)": concentrate_has_HT,
                    "Dilution Stability": dilution_stability,
                    'Dilution Ratio': np.nan,
                    **{k: v for k, v in last_formulation.items() if k != "SampleID"}
                })
            continue
        if 'dilution' in cell.lower():
            # This is the dilution header
            dilution_row = df.iloc[row].fillna("").astype(str).tolist()
            last_dilution_text = " -  ".join([x.strip() for x in dilution_row if x.strip()])
            last_date = df.iat[row, 2] if df.shape[1] > 2 else None
            row += 1
            if row >= df.shape[0]:
                break

            header_row = df.iloc[row].fillna("").astype(str).tolist()
            header_map = {name.lower().strip(): idx for idx, name in enumerate(header_row)}
            row += 1

            # Now parse dilution data rows under this header
            while row < df.shape[0]:
                time_val = str(df.iat[row, 0]).strip()

                # Stop if new formulation or dilution row starts or empty row
                if time_val.lower().startswith("time") or time_val == "":
                    row += 1
                    continue
                if any(sym in time_val.lower() for sym in ["%", "ppm"]) and "(" in time_val:
                    # new formulation detected, break to outer loop
                    break
                if 'dilution' in time_val.lower():
                    # new dilution header detected, break to outer loop to update last_dilution_text
                    break
                if not re.match(r"^\d+(\.\d+)?[hdm]?$", time_val, re.IGNORECASE):
                    row += 1
                    continue

                # Append the sample for this dilution data row
                sample = {
                    "SampleID": last_sample_id,
                    'Dilution Ratio': last_dilution_text,
                    "Time (min)": time_val,
                    "Date": last_date,
                    "Baseline": df.iat[row, header_map.get("baseline", -1)] if "baseline" in header_map else None,
                    "Foam Layer (cc)": df.iat[row, header_map.get("foam layer (cc)", -1)] if "foam layer (cc)" in header_map else None,
                    "Foam Texture": df.iat[row, header_map.get("foam texture", -1)] if "foam texture" in header_map else None,
                    "Is_stable": is_stable,
                    "Concentrate Stability (8C)": concentrate_has_8c,
                    "Concentrate Stability (RT)": concentrate_has_RT,
                    "Concentrate Stability (4C)": concentrate_has_4c,
                    "Concentrate Stability (HT)": concentrate_has_HT,
                    "Dilution Stability": dilution_stability
                }
                sample.update({k: v for k, v in last_formulation.items() if k != "SampleID"})
                samples.append(sample)
                row += 1
            continue

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
    if "Concentrate manufacturing method (Ratio)" not in df.columns:
        df["Concentrate manufacturing method (Ratio)"] = ""
    else:
        df["Concentrate manufacturing method (Ratio)"] = df["Concentrate manufacturing method (Ratio)"].astype(str)
    # Function to extract ratio and clean dilution
    def process_dilution_text(text):
        if pd.isna(text):
            return "", text
        match = re.search(r"\(?(\d+):(\d+)\)?\s*ratio", text)
        ratio = match.group(1) if match else np.nan
        new_text = re.sub(r"\s*\(\d+:\d+\)\s*ratio", "", text).strip()
        new_text = new_text.replace("-", "").strip()  # Remove hyphens
        return ratio, new_text
    # Apply the function to each row
    df[['ExtractedRatio', 'CleanDilution']] = df['Dilution Ratio'].apply(
        lambda x: pd.Series(process_dilution_text(x))
    )
    # Update the Ratio and Dilution columns
    df["Concentrate manufacturing method (Ratio)"] = df['ExtractedRatio'].where(df['ExtractedRatio'] != "", df['Ratio'])
    df['Dilution Ratio'] = df['CleanDilution']
    # Drop temporary columns
    df.drop(columns=['ExtractedRatio', 'CleanDilution'], inplace=True)
    return df

def check_password():
    st.title("Foam Oil Sample Parser")
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

def sort_columns_custom(df):
    # Step 1: Columns containing '%'
    percent_cols = [col for col in df.columns if ('%' in col) or ('ppm' in col)]
    # Step 2: Specific fixed columns
    fixed_order = ['SampleID', 'Date', 'Concentrate manufacturing method (Ratio)', 'Dilution Ratio', 'Brine Type']
    # Step 3: Columns with "Day X - Date" and "Day X - Observation"
    day_pattern = re.compile(r'Day (\d+) - (Date|Observation)', re.IGNORECASE)
    day_dict = {}

    for col in df.columns:
        match = day_pattern.match(col)
        if match:
            day_num = int(match.group(1))
            label = match.group(2).capitalize()
            day_dict.setdefault(day_num, {})[label] = col

    # Interleave Date and Observation for each day in order
    time_cols = []
    for day in sorted(day_dict.keys()):
        if 'Date' in day_dict[day]:
            time_cols.append(day_dict[day]['Date'])
        if 'Observation' in day_dict[day]:
            time_cols.append(day_dict[day]['Observation'])
    # Step 4: Performance-related columns
    performance_cols = ['Concentrate Stability (4C)', 'Concentrate Stability (8C)',  'Dilution Stability', 'Initial Foam Temp (dilution Temp)', 
                        'Temp Foam Monitoring', 'Sonicated', 'Baseline', 'Sample Description']
    # Combine and keep only those that exist in df
    all_desired_order = (percent_cols + fixed_order + Time_cols + performance_cols)
    existing_cols = [col for col in all_desired_order if col in df.columns]
    # Add the rest of columns not yet included
    used = set(existing_cols)
    remaining_cols = [col for col in df.columns if col not in used]
    # Final sorted column list
    sorted_cols = existing_cols + remaining_cols
    return df[sorted_cols]


def make_sampleid_unique(df):
    df = df.copy()
    df["SampleID"] = df["SampleID"].astype(str)
    
    # Ensure consistent sorting
    df.sort_values(by=["SampleID", "Dilution Ratio", "Date", "Time (min)"], inplace=True)
    
    # Group by SampleID and Dilution Ratio
    updated_rows = []

    for (sample_id, dilution), group in df.groupby(["SampleID", "Dilution Ratio"]):
        suffix = 0
        current_suffix = ""
        for idx, row in group.iterrows():
            if pd.notna(row["Time (min)"]) and row["Time (min)"].strip().lower() == "0.0":
                if suffix == 0:
                    current_suffix = ""  # original
                else:
                    current_suffix = f"-{suffix}"
                suffix += 1
            new_row = row.copy()
            new_row["SampleID"] = f"{sample_id}{current_suffix}"
            updated_rows.append(new_row)
    
    return pd.DataFrame(updated_rows)

def extract_half_life_samples(df):
    df = df.applymap(lambda x: x.replace('*', '') if isinstance(x, str) else x)
    # Select only foam-related columns
    foam_cols = [col for col in df.columns if col.startswith("Time (") and "Foam" in col]
    if not foam_cols:
        raise ValueError("No columns found in the format 'Time (X[h]) - Foam (cc)'")

    time_values = []
    valid_foam_cols = []

    for col in foam_cols:
        time_str = col.split('(')[1].split(')')[0]
        try:
            if time_str.endswith('h'):  # Time in hours
                time_val = float(time_str[:-1]) * 60
            else:  # Time in minutes
                time_val = float(time_str)
            time_values.append(time_val)
            valid_foam_cols.append(col)
        except ValueError:
            continue  # skip malformed time strings

    if not valid_foam_cols:
        raise ValueError("No foam columns with valid time values found.")

    # Sort columns by actual time in minutes
    sorted_foam_cols = [col for _, col in sorted(zip(time_values, valid_foam_cols))]

    def has_half_life(row):
        foam_values = row[sorted_foam_cols].values.astype(float)
        initial_value = foam_values[0]
        if initial_value <= 0 or pd.isna(initial_value):
            return False
        half_value = initial_value / 2.0
        return any(v == half_value for v in foam_values[1:])

    return df[df.apply(has_half_life, axis=1)].reset_index(drop=True)

 
 
