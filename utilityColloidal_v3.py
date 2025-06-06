import pandas as pd
import numpy as np
import re
import math
import streamlit as st

def main_parse(df_input):
    records = []
    i = 0
    n = len(df_input)

    while i < n:
        row = df_input.iloc[i]
        first_cell = str(row[0]) if pd.notna(row[0]) else ""

        if is_formulation_row(first_cell):
            current_sample = parse_formulation(first_cell, index=i)
            added = False
            i += 1

            while i < n:
                row = df_input.iloc[i]
                first_cell = str(row[0]) if pd.notna(row[0]) else ""

                # New formulation starts
                if is_formulation_row(first_cell):
                    break

                # New dilution block
                if "x dilution" in first_cell.lower():
                    current_dilution = " - ".join(str(cell).strip() for cell in row[:11] if pd.notna(cell))
                    i += 1

                    # Skip optional header row if it contains "Date" and 'pH_Dilution'
                    if i < n:
                        next_row = df_input.iloc[i]
                        if any("Date" in str(cell) for cell in next_row) and any('pH' in str(cell) for cell in next_row):
                            column_headers = [str(col).strip() for col in next_row[3:] if pd.notna(col)]
                            i += 1
                        else:
                            column_headers = []

                    # Read observations
                    while i < n:
                        obs_row = df_input.iloc[i]
                        obs_first = str(obs_row[0]) if pd.notna(obs_row[0]) else ""

                        # Stop if next dilution or formulation starts
                        if is_formulation_row(obs_first) or ("x dilution" in obs_first.lower()):
                            break

                        # Skip repeated headers
                        if "date" in obs_first.lower() and 'pH' in str(obs_row[1]).lower():
                            i += 1
                            continue

                        # Valid observation row
                        if obs_first.startswith("Day") or pd.notna(obs_row[1]):
                            sample_record = current_sample.copy()
                            sample_record['Dilution Ratio'] = current_dilution
                            sample_record["Day"] = obs_row[0] if obs_first.startswith("Day") else None
                            sample_record["Date"] = obs_row[1]
                            sample_record['pH_Dilution'] = obs_row[2] if pd.notna(obs_row[2]) else None

                            for idx, header in enumerate(column_headers):
                                val = obs_row[3 + idx] if (3 + idx) < len(obs_row) else None
                                sample_record[header] = val if pd.notna(val) else None

                            records.append(sample_record)
                            added = True

                        i += 1

                    continue  # process next dilution or row

                i += 1  # move to next row if not a dilution

            if not added:
                records.append(current_sample)

        else:
            i += 1

    return pd.DataFrame(records)



# Function to parse formulation text
def parse_formulation(text, index=None):
    # Step 1: extract and remove the final (SampleID)
    matches = re.findall(r"\(([^)]*?)\)", text)
    if "(" in text:
        sample_id = text.split("(", 1)[1].strip(" )")
        text = text.rsplit("(", 1)[0].strip()
    else:
        sample_id = f"Sample {index}" if index is not None else "Unknown"
        print(text)

    data = {"SampleID": sample_id}
    seen_keys_lower = set()
    parts = re.split(r'[,-]', text)

    for p in parts:
        p = p.strip()

        # Match "12% HS"
        percent_match = re.match(r"(\d+\.?\d*)%\s*(.*)", p)
        if percent_match:
            val, chem = percent_match.groups()
            chem = chem.strip()
            chem_key = f"{chem} (%)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())
            continue

        # Match "HS-12%"
        reverse_percent_match = re.match(r"(.*?)-(\d+\.?\d*)%", p)
        if reverse_percent_match:
            chem, val = reverse_percent_match.groups()
            chem = chem.strip()
            chem_key = f"{chem} (%)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())
            continue

        # Match "1500ppm CapB"
        ppm_match = re.match(r"(\d+\.?\d*)\s*ppm\s*(.*)", p, re.IGNORECASE)
        if ppm_match:
            val, chem = ppm_match.groups()
            chem = chem.strip()
            chem_key = f"{chem} (ppm)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())
            continue

        # Match "CapB-1500ppm"
        reverse_ppm_match = re.match(r"(.*?)-(\d+\.?\d*)\s*ppm", p, re.IGNORECASE)
        if reverse_ppm_match:
            chem, val = reverse_ppm_match.groups()
            chem = chem.strip()
            chem_key = f"{chem} (ppm)"
            if chem_key.lower() not in seen_keys_lower:
                data[chem_key] = float(val)
                seen_keys_lower.add(chem_key.lower())
            continue

    return data

# Helper function to detect if a row starts a formulation
def is_formulation_row(cell):
    return "%" in cell and 'Dilution Ratio' not in cell.lower()

def refine_dilution_column(df):
    df = df.copy()
    
    # Ensure target columns exist
    for col in ['Temperature', 'Discontinued', 'Removed for space', 'Stability', 'Ratio']:
        if col not in df.columns:
            df[col] = pd.NA

    # Define cleaning function
    def clean_dilution(dil):
        temp = None
        discontinued = None
        removed_space = None
        stability = None
        ratio = None
        
        if pd.isna(dil):
            return pd.NA, temp, discontinued, removed_space, stability, ratio

        dil = str(dil)

        # Handle "discontinued"
        if "discontinued" in dil.lower():
            discontinued = True
            #dil = re.sub(r'discontinued', '', dil, flags=re.IGNORECASE)

        # Handle "removed for space"
        if "removed for space" in dil.lower():
            removed_space = True
            #dil = re.sub(r'removed for space', '', dil, flags=re.IGNORECASE)

        # Handle "unstable" or "stable" (both set False in Stability)
        if re.search(r'\bunstable\b', dil, flags=re.IGNORECASE) or re.search(r'\bstable\b', dil, flags=re.IGNORECASE):
            stability = False
            #dil = re.sub(r'\bunstable\b|\bstable\b', '', dil, flags=re.IGNORECASE)

        # Handle "Temperature" extraction like "25C", "30C"
        match_temp = re.search(r'(\d+)\s*[Cc]', dil)
        if match_temp:
            temp = int(match_temp.group(1))
            #dil = re.sub(r'\d+\s*[Cc]', '', dil)

        # Handle Ratio extraction like "1:2"
        match_ratio = re.search(r'(\d)\s*:\s*(\d)', dil)
        if match_ratio:
            ratio = f"{match_ratio.group(1)}-{match_ratio.group(2)}"
            #dil = re.sub(r'\d\s*:\s*\d', '', dil)

        # Remove unwanted words/symbols
        dil = re.sub(r'\b(?:Temp|ratio|Dilution)\b|[-()]+', '', dil, flags=re.IGNORECASE)

        # Final cleanup: strip extra whitespace
        dil = dil.strip()

        return dil, temp, discontinued, removed_space, stability, ratio

    # Apply cleaning
    results = df['Dilution Ratio'].apply(clean_dilution)

    df['Dilution Ratio'] = results.apply(lambda x: x[0])
    df['Temperature'] = results.apply(lambda x: x[1])
    df['Discontinued'] = results.apply(lambda x: x[2])
    df['Removed for space'] = results.apply(lambda x: x[3])
    df['Stability'] = results.apply(lambda x: x[4])
    df['Ratio'] = results.apply(lambda x: x[5])

    return df


# Redefine the function with corrected logic based on most recent clarification:
# Keep ALL rows but remove duplicate 'Observation' values (excluding "Clear solution") per SampleID + Dilution group,
# keeping only the first occurrence of each unique value.
def retain_observation_sequence(df):
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")
    result = []

    for (sample_id, dilution), group in df.groupby(["SampleID", 'Dilution Ratio']):
        group = group.sort_values("Date")
        first_row = group.iloc[0]
        last_row = group.iloc[-1]

        # Initialize set to track seen non-clear Observation values (excluding "Clear solution")
        seen = set()
        selected_rows = [first_row]  # Start with first row

        # Iterate through all rows except the first and last
        for i in range(1, len(group) - 1):
            row = group.iloc[i]
            obs = row["Observation"]
            if pd.notna(obs) and obs != "Clear solution" and obs not in seen:
                selected_rows.append(row)
                seen.add(obs)

        selected_rows.append(last_row)  # Always include last row

        result.append(pd.DataFrame(selected_rows))

    return pd.concat(result, ignore_index=True)


def create_columns(df):
    required_cols = ["Stability", "Temperature", "Removed for space", "Discontinued", "Ratio"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan  # or you can use any default value like '' or 0
    return df


def remove_observation(df):
    df = df.copy()

    # Extract numeric day value safely, allowing for NaNs
    df["Day_Num"] = df["Day"].str.extract(r'(\d+)')
    df["Day_Num"] = df["Day_Num"].astype("Int64")  # Nullable integer type

    # Remove rows without valid Day_Num
    df = df[df["Day_Num"].notna()]

    # Sort
    df = df.sort_values(by=["SampleID", 'Dilution Ratio', "Day_Num"]).reset_index(drop=True)

    # Identify where the observation changes
    df["Obs_Change_Group"] = (df["Observation"] != df["Observation"].shift()).cumsum()

    # Keep first and last of each observation group
    first_last_obs = df.groupby(["SampleID", 'Dilution Ratio', "Obs_Change_Group"]).agg(
        {'Day_Num': ['idxmin', 'idxmax']}
    )
    idx_to_keep = set(first_last_obs["Day_Num"]["idxmin"]).union(first_last_obs["Day_Num"]["idxmax"])

    # Also keep the first and last overall Day_Num per SampleID/Dilution
    first_last_day = df.groupby(["SampleID", 'Dilution Ratio']).agg(
        first_idx=("Day_Num", "idxmin"),
        last_idx=("Day_Num", "idxmax")
    )
    idx_to_keep.update(first_last_day["first_idx"])
    idx_to_keep.update(first_last_day["last_idx"])

    # Filter the dataframe
    result_df = df.loc[sorted(idx_to_keep)].drop(columns=["Obs_Change_Group", "Day_Num"]).reset_index(drop=True)

    return result_df


import pandas as pd

def recompute_days_from_day0(df):
    df = df.copy()

    # Parse date while preserving the year in the string (like 24 in 25-Jun-24)
    # df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    def process_group(group):
        group = group.sort_values("Date")
        
        # Try to find Day 0
        day0_row = group[group["Day"].str.contains("Day 0", na=False)]
        base_date = None

        if not day0_row.empty:
            base_date = day0_row["Date"].iloc[0]
        else:
            base_date = group["Date"].min()

        # Calculate day difference from base_date
        if pd.notna(base_date):
            group["Day"] = group["Date"].apply(
                lambda d: f"Day {(d - base_date).days}" if pd.notna(d) else None
            )

        return group

    # Apply per SampleID–Dilution group
    df = df.groupby(["SampleID", 'Dilution Ratio'], group_keys=False).apply(process_group)

    # Format date back to short format
    df["Date"] = df["Date"].dt.strftime("%d-%b-%y")

    return df


    # Apply group-wise
    df = df.groupby(["SampleID", 'Dilution Ratio'], group_keys=False).apply(process_group)

    # Restore short format of Date
    df["Date"] = df["Date"].dt.strftime("%d-%b-%Y")

    return df


def replace_text_in_df(df):
    # Dictionary of regex patterns and their replacements
    replacements = {
        r'\bconc\w*\.?': 'concentrate',  # handles 'conc' and 'conc.' (case-insensitive)
        r'\bCap B\b': 'CapB',
        r'\bcap B\b': 'CapB',
        r'\bCAPB\b': 'CapB',
        r'\bCAPB\b': 'CapB',
        r'\bcitiric\b': 'Citric',
        r'\bcitric\b': 'Citric',
        r'\bObservations\b': 'Observation',
        r'\bHS+\s': 'HS, ',
    }
    def replace_text(val):
        if isinstance(val, str):
            for pattern, repl in replacements.items():
                val = re.sub(pattern, repl, val, flags=re.IGNORECASE)
        return val
    return df.applymap(replace_text)


def clean_Dilution(df):
    # Ensure 'Tempreture' column exists
    if 'Tempreture' not in df.columns:
        df['Tempreture'] = None

    def process_group(group):
        dilution_val = group.name[1]
        # Match patterns like 'Temp 70C', '70 C', '(70C)', 'Temp70C'
        match = re.search(r'(?:Temp\s*)?(\d+\s*[°]?[Cc])', dilution_val)
        if match:
            temp = match.group(1).replace(' ', '')  # Clean like '70 C' to '70C'
            group['Tempreture'] = temp
            # Remove the temp part from dilution
            cleaned_dilution = re.sub(r'(?:Temp\s*)?\(?\d+\s*[°]?[Cc]\)?', '', dilution_val, flags=re.IGNORECASE).strip(' -')
            #group['Dilution Ratio'] = cleaned_dilution.strip()
        return group

    df = df.groupby(['SampleID', 'Dilution Ratio'], group_keys=False).apply(process_group)
    return df

def extract_duplicate_day0_samples(df):
    # Filter only Day 0 rows
    day0_df = df[df["Day"].astype(str).str.strip().str.lower() == "day 0"].copy()

    # Find duplicated combinations of SampleID, Dilution, and Date
    duplicates = day0_df.duplicated(subset=["SampleID", 'Dilution Ratio'], keep=False)

    # Return only those duplicated rows
    return day0_df[duplicates]

def extract_from_dilution(text):
    ratio = None
    tube_volume = None
    cleaned_dilution = text
    temp = np.nan
    sonic = np.nan
    brine_type=np.nan
    Discontinued = np.nan
    pH_Concentrate = np.nan
    has_4c = np.nan
    has_8c = np.nan
    has_RT = np.nan
    has_HT = np.nan
    date = np.nan    
    dilution_stability = np.nan

    if pd.notna(text):
        #text = re.sub(r"-", " ", text).strip(" ")
        # Extract pH_Concentrate
        pH_match = re.search(r'\b(?:pH[:\s]*([0-9]*\.?[0-9]+)|([0-9]*\.?[0-9]+)\s*pH)\b', text, flags=re.IGNORECASE)
        if pH_match:
               pH_Concentrate = float(pH_match.group(1) or pH_match.group(2))
        # Extract Discontinued
        Discontinued_match = re.search(r'\bdiscontinue\w*', text, flags=re.IGNORECASE)
        if Discontinued_match:
            Discontinued = True
        pattern = r'-\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:\s+(conc\w*))?'
        date_matches = re.findall(pattern, text, flags=re.IGNORECASE)
        # Extract ratio
        match_ratio = re.search(r'\(?([0-9]):0*([0-9])\)?(?:\s*ratio)?', text, flags=re.IGNORECASE)
        if match_ratio:
            ratio = f"{match_ratio.group(1)}::{match_ratio.group(2)}"
        # Extract and remove "X stream"  
        match_stream = re.search(r"\b(\d+)\s*stream\b", text, flags=re.IGNORECASE)
        if match_stream:
            stream_value = match_stream.group(1)
            ratio = f"{stream_value} stream"
        # Extract mL
        match_ml = re.search(r"(\d+)\s*mL", text, re.IGNORECASE)
        if match_ml:
            tube_volume = match_ml.group(1)
        # Extract sonicated
        if re.search(r"[-\s]{0,10}(no|not)[-\s]{0,10}\w*sonic[\w-]*", text, flags=re.IGNORECASE):
            sonic = False
        elif re.search(r"[-\s]{0,10}sonic[\w-]*", text, flags=re.IGNORECASE):
            sonic = True
        # Extract and remove Xc (e.g. 45c, not 45cc)
        temp_match = re.search(r'\b(?:Temp\s*)?(\d{2})\s*[cC](?![cC])(?:\s*Temp)?\b', text, flags=re.IGNORECASE)
        if temp_match:
            temp = int(temp_match.group(1))
        room_temp_match = re.search(r'\b(?:Room\s*Temperature|Room\s*T|RT)\b', text, flags=re.IGNORECASE)
        if room_temp_match:
            temp = 'RT'
        # extract Date
        date_match = re.search(r'-\s*(Jan|Feb|March|April|May|June|July|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:\s+concentrate)?\b', text, flags=re.IGNORECASE)   
        if date_match and date_match.lastindex >= 2:
            date = f"{date_match.group(1)} {date_match.group(2)}"
        elif date_match and date_match.lastindex < 2:
            date = date_match
        # Extract and remove Xcc (e.g. 5cc)
        xcc_match = re.search(r"(\d+)\s*cc", text, flags=re.IGNORECASE)
        if xcc_match:
            tube_volume = xcc_match.group(1).strip()
        # Extract and remove Xcc (e.g. 5cc)
        brine_match = re.search(r"synthetic brine", text, flags=re.IGNORECASE)
        if brine_match:
            brine_type = "Synthetic Brine"
            cleaned_dilution = re.sub(r"synthetic brine", "", text, flags=re.IGNORECASE)
        else:
            brine_type = "Field Brine"
            cleaned_dilution = re.sub(r"Field brine", "", text, flags=re.IGNORECASE)

        # stability Extraction
        if (("unstable" in text.lower() or "cloudy" in text.lower()) and re.search(r"\bconc\w*\.?\b", text.lower()) and 
                    not re.search(r'\b(8|4)[cC]\b', text.lower(), flags=re.IGNORECASE)):
            
            has_RT = False
            has_8c = False
            has_4c = False
            has_HT = False
        if (("unstable dilution" in text.lower() or "cloudy dilution" in text.lower())) :
            dilution_stability = False
        else:
            dilution_stability = True
        if (("unstable" in text.lower() or "cloudy" in text.lower()) and  re.search(r"8\s*[cC]", text.lower())):
            has_8c = False
        elif ("stable" in text.lower() and  re.search(r"8\s*[cC]", text.lower())):
            has_8c = True
        if (("unstable" in text.lower() or "cloudy" in text.lower()) and  re.search(r"4\s*[cC]", text.lower())):
            has_4c = False
        elif ("stable" in text.lower() and  re.search(r"4\s*[cC]", text.lower())):
            has_4c = True
        if (("unstable" in text.lower() or "cloudy" in text.lower()) and  re.search(r"rt", text.lower())):
            has_RT = False
        elif ("stable" in text.lower() and  re.search(r"rt", text.lower())):
            has_RT = True

        #cleaned_dilution = re.sub(r'\b(?:Temp\s*)?(\d{2})\s*[cC](?![cC])(?:\s*Temp)?\b', "", cleaned_dilution, flags=re.IGNORECASE) # remove XC-Temp XC
        #cleaned_dilution = re.sub(r'\bdiscontinue\w*', '', cleaned_dilution)  # Remove Discontinue
        #cleaned_dilution = re.sub(r'\b(?:Room\s*Temperature|Room\s*T|RT)\b', "", cleaned_dilution, flags=re.IGNORECASE) # Remove RT-Room T-Room Temperature
        #cleaned_dilution = re.sub(r'\b(?:-?\s*(Jan|Feb|March|April|May|June|July|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})(?:\s+concentrate)?\b', "", cleaned_dilution, flags=re.IGNORECASE) # Remove Date concentrate-Date
        # Now remove all matched patterns from text
        #cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}:\d{2}", "", cleaned_dilution)  # Remove time
        #cleaned_dilution = re.sub(r"\s*-\s*\d{1,2}-[A-Za-z]{3}", "", cleaned_dilution)  # Remove date
        #cleaned_dilution = re.sub(r'\b(?:Temp\s*)?(\d{2})\s*[cC](?![cC])(?:\s*Temp)?\b', "", cleaned_dilution, flags=re.IGNORECASE)  
        #cleaned_dilution = re.sub(r'\(?([0-9]):0*([0-9])\)?(?:\s*ratio)?', "", cleaned_dilution, flags=re.IGNORECASE) # Remove (X:Y) ratio
        #cleaned_dilution = re.sub(r"\s*\d+\s*mL", "", cleaned_dilution, flags=re.IGNORECASE)  # Remove mL
         #Remove leftover " - " at ends or doubled spaces
        #cleaned_dilution = re.sub(r"-", " ", cleaned_dilution).strip(" ")
        #cleaned_dilution = re.sub(r"\s*-\s*", " -", cleaned_dilution).strip(" -")
        #cleaned_dilution = re.sub(r"\s{2,}", " ", cleaned_dilution).strip()
        #cleaned_dilution = re.sub(r'\b\d+[Xx]\s+Dilution\s*-?\s*', '', cleaned_dilution, flags=re.IGNORECASE) # Remove "XY Dilution -" or "Xy Dilution -"
        #cleaned_dilution = re.sub(r'nan ', '', cleaned_dilution, flags=re.IGNORECASE) # Remove "nan -"
    return pd.Series([cleaned_dilution, temp, Discontinued, ratio, has_4c, has_8c, has_RT, has_HT, dilution_stability, pH_Concentrate, brine_type,sonic,date])

def parse_SampleDescription (text):
    has_4c = np.nan
    has_8c = np.nan
    has_RT = np.nan
    has_HT = np.nan
    date = np.nan
    dilution_stability = np.nan
    if (("unstable" in text.lower() or "cloudy" in text.lower()) and re.search(r"\bconc\w*\.?\b", text.lower()) and 
                not re.search(r'\b(8|4)[cC]\b', text.lower(), flags=re.IGNORECASE)):
        
        has_RT = False
        has_8c = False
        has_4c = False
        has_HT = False
    elif ("stable" in text.lower() and re.search(r"\bconc\w*\.?\b", text.lower()) and 
                not re.search(r'\b(8|4)[cC]\b', text.lower(), flags=re.IGNORECASE)):
        
        has_RT = True
    if (("unstable dilution" in text.lower() or "cloudy dilution" in text.lower())) :
        dilution_stability = False
    else:
        dilution_stability = True
    if (("unstable" in text.lower() or "cloudy" in text.lower()) and  re.search(r"8\s*[cC]", text.lower())):
        has_8c = False
    elif ("stable" in text.lower() and  re.search(r"8\s*[cC]", text.lower())):
        has_8c = True
    if (("unstable" in text.lower() or "cloudy" in text.lower()) and  re.search(r"4\s*[cC]", text.lower())):
        has_4c = False
    elif ("stable" in text.lower() and  re.search(r"4\s*[cC]", text.lower())):
        has_4c = True
    if (("unstable" in text.lower() or "cloudy" in text.lower()) and  (re.search(r"room temperature", text.lower()) or re.search(r"rt", text.lower()) or re.search(r"room t", text.lower()))):
        has_RT = False
    elif ("stable" in text.lower() and  (re.search(r"room temperature", text.lower()) or re.search(r"rt", text.lower()) or re.search(r"room t", text.lower()))):
        has_RT = True
    return pd.Series([has_4c, has_8c, has_RT, has_HT,dilution_stability])

def reshape_df(df):
    # First, select only relevant columns to pivot
    df_subset = df[['SampleID', 'Dilution Ratio', 'Day', 'Observation', 'Date']].copy()

    # Pivot the 'Observation' column by Day
    obs_pivot = df_subset.pivot_table(
        index=['SampleID', 'Dilution Ratio'],
        columns='Day',
        values='Observation',
        aggfunc='first'
    )
    obs_pivot.columns = [f"{col} - Observation" for col in obs_pivot.columns]

    # Pivot the 'Date' column by Day
    date_pivot = df_subset.pivot_table(
        index=['SampleID', 'Dilution Ratio'],
        columns='Day',
        values='Date',
        aggfunc='first'
    )
    date_pivot.columns = [f"{col} - Date" for col in date_pivot.columns]

    # Combine all wide-format data
    wide_df = pd.concat([obs_pivot, date_pivot], axis=1).reset_index()

    # Merge with other columns (only take the first row for each group to avoid duplication)
    other_cols = df.drop(columns=['Observation', 'Date', 'Day']).drop_duplicates(subset=['SampleID', 'Dilution Ratio'])
    final_df = pd.merge(other_cols, wide_df, on=['SampleID', 'Dilution Ratio'], how='left')
    return final_df

def only_keep_dilution(text):
    if isinstance(text, str):
        # Search for pattern like '20x Dilution' or '5X Dilution'
        match = re.search(r'\b(\d+[xX]\s*Dilution)\b', text)
        if match:
            return match.group(1)
    return ""  # or np.nan if you prefer

def sort_columns_custom(df):
    # Step 1: Columns containing '%'
    percent_cols = [col for col in df.columns if ('%' in col) or ('ppm' in col)]
    print(percent_cols)
    # Step 2: Specific fixed columns

    fixed_order = ['pH_Dilution', 'Density (RT)', 'SampleID', 'Concentrate manufacturing method (Ratio)', 
                   'Dilution Ratio', 'Brine Type', 'Tempreture', 'Sample Description']
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
    performance_cols = ['Concentrate Stability (4C)', 'Concentrate Stability (8C)', 'Concentrate Stability (RT)','Concentrate Stability (HT)',
                        'Dilution Stability']
    # Combine and keep only those that exist in df
    all_desired_order = (percent_cols + fixed_order + time_cols + performance_cols)
    existing_cols = [col for col in all_desired_order if col in df.columns]
    # Add the rest of columns not yet included
    used = set(existing_cols)
    remaining_cols = [col for col in df.columns if col not in used]
    # Final sorted column list
    sorted_cols = existing_cols + remaining_cols
    return df[sorted_cols]

def remove_dilution_pattern(text):
    if isinstance(text, str):
        # Remove patterns like '20x Dilution' or '5X Dilution' (case-insensitive)
        return re.sub(r'\b\d+[xX]\s*Dilution\b', '', text).strip()
    return text


