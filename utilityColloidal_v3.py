import pandas as pd
import numpy as np
import re
import math
import streamlit as st

# Function to parse formulation text
def parse_formulation(text, index=None):
    # Step 1: extract and remove the final (SampleID)
    matches = re.findall(r"\(([^)]*?)\)", text)
    if matches:
        sample_id = matches[-1].strip()
        text = text.rsplit(f"({sample_id})", 1)[0].strip()
    else:
        sample_id = f"Sample {index}" if index is not None else "Unknown"

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
    return "%" in cell and "dilution" not in cell.lower()

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
    results = df['Dilution'].apply(clean_dilution)

    df['Dilution'] = results.apply(lambda x: x[0])
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

    for (sample_id, dilution), group in df.groupby(["SampleID", "Dilution"]):
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