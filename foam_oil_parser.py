import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Foam (Oil) Sample Data Extractor")
st.title("🧪 Foam (Oil) Sample Data Extractor")

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, header=None)

    def parse_formulation(text):
        data = {
            "SampleID": re.search(r"\\((.*?)\\)", text).group(1).strip() if re.search(r"\\((.*?)\\)", text) else None
        }
        seen_keys_lower = set()
        parts = re.split(r'[,-]', text)
        for p in parts:
            p = p.strip()
            percent_match = re.match(r"(\\d+\\.?\\d*)%\\s*(.*)", p)
            if percent_match:
                val, chem = percent_match.groups()
                chem = re.sub(r"\\(.*?\\)", "", chem).strip()
                chem_key = f"{chem} (%)"
                if chem_key.lower() not in seen_keys_lower:
                    data[chem_key] = float(val)
                    seen_keys_lower.add(chem_key.lower())
                continue
            reverse_match = re.match(r"(.*?)\-(\\d+\\.?\\d*)%", p)
            if reverse_match:
                chem, val = reverse_match.groups()
                chem = re.sub(r"\\(.*?\\)", "", chem).strip()
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
            if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
                last_formulation = parse_formulation(cell)
                last_sample_id = last_formulation.get("SampleID")
                is_stable = np.nan
                for col_offset in range(1, min(8, df.shape[1])):
                    try:
                        col_val = str(df.iat[row, col_offset]).lower().strip()
                        if "unstable" in col_val:
                            is_stable = False
                            break
                        elif "stable" in col_val:
                            is_stable = True
                    except:
                        continue
                row += 1
                next_cell = str(df.iat[row, 0]).lower() if row < df.shape[0] else ""
                if "dilution" not in next_cell:
                    samples.append({"SampleID": last_sample_id, "Is_stable": is_stable, "Dilution": np.nan, **{k: v for k, v in last_formulation.items() if k != "SampleID"}})
                continue
            if "dilution" in cell.lower():
                dilution_row = df.iloc[row].fillna("").astype(str).tolist()
                last_dilution_text = " - ".join([x.strip() for x in dilution_row if x.strip()])
                last_date = df.iat[row, 2] if df.shape[1] > 2 else None
                row += 1
                if row >= df.shape[0]: break
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
                    if not re.match(r"^\\d+(\\.\\d+)?[hdm]?$", time_val, re.IGNORECASE):
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

    df = extract_all_samples_with_corrected_stability(df_raw)

    def extract_from_dilution(text):
        ratio = oil_pct = tube_volume = start_time = None
        cleaned_dilution = text
        if pd.notna(text):
            text = str(text)
            match_ratio = re.search(r"\\((\\d:\\d)\\)\\s*ratio", text)
            if match_ratio: ratio = match_ratio.group(1)
            match_oil = re.search(r"(\\d+)%\\s*oil", text, re.IGNORECASE)
            if match_oil: oil_pct = f"{match_oil.group(1)}%"
            match_ml = re.search(r"(\\d+)\\s*mL", text, re.IGNORECASE)
            if match_ml: tube_volume = match_ml.group(1)
            match_time = re.search(r"\\b(\\d{1,2}:\\d{2})\\b", text)
            if match_time: start_time = match_time.group(1)
            cleaned_dilution = re.sub(r"\\s*-\\s*\\d{1,2}:\\d{2}|\\(\\d:\\d\\)\\s*ratio|\\d+% oil|\\d+\\s*mL", "", text).strip(" -")
        return pd.Series([ratio, oil_pct, tube_volume, start_time, cleaned_dilution])

    df[["Ratio", "Oil (%)", "Tube Volume (mL)", "Start Time", "Dilution"]] = df["Dilution"].apply(extract_from_dilution)
    if "Time (min)" in df.columns:
        def convert_to_minutes(value):
            if pd.isna(value):
                return None
            value = str(value).strip().lower()
            if value.endswith("h"):
                return float(value[:-1]) * 60
            elif value.endswith("d"):
                return float(value[:-1]) * 24 * 60
            elif value.replace('.', '', 1).isdigit():
                return float(value)
            return None

        df["Time (min)"] = df["Time (min)"].apply(convert_to_minutes)
    else:
        st.error("❌ 'Time (min)' column is missing in the parsed dataframe.")
        df[["SampleID", "Dilution", "Date", "Start Time", "Ratio", "Oil (%)"]] = df[["SampleID", "Dilution", "Date", "Start Time", "Ratio", "Oil (%)"]].ffill()

    foam_data = pd.DataFrame()
    for column, label in [("Foam Layer (cc)", "Foam (cc)"), ("Foam Texture", "Foam Texture")]:
        temp = df[["SampleID", "Date", "Dilution", "Time (min)", column]].copy()
        temp['Column Name'] = "Time (" + temp['Time (min)'].astype(str) + f") - {label}"
        temp = temp.rename(columns={column: "Value"})
        foam_data = pd.concat([foam_data, temp], ignore_index=True)

    group_cols = ['SampleID', 'Date', 'Dilution']
    foam_pivot = foam_data.pivot_table(index=group_cols, columns='Column Name', values='Value', aggfunc='first').reset_index()
    baseline_map = df.groupby(group_cols)['sBaseline'].apply(lambda x: "*" if "*" in x.astype(str).values else "").reset_index()
    start_time_map = df.groupby(group_cols)['Start Time'].apply(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().any() else "").reset_index()
    static_cols = ["Ratio", "Oil (%)", "HS (%)", "Citric (%)", "CapB (%)", "AOS (%)", "APG (%)", "chinese HS (%)", "citric (%)", "CAPB (%)", "LBHP (%)"]
    static_info = df.groupby(group_cols)[static_cols].first().reset_index()

    final_df = foam_pivot.merge(baseline_map, on=group_cols, how='left')
    final_df = final_df.merge(start_time_map, on=group_cols, how='left')
    final_df = final_df.merge(static_info, on=group_cols, how='left')

    prefix_cols = ['SampleID', 'Date', 'Ratio', 'Oil (%)', 'Dilution', 'Start Time', 'Baseline']
    time_cols = sorted([col for col in final_df.columns if col.startswith("Time (")])
    all_cols = prefix_cols + time_cols + [col for col in static_cols if col not in prefix_cols]
    final_df = final_df[all_cols]

    st.success("✅ Parsing Complete")
    st.subheader(f"📊 {df['SampleID'].nunique()} Samples are extracted.")
    st.dataframe(df)

    st.download_button("📥 Download Multiple Line", df.to_csv(index=False).encode('utf-8'), "Parsed_Yates_Oil_Processed_Time.csv", "text/csv")
    st.download_button("📥 Download Single Line", final_df.to_csv(index=False).encode('utf-8'), "Parsed_Yates_Oil_Processed_Time_Single.csv", "text/csv")
