import streamlit as st
import pandas as pd
import utilityNoOil as utl
from io import StringIO

st.set_page_config(page_title="Yates Oil Sample Parser", layout="wide")
st.title("Yates Oil Sample Parser")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    st.success("Parsing complete!")

    # Step 1: Read and parse the file
    df = pd.read_csv(uploaded_file, header=None, encoding="cp1252")
    df = utl.extract_all_samples_with_corrected_stability(df)
    df["Ratio"] = pd.NA
    df["Oil (%)"] = pd.NA
    df["Tube Volume (mL)"] = pd.NA
    df["Start Time"] = pd.NA

    df[["Ratio", "Oil (%)", "Tube Volume (mL)", "Start Time", "Dilution"]] = df["Dilution"].apply(utl.extract_from_dilution)
    df["Dilution"] = df["Dilution"].str.replace(r"\b[Dd]ilution\b", "", regex=True).str.strip()
    df["Dilution"] = df["Dilution"].str.replace(r"-{2,}", " - ", regex=True).str.strip()

    parsed_df = df.copy()
    parsed_df["Time (min)"] = parsed_df["Time (min)"].apply(utl.convert_to_minutes)
    parsed_df[['SampleID', 'Dilution', 'Date', 'Start Time', 'Ratio', 'Oil (%)']] = parsed_df[['SampleID', 'Dilution', 'Date', 'Start Time', 'Ratio', 'Oil (%)']].ffill()
    parsed_df['Time (min)'] = parsed_df['Time (min)'].astype(str)

    foam_data = pd.DataFrame()
    for column, label in [('Foam Layer (cc)', 'Foam (cc)'), ('Foam Texture', 'Foam Texture')]:
        temp = parsed_df[['SampleID', 'Date', 'Dilution', 'Time (min)', column]].copy()
        temp['Column Name'] = "Time (" + temp['Time (min)'] + f") - {label}"
        temp = temp.rename(columns={column: "Value"})
        foam_data = pd.concat([foam_data, temp], ignore_index=True)

    group_cols = ['SampleID', 'Date', 'Dilution']
    foam_pivot = foam_data.pivot_table(index=group_cols, columns='Column Name', values='Value', aggfunc='first').reset_index()
    baseline_map = parsed_df.groupby(group_cols)['Baseline'].apply(lambda x: "*" if "*" in x.astype(str).values else "").reset_index()
    start_time_map = parsed_df.groupby(group_cols)['Start Time'].apply(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().any() else "").reset_index()

    static_cols = ['Ratio', 'Oil (%)', 'HS (%)', 'Citric (%)', 'CapB (%)', 'AOS (%)', 'APG (%)',
                   'chinese HS (%)', 'LBHP (%)']
    static_info = parsed_df.groupby(group_cols)[static_cols].first().reset_index()

    final_df = foam_pivot.merge(baseline_map, on=group_cols, how='left')
    final_df = final_df.merge(start_time_map, on=group_cols, how='left')
    final_df = final_df.merge(static_info, on=group_cols, how='left')

    prefix_cols = ['SampleID', 'Date', 'Ratio', 'Oil (%)', 'Dilution', 'Start Time', 'Baseline']
    time_cols = sorted([col for col in final_df.columns if col.startswith("Time (")])
    all_cols = prefix_cols + time_cols + [col for col in static_cols if col not in prefix_cols]
    final_df = final_df[all_cols]
    final_df = utl.sort_time_columns_in_df(final_df)
    final_df = utl.convert_time_columns_to_float_hour(final_df)

    # Show output for Parsed_Yates_Oil_Processed.csv
    st.subheader("Parsed_Yates_Oil_Processed.csv (Raw Processed)")
    st.dataframe(df)

    # Buttons to download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Parsed_Yates_Oil_Processed.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name="Parsed_Yates_Oil_Processed.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download Parsed_Yates_Oil_Processed_Time_Single_sorted.csv",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="Parsed_Yates_Oil_Processed_Time_Single_sorted.csv",
            mime="text/csv"
        )

    # Search functionality
    st.subheader("Search by SampleID")
    search_id = st.text_input("Enter SampleID (case-sensitive):")

    if search_id:
        result_df = df[df["SampleID"].str.lower() == search_id.lower()]
        if result_df.empty:
            st.warning("No matching SampleID found.")
        else:
            st.success(f"Found {len(result_df)} row(s) for SampleID: {search_id}")
            st.dataframe(result_df)
