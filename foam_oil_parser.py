import streamlit as st
import pandas as pd
import utilityOil as utl
from io import StringIO

st.set_page_config(page_title="Foam Oil Sample Parser", layout="wide")
if utl.check_password():
    st.title("Foam Oil Sample Parser")
    st.write("✅ Access granted! Continue with the app.")
    
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
        df["Dilution"] = df["Dilution"].fillna("00x")
        df["Date"] = df["Date"].fillna("No Date")
        unique_ID_multi = df['SampleID'].nunique()
        df = utl.extract_ratio_from_dilution(df)

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

        foam_data["Dilution"] = foam_data["Dilution"].fillna("00x")
        #foam_data["Date"] = foam_data["Date"].fillna("No Date")

        group_cols = ['SampleID', 'Date', 'Dilution']
        foam_pivot = foam_data.pivot_table(index=group_cols, columns='Column Name', values='Value', aggfunc='first').reset_index()
        baseline_map = parsed_df.groupby(group_cols)['Baseline'].apply(lambda x: "*" if "*" in x.astype(str).values else "").reset_index()
        start_time_map = parsed_df.groupby(group_cols)['Start Time'].apply(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().any() else "").reset_index()
    
        static_cols = ['Is_stable','Ratio', 'Oil (%)', 'HS (%)', 'Citric (%)', 'CapB (%)', 'AOS (%)', 'APG (%)',
                       'chinese HS (%)', 'LBHP (%)']
        static_info = parsed_df.groupby(group_cols)[static_cols].first().reset_index()
    
        final_df = foam_pivot.merge(baseline_map, on=group_cols, how='left')
        final_df = final_df.merge(start_time_map, on=group_cols, how='left')
        final_df = final_df.merge(static_info, on=group_cols, how='left')
        unique_ID_Multiple = df['SampleID'].nunique()
        unique_combinations = df[['SampleID', 'Dilution']].drop_duplicates()
        unique_combinations = len(unique_combinations)
        unique_combinations_date = df[['SampleID', 'Dilution', 'Date']].drop_duplicates()
        unique_combinations_date = len(unique_combinations)
        prefix_cols = ['SampleID', 'Date', 'Ratio', 'Oil (%)', 'Dilution', 'Start Time', 'Baseline']
        time_cols = sorted([col for col in final_df.columns if col.startswith("Time (")])
        all_cols = prefix_cols + time_cols + [col for col in static_cols if col not in prefix_cols]
        final_df = final_df[all_cols]
        final_df = utl.sort_time_columns_in_df(final_df)
        final_df = utl.convert_time_columns_to_float_hour(final_df)
        final_df = utl.extract_ratio_from_dilution(final_df)
        unique_ID_single = df['SampleID'].nunique()

        foam_texture_cols = [col for col in final_df.columns if "foam texture" in col.lower()]
        final_df["Foam Description"] = final_df[foam_texture_cols].astype(str).apply(
            lambda row: " | ".join([val for val in row if val.lower() != "nan"]), axis=1
        )
        #final_df.drop(columns=foam_texture_cols, inplace=True)
        phrase="Foam Texture"
        columns_to_drop = [col for col in final_df.columns if phrase.lower() in col.lower()]
        final_df = final_df.drop(columns=columns_to_drop, inplace=False)
        
        # Show output for Parsed_Yates_Oil_Processed.csv
        st.subheader("Extracted Multi Row Samples with Oil (Yates)")
        st.dataframe(df)
        st.success(f"Total unique samplesID in df: {unique_ID_Multiple}")
        st.success(f"Total unique samplesID - Dilution in df: {unique_combinations}")
        st.success(f"Total unique samplesID - Dilution - Date in df: {unique_combinations_date}")
        st.success(f"Total unique samplesID in final_df: {unique_ID_single}")
        st.success(f"Number of row in final_df: {len(final_df)}")

        # Buttons to download
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Multi Row Samples.csv",
                df.to_csv(index=False).encode("utf-8"),
                file_name="Parsed_Yates_Oil_Processed.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "Download Single Row Sample.csv",
                final_df.to_csv(index=False).encode("utf-8"),
                file_name="Parsed_Yates_Oil_Processed_Time_Single_sorted.csv",
                mime="text/csv"
            )
    
        # Search functionality
        st.subheader("Search by SampleID")
        search_id = st.text_input("Enter SampleID:")
    
        search_type = st.radio("Search type", ["Exact Match", "Contains"])
        view_option = st.radio("Choose view mode:", ["Multi Row Samples", "Single Row Samples"])
    
        if search_id:
            if search_type == "Exact Match":
                result_df_multi = df[df["SampleID"].str.lower() == search_id.lower()]
                result_df_single = final_df[final_df["SampleID"].str.lower() == search_id.lower()]
            else:  # Contains
                result_df_multi = df[df["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
                result_df_single = final_df[final_df["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
    
            if view_option == "Multi Row Samples":
                st.markdown("### Multi Row Samples")
                if result_df_multi.empty:
                    st.warning("No Multi Row Sample found for this SampleID.")
                else:
                    st.success(f"Found {len(result_df_multi)} row(s) in Multi Row Samples.")
                    st.dataframe(result_df_multi)
    
            elif view_option == "Single Row Samples":
                st.markdown("### Single Row Samples")
                if result_df_single.empty:
                    st.warning("No Single Row Sample found for this SampleID.")
                else:
                    st.success("Found matching sample in Single Row Samples.")
                    st.dataframe(result_df_single)

else:
    st.title("🔒 Access Restricted")
    st.info("Please enter the correct password to use this app.")
