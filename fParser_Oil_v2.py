import streamlit as st
import pandas as pd
import utilityOil_v2 as utl
from io import StringIO
import numpy as np

st.set_page_config(page_title="Foam Oil Sample Parser", layout="wide")
if utl.check_password():
    #st.title("Foam Oil Sample Parser")
    st.write("✅ Access granted! Continue with the app.")
    
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    
    if uploaded_file:
        st.success("Parsing complete!")
    
        # Step 1: Read and parse the file
        df = pd.read_csv(uploaded_file, header=None, encoding="cp1252")
        df = utl.extract_all_samples_with_corrected_stability(df)
        df['Concentrate manufacturing method (Ratio)'] = pd.NA
        df["Oil (%)"] = pd.NA
        df["Tube Volume (mL)"] = pd.NA
        df["Brine Type"] = pd.NA
        df["Start Time"] = pd.NA
        df["Temp Foam Monitoring"] = np.nan
        df["Initial Foam Temp (dilution Temp)"] = np.nan     # No logic yet, reserved
        df["Sonicated"] = np.nan  
        df[['Concentrate manufacturing method (Ratio)', "Oil (%)", "Tube Volume (mL)", 
            "Start Time", 'Dilution Ratio',"Temp Foam Monitoring","Sonicated","Brine Type"]] = df['Dilution Ratio'].apply(utl.extract_from_dilution)
        
        #df[['Concentrate manufacturing method (Ratio)', "Oil (%)", "Tube Volume (mL)", "Start Time", 'Dilution Ratio']] = df['Dilution Ratio'].apply(utl.extract_from_dilution)
        #df['Dilution Ratio'] = df['Dilution Ratio'].str.replace(r"\b[Dd]ilution\b", "", regex=True).str.strip()
        df['Dilution Ratio'] = df['Dilution Ratio'].astype(str).str.replace(r"\b[Dd]ilution\b", "", regex=True).str.strip()
        #df['Dilution Ratio'] = df['Dilution Ratio'].str.replace(r"-{2,}", " - ", regex=True).str.strip()
        #df['Dilution Ratio'] = df['Dilution Ratio'].fillna("00x")
        #df["Date"] = df["Date"].fillna("No Date")
        unique_ID_multi = df['SampleID'].nunique()
        #df = utl.extract_ratio_from_dilution(df)
        parsed_df = df.copy()
        if 'Time (min)' in parsed_df.columns:
            parsed_df['Time (min)'] = parsed_df['Time (min)'].apply(utl.convert_to_minutes)
        else:
            st.error("'Time (min)' column is missing. Please check the input format.")
        parsed_df['Dilution Ratio'] = parsed_df['Dilution Ratio'].replace("nan", pd.NA)
        parsed_df['Time (min)'] = parsed_df['Time (min)'].apply(utl.convert_to_minutes)
        parsed_df[['SampleID', 'Dilution Ratio', 'Date', 'Start Time',"Oil (%)"]] = parsed_df[['SampleID', 'Dilution Ratio', 'Date', 'Start Time',"Oil (%)"]].ffill()
        #parsed_df[['Concentrate manufacturing method (Ratio)', "Oil (%)", "Tube Volume (mL)", "Start Time", 'Dilution Ratio']] = parsed_df['Dilution Ratio'].apply(utl.extract_from_dilution)

        #parsed_df[['SampleID', 'Dilution Ratio', 'Date', 'Start Time',  'Oil (%)']] = parsed_df[['SampleID', 'Dilution Ratio', 'Date', 'Start Time', 'Oil (%)']].ffill()

        parsed_df['Time (min)'] = parsed_df['Time (min)'].astype(str)
        parsed_df= utl.make_sampleid_unique(parsed_df)
        
        #foam_data = pd.DataFrame()
        #for column, label in [('Foam Layer (cc)', 'Foam (cc)'), ('Foam Texture', 'Foam Texture')]:
        #    temp = parsed_df[['SampleID', 'Date', 'Dilution Ratio', 'Time (min)', column]].copy()
        #    temp['Column Name'] = "Time (" + temp['Time (min)'] + f") - {label}"
        #    temp = temp.rename(columns={column: "Value"})
        #    foam_data = pd.concat([foam_data, temp], ignore_index=True)
        parsed_df['Foam Layer (cc)'] = parsed_df.apply(
            lambda row: f"{row['Foam Layer (cc)']}*" if pd.notna(row['Baseline']) and '*' in str(row['Baseline']) else row['Foam Layer (cc)'],
            axis=1
        )
        foam_data = pd.DataFrame()
        for column, label in [('Foam Layer (cc)', 'Foam (cc)'), ('Foam Texture', 'Foam Texture')]:
            cols = ['SampleID', 'Date', 'Dilution Ratio', 'Time (min)', column]
            # Only add 'Foam Texture' for first pass (when checking for *)
            if column == 'Foam Layer (cc)':
                cols.append('Foam Texture')
            
            temp = parsed_df[cols].copy()

            if column == 'Foam Layer (cc)':
                temp[column] = temp.apply(
                    lambda row: f"{row[column]}*" if pd.notna(row.get('Foam Texture')) and '*' in str(row.get('Foam Texture')) else row[column],
                    axis=1
                )

            temp['Column Name'] = "Time (" + temp['Time (min)'].astype(str) + f") - {label}"
            
            # Rename measurement column to "Value"
            temp = temp.rename(columns={column: "Value"})

            # Remove 'Foam Texture' if present (only used internally)
            if 'Foam Texture' in temp.columns:
                temp.drop(columns='Foam Texture', inplace=True)

            temp = temp.reset_index(drop=True)
            
            foam_data = pd.concat(
                [foam_data, temp[['SampleID', 'Date', 'Dilution Ratio', 'Time (min)', 'Column Name', 'Value']]],
                ignore_index=True
            )
        nan_dilution_rows = parsed_df[parsed_df['Time (min)']=='nan']
        nan_dilution_rows = utl.sort_columns_custom(nan_dilution_rows)    
        #st.dataframe(nan_dilution_rows)
        group_cols = ['SampleID', 'Date', 'Dilution Ratio']
        foam_pivot = foam_data.pivot_table(index=group_cols, columns='Column Name', values='Value', aggfunc='first').reset_index()
        baseline_map = parsed_df.groupby(group_cols)['Baseline'].apply(lambda x: "*" if "*" in x.astype(str).values else "").reset_index()
        start_time_map = parsed_df.groupby(group_cols)['Start Time'].apply(lambda x: x.dropna().astype(str).iloc[0] if x.dropna().any() else "").reset_index()

        percent_cols = [col for col in parsed_df.columns if '%' in col or 'ppm' in col]
        static_cols = ['Concentrate manufacturing method (Ratio)','Concentrate Stability (8C)',
                       'Concentrate Stability (4C)','Concentrate Stability (RT)','Concentrate Stability (HT)',
                       "Temp Foam Monitoring","Brine Type","Dilution Stability"]
        static_cols = list(set(static_cols + percent_cols))
        static_info = parsed_df.groupby(group_cols)[static_cols].first().reset_index()
        final_df = foam_pivot.merge(baseline_map, on=group_cols, how='left')
        final_df = final_df.merge(start_time_map, on=group_cols, how='left')
        final_df = final_df.merge(static_info, on=group_cols, how='left')
        
        #unique_ID_Multiple = df['SampleID'].nunique()
        #unique_combinations = foam_data[['SampleID', 'Dilution Ratio']].drop_duplicates()
        #unique_combinations = len(unique_combinations)
        unique_combinations_date = foam_data[['SampleID', 'Dilution Ratio', 'Date']].drop_duplicates()
        unique_combinations_date = len(unique_combinations_date)
        prefix_cols = ['SampleID', 'Date', 'Dilution Ratio', 'Concentrate manufacturing method (Ratio)', 'Oil (%)', 'Brine Type','Start Time','Concentrate Stability (8C)',
                       'Concentrate Stability (4C)','Concentrate Stability (RT)','Concentrate Stability (HT)',"Dilution Stability", "Temp Foam Monitoring", 'Baseline']
        time_cols = sorted([col for col in final_df.columns if col.startswith("Time (")])
        all_cols =  [col for col in static_cols if col not in prefix_cols]+ prefix_cols + time_cols  
        
        final_df = final_df[all_cols]
        final_df = utl.sort_time_columns_in_df(final_df)
        final_df = utl.convert_time_columns_to_float_hour(final_df)
        foam_texture_cols = [col for col in final_df.columns if "foam texture" in col.lower()]
        final_df["Sample Description"] = final_df[foam_texture_cols].astype(str).apply(
            lambda row: " | ".join([val for val in row if val.lower() != "nan"]), axis=1
        )

        # Assign and Remove * from 'Sample Description'
        final_df['Baseline'] = final_df.apply(
            lambda row: f"{row['Baseline']}*" if pd.notna(row['Sample Description']) and '*' in str(row['Sample Description']) else row['Baseline'],
            axis=1
        )
        
        final_df['Sample Description'] = final_df['Sample Description'].astype(str).str.replace('*', '', regex=False)
        # Assign and Remove 'synthetic brine' from 'Dilution Ratio'
        final_df['Dilution Ratio'] = final_df['Dilution Ratio'].astype(str)
        final_df['Brine Type'] = final_df['Brine Type'].astype(str)
        final_df['Brine Type'] = final_df.apply(
            lambda row: f"{row['Brine Type']} synthetic brine".strip()
            if 'synthetic brine' in row['Dilution Ratio'].lower() and 'synthetic brine' not in row['Brine Type']
            else row['Brine Type'],
            axis=1
        )
        final_df['Brine Type'] = final_df['Brine Type'].astype(str).str.replace('nan', '', regex=False)
        final_df['Dilution Ratio'] = final_df['Dilution Ratio'].astype(str).str.replace('synthetic brine', '', regex=False)
        #final_df.drop(columns=foam_texture_cols, inplace=True)
        phrase="Foam Texture"
        columns_to_drop = [col for col in final_df.columns if phrase.lower() in col.lower()]
        final_df = final_df.drop(columns=columns_to_drop, inplace=False)
        
        # Show output for Parsed_Yates_Oil_Processed.csv
        st.subheader("Extracted Single Row Samples with Oil")
        st.dataframe(final_df)
        #st.success(f"Total number of extracted samples: {unique_ID_Multiple}")
        #st.success(f"Total unique samplesID - Dilution in df: {unique_combinations}")
        st.success(f"Total number of extracted samples (with unique SampleID-Dilution-Date): {unique_combinations_date}")
        #st.success(f"Total unique samplesID in final_df: {unique_ID_single}")
        #st.success(f"Number of row in final_df: {len(final_df)}")

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
                "Download Single Row Samples.csv",
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
                result_df_multi = final_df[final_df["SampleID"].str.lower() == search_id.lower()]
                result_df_single = final_df[final_df["SampleID"].str.lower() == search_id.lower()]
            else:  # Contains
                result_df_multi = final_df[final_df["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
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

