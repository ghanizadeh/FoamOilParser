import streamlit as st
import pandas as pd
import re
import numpy as np
import importlib
import utilityColloidal_v3 as utl

importlib.reload(utl) 


st.set_page_config(page_title="Colloidal Stability Parser", layout="wide")
#if utl.check_password():
st.title("Colloidal Stability Parser")
#st.write("✅ Access granted! Continue with the app.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    #st.success("Parsing complete!")
    # Step 1: Read and parse the file
    df_input = pd.read_csv(uploaded_file, header=None, encoding="cp1252")
    # make new columns
    df_input['Tempreture'] = pd.NA
    df_input['DEnsity (RT)'] = pd.NA
    df_input['Discontinuation of dilutioin stability monitoring'] = pd.NA
    df_input['Concentrate manufacturing method (Ratio)'] = pd.NA
    df_input['Sonicated'] = pd.NA
    df_input['pH_Concentrate'] = pd.NA
    df_input['Brine Type'] = pd.NA
    df_input['Concentrate Stability (4C)'] = pd.NA
    df_input['Concentrate Stability (8C)'] = pd.NA
    df_input['Concentrate Stability (RT)'] = pd.NA
    df_input['Concentrate Stability (HT)'] = pd.NA
    df_input['Date of manufacturing concentrate to make dilution']=pd.NA
    df_input['Dilution Stability'] = pd.NA
    df_input['Sample Description'] = pd.NA
    df_input = utl.replace_text_in_df(df_input)
    final_df = utl.main_parse(df_input)
 

    # ---------- Extract Samples with NAN Dilutio (Organize SampleID & Sample Description) ----------
    df_noDilu = final_df[final_df['Dilution Ratio'].isna()]
    df_noDilu['Sample Description'] = df_noDilu['SampleID'].apply(lambda x: x.split(")", 1)[1].strip() if isinstance(x, str) and ")" in x else "") # For Sample Description, get text after the first ")" in SampleId
    df_noDilu['SampleID'] = df_noDilu['SampleID'].apply(lambda x: x.split(")", 1)[0] if isinstance(x, str) and ")" in x else x) # For SampleID, keep text up to and including the first ")" 

    # ---------- Organize Dilution Ratio & Sample Description) ----------
    final_df['Dilution Ratio'] = final_df.apply(lambda row: str(row['Dilution Ratio']) + ' ' + row['SampleID'].split(")", 1)[1].strip() if ")" in row['SampleID'] else row['Dilution Ratio'],
        axis=1) # For Dilution Ratio, get text after the first ")" in SampleID

    # ---------- Parse Dilution ----------
    final_df[['Sample Description', 'Tempreture', 'Discontinuation of dilutioin stability monitoring', 
    'Concentrate manufacturing method (Ratio)', 'Concentrate Stability (4C)','Concentrate Stability (8C)',
    'Concentrate Stability (RT)', 'Concentrate Stability (HT)', 'Dilution Stability', 'pH_Concentrate',
    'Brine Type','Sonicated','Date of manufacturing concentrate to make dilution']] = final_df['Dilution Ratio'].apply(utl.extract_from_dilution)
     
    final_df=utl.recompute_days_from_day0(final_df)
    final_df=utl.remove_observation(final_df)

    # ---------- Keep only everything before and including the first ")" in SampleID in Final_df ----------
    final_df['SampleID'] = final_df['SampleID'].apply(lambda x: x.split(")", 1)[0].strip() if ")" in x else x)
     
    # ----------  Merge NO Dilu Sample with the others ----------
    common_cols = final_df.columns.intersection(df_noDilu.columns)
    final_df = pd.concat([final_df, df_noDilu[common_cols]], ignore_index=True)    
    noSort_final_df = final_df 
    reshape_df = utl.reshape_df(final_df)
    st.subheader("Extracted Samples (Colloidal Stability)")

    
    #st.success(f"Total number of extracted samples: {unique_ID_Multiple}")
    unique_combinations = final_df[['SampleID', 'Dilution Ratio']].drop_duplicates().shape[0]
    unique_combinations = unique_combinations 
    st.success(f"Total number of extracted samples: {unique_combinations}")
    #df_day0 = final_df[final_df['Day'].astype(str).str.strip().str.lower() == 'day 0']
    #unique_combinations_date = df_day0[['SampleID', 'Dilution Ratio', 'Date']].drop_duplicates().shape[0]
    #unique_combinations_date = unique_combinations_date + len(df_noDilu)
    #st.success(f"Total number of extracted samples (with unique SampleID-Dilution-Date Day 0): {unique_combinations_date}")
    #st.success(f"Number of samples in reshape_df: {len(reshape_df)}")
 




    # ---------- Only Keep Dilution pattern in Dilution Ratio ----------
    final_df['Dilution Ratio'] = final_df['Dilution Ratio'].apply(lambda x: utl.only_keep_dilution(x) if pd.notna(x) else x)  
    reshape_df['Dilution Ratio'] = reshape_df['Dilution Ratio'].apply(lambda x: utl.only_keep_dilution(x) if pd.notna(x) else x) 
    
    # ---------- Remove Dilution pattern in Sample Description ----------
    #final_df['Sample Description'] = final_df['Sample Description'].apply(lambda x: utl.remove_dilution_pattern(x) if pd.notna(x) else x) 
    #reshape_df['Sample Description'] = reshape_df['Sample Description'].apply(lambda x: utl.remove_dilution_pattern(x) if pd.notna(x) else x)   

    # ---------- Parse Sample Description for sampple perfomance (including NO DILUTION) ----------
    reshape_df [['Concentrate Stability (4C)','Concentrate Stability (8C)',
    'Concentrate Stability (RT)', 'Concentrate Stability (HT)', 'Dilution Stability']] = reshape_df['Sample Description'].apply(utl.parse_SampleDescription)


    # ---------- Sort Columns ----------
    final_df = utl.sort_columns_custom(final_df)
    reshape_df = utl.sort_columns_custom(reshape_df)

    # ---------- Show Data ----------
    st.dataframe(reshape_df)






    # Buttons to download
    col1, col2 = st.columns(2) 

    with col1:
        st.download_button(
            "Download Multi Row Samples.csv",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="Parsed_Yates_Oil_Processed.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download Single Row Samples.csv",
            reshape_df.to_csv(index=False).encode("utf-8"),
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
            result_df_multi = noSort_final_df[noSort_final_df["SampleID"].str.lower() == search_id.lower()]
            result_df_single = reshape_df[reshape_df["SampleID"].str.lower() == search_id.lower()]
        else:  # Contains
            result_df_multi = noSort_final_df[noSort_final_df["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
            result_df_single = reshape_df[reshape_df["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]

        if view_option == "Multi Row Samples":
            st.markdown("### Multi Row Samples")
            if result_df_single.empty:
                st.warning("No Multi Row Sample found for this SampleID.")
            else:
                st.success("Found matching sample in Multi Row Samples.")
                st.dataframe(result_df_multi)

        if view_option == "Single Row Samples":
            st.markdown("### Single Row Samples")
            if result_df_single.empty:
                st.warning("No Single Row Sample found for this SampleID.")
            else:
                st.success("Found matching sample in Single Row Samples.")
                st.dataframe(result_df_single)
#else:
#    st.title("🔒 Access Restricted")
#    st.info("Please enter the correct password to use this app.")

