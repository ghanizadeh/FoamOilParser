import streamlit as st
import pandas as pd
import re
import numpy as np
import importlib
import utilityColloidal_v3 as utl

importlib.reload(utl) 
# Helper function to detect if a row starts a formulation
def is_formulation_row(cell):
    return "%" in cell and "dilution" not in cell.lower()

st.set_page_config(page_title="Colloidal Stability Parser", layout="wide")
#if utl.check_password():
#st.title("Foam Oil Sample Parser")
st.write("✅ Access granted! Continue with the app.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    st.success("Parsing complete!")

    # Step 1: Read and parse the file
    df_input = pd.read_csv(uploaded_file, header=None, encoding="cp1252")
    df_input = df_input.replace(r"\s+ppm", "ppm", regex=True)
    df_input = df_input.replace(r"HS+\s", "HS, ", regex=True)
    df_input = df_input.replace(r"Observations", "Observation", regex=True)
    df_input = df_input.replace(r"ppm,", "ppm CapB,", regex=True)
    #df_input['reserved_1']=np.nan


    # Main parsing logic
    records = []
    i = 0
    n = len(df_input)


    while i < n:
        row = df_input.iloc[i]
        first_cell = str(row[0]) if pd.notna(row[0]) else ""
        t=first_cell
        if is_formulation_row(first_cell):
            current_sample = utl.parse_formulation(first_cell, index=i)
            added = False
            i += 1
            
            while i < n:
                row = df_input.iloc[i]
                first_cell = str(row[0]) if pd.notna(row[0]) else ""
 
                if is_formulation_row(first_cell):
                    break  # next formulation begins
                
                # Dilution block
                if isinstance(row[0], str) and "x Dilution" in row[0]:
                    current_dilution = " - ".join(str(cell).strip() for cell in row[:11] if pd.notna(cell))
                    i += 1

                    # Header line
                    if i < n and any("Date" in str(cell) for cell in df_input.iloc[i]) and any("pH" in str(cell) for cell in df_input.iloc[i]):
                        header_row = df_input.iloc[i]
                        column_headers = [str(col).strip() for col in header_row[3:] if pd.notna(col)]
                        i += 1
                    else:
                        column_headers = []

                    # Observations
                    while i < n and isinstance(df_input.iloc[i][0], str) and df_input.iloc[i][0].startswith("Day"):
                        obs_row = df_input.iloc[i]
                        sample_record = current_sample.copy()
                        sample_record["Dilution"] = current_dilution
                        #sample_record['reserved_1']=t
                        sample_record["Day"] = obs_row[0]
                        sample_record["Date"] = obs_row[1]
                        sample_record["pH"] = obs_row[2] if pd.notna(obs_row[2]) else None
                        for idx, header in enumerate(column_headers):
                            val = obs_row[3 + idx] if (3 + idx) < len(obs_row) else None
                            sample_record[header] = val if pd.notna(val) else None
                        records.append(sample_record)
                        added = True
                        i += 1
                    continue

                i += 1

            # Add the formulation even if no dilution or stability was found
            if not added:
                records.append(current_sample)

            continue

        i += 1
    
    # Save or return final DataFrame
    final_df = pd.DataFrame(records)
    #final_df.to_csv("parsed_output.csv", index=False)
    #print("✅ Done. Results saved to 'parsed_output.csv'")     
    # Show output for Parsed_Yates_Oil_Processed.csv
    st.subheader("Extracted Single Row Samples with Oil")
    st.dataframe(final_df)
    #st.success(f"Total number of extracted samples: {unique_ID_Multiple}")
    #st.success(f"Total unique samplesID - Dilution in df: {unique_combinations}")
    #st.success(f"Total number of extracted samples (with unique SampleID-Dilution-Date): {unique_combinations_date}")
    #st.success(f"Total unique samplesID in final_df: {unique_ID_single}")
    #st.success(f"Number of row in final_df: {len(final_df)}")

    # Buttons to download
    col1= st.columns(1)[0]
    with col1:
        st.download_button(
            "Download Multi Row Samples.csv",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name="Parsed_Yates_Oil_Processed.csv",
            mime="text/csv"
        )
    #with col2:
        #st.download_button(
        #    "Download Single Row Samples.csv",
        #    final_df.to_csv(index=False).encode("utf-8"),
            #   file_name="Parsed_Yates_Oil_Processed_Time_Single_sorted.csv",
        #    mime="text/csv"
        #)

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

