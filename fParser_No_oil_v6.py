import streamlit as st
import numpy as np
import pandas as pd
import utilityNoOil_v6 as utl
from io import StringIO

st.set_page_config(page_title="Foam Parser - No Oil", layout="wide")
if utl.check_password():
    #st.title("Foam Oil Sample Parser")
    st.write("✅ Access granted! Continue with the app.")
    
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    
    if uploaded_file:
        st.success("Parsing complete!")
    
        # Step 1: Read and parse the file
        df_input = pd.read_csv(uploaded_file, header=None, encoding="cp1252")
        samples, formulations = utl.extract_samples_complete_fixed(df_input)
        df_samples = pd.DataFrame(samples)
        df_formulations = pd.DataFrame.from_dict(formulations, orient="index")
        df_formulations["SampleID"] = df_formulations.index
        df_multiple = df_samples.merge(df_formulations, on="SampleID", how="left")
        #df_multiple.to_csv("noProcess.csv")
        # Create the new columns with default NaN
        df_multiple["Initial Foam Volume (cc)"] = "5cc"  # Set default value
        df_multiple["Pilot"] = np.nan
        df_multiple["Temp Foam Monitoring"] = np.nan
        df_multiple["Initial Foam Temp (dilution Temp)"] = np.nan     # No logic yet, reserved
        df_multiple["Water (cc)"] = np.nan   
        df_multiple["Sonicated"] = np.nan  
        df_multiple["Brine Type"] = np.nan 
        df_multiple["Sample Description"] = pd.NA
    

        df_multiple_no_process = df_multiple
        #df_multiple_no_process = df_multiple_no_process.drop(columns=["Initial Foam Volume (cc)", "Pilot", "Water (cc)", "Tube Volume (mL)"])
        rows_to_move = df_multiple_no_process[df_multiple_no_process["Dilution Ratio"].isna()]
        #df_multiple_no_process.to_csv("1_Parser_No_Oil_No_Process.csv")
        # Apply the processing


        #df_multiple[["Pilot", "Temps Foam Monitoring", "Initial Foam Volume (cc)", "Dilution Ratio", "Concentrate manufacturing method (Ratio)", "Sonicated"]] = df_multiple.apply(
        #    lambda row: pd.Series(utl.process_dilution(row["Dilution Ratio"])),
        #    axis=1
        #)

        df_multiple[["Pilot", "Temp Foam Monitoring", "Initial Foam Volume (cc)", "Dilution Ratio", "Concentrate manufacturing method (Ratio)", 
                     "Sonicated","Sample Description","Date"]] = df_multiple.apply(
            lambda row: pd.Series(utl.process_dilution(row["Dilution Ratio"], row["Date"])),
            axis=1
        )

        df_multiple["Date"] = df_multiple["Date"].fillna("No Date")
        df_multiple["Dilution Ratio"] = df_multiple["Dilution Ratio"].fillna("00X")
        df_multiple = utl.make_sampleid_unique(df_multiple)
        df_multiple["Tube Volume (mL)"] = df_multiple["Tube Volume (mL)"].astype(str).str.replace(r"mL\s*tube", "", case=False, regex=True).str.strip()
        df_single = utl.assign_pilot_column(df_multiple)
        #df_single = df_single.drop_duplicates()
        df_single = df_single.replace({None: np.nan}).infer_objects(copy=False)
        df_single_final = utl.clean_dilution(df_single)
        df_single_final = pd.DataFrame(df_single_final)

        # remove
        #final_df = utilityNoOil.update_sampleid_with_sonicated_status(output_rows)
        # Create DataFrame

        # Add Samples without Dilution from Multiple Format
        df_single_final = pd.concat([df_single_final, rows_to_move], ignore_index=True)
        # Drop unwanted cols
        df_multiple_final = df_multiple.drop(columns=["Initial Foam Volume (cc)", "Pilot", "Water (cc)", "Tube Volume (mL)"])
        df_single_final = df_single_final.drop(columns=["Initial Foam Volume (cc)", "Pilot", "Water (cc)", "Tube Volume (mL)", "Day", "Foam (cc)", "Foam Texture"])
        #df_single_final.to_csv("2_Parser_No_Oil_No_Process.csv")
        df_single_final = utl.clean_dilution_ratio(df_single_final)
        df_single_final['Sample Description'] = df_single_final['Sample Description'].str.replace(r'\b\d{2,3}X\b', '', regex=True).str.strip()

        # Sort Colmuns
        #df_multiple_final= utl.sort_columns_custom(df_multiple_final)
        df_single_final = utl.sort_columns_custom(df_single_final)
        # Show output for Parsed_Yates_Oil_Processed.csv
        st.subheader("All Samples (No Oil)")
        st.dataframe(df_single_final)
        st.subheader("Samples with half life")
        df_half_life = utl.extract_half_life_samples(df_single_final)
        st.dataframe(df_half_life)
        
        st.success(f"Total number of extracted samples: {len(df_single_final)}")
        st.success(f"Total number of extracted samples (half-life): {len(df_half_life)}")

        #unique_combinations = df_single_final[['SampleID', 'Dilution Ratio']].drop_duplicates()
        #total_unique_combinations = unique_combinations.shape[0]
        #st.success(f"Total unique samplesID - Dilution in df: {total_unique_combinations}")
        #unique_combinations_date = df_single_final[['SampleID', 'Dilution Ratio', 'Date']].drop_duplicates()
        #total_unique_combinations_date = unique_combinations_date.shape[0]
        #st.success(f"Total number of extracted samples (with unique SampleID-Dilution-Date): {total_unique_combinations_date}")
        #st.success(f"Total unique samplesID in final_df: {unique_ID_single}")
        #st.success(f"Number of row in final_df: {len(final_df)}")

        # Buttons to download
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Multi Row Samples.csv",
                df_multiple_final.to_csv(index=False).encode("utf-8"),
                file_name="1_Parser_No_Oil_No_Process.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "Download Single Row Samples.csv",
                df_single_final.to_csv(index=False).encode("utf-8"),
                file_name="2_Parser_No_Oil_Single.csv",
                mime="text/csv"
            )
    
        # Search functionality
        st.subheader("Search by SampleID")
        search_id = st.text_input("Enter SampleID:")
        search_type = st.radio("Search type", ["Exact Match", "Contains"])
        view_option = st.radio("Choose view mode:", ["Multi Row Samples", "Single Row Samples"])
        if search_id:
            if search_type == "Exact Match":
                result_df_multi = df_multiple_final[df_multiple_final["SampleID"].str.lower() == search_id.lower()]
                result_df_single = df_single_final[df_single_final["SampleID"].str.lower() == search_id.lower()]
            else:  # Contains
                result_df_multi = df_multiple_final[df_multiple_final["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
                result_df_single = df_single_final[df_single_final["SampleID"].str.lower().str.contains(search_id.lower(), na=False)]
                
            if view_option == "Multi Row Samples":
                st.markdown("### Multi Row Samples")
                if result_df_multi.empty:
                    st.warning("No Multi Row Sample found for this SampleID.")
                else:
                    st.success(f"Found {len(result_df_multi)} row(s) in Multi Row Samples.")
                    st.dataframe(result_df_multi, use_container_width=True)
            elif view_option == "Single Row Samples":
                st.markdown("### Single Row Samples")
                if result_df_single.empty:
                    st.warning("No Single Row Sample found for this SampleID.")
                else:
                    st.success("Found matching sample in Single Row Samples.")
                    st.dataframe(result_df_single, use_container_width=True)

else:
    st.title("🔒 Access Restricted")
    st.info("Please enter the correct password to use this app.")
