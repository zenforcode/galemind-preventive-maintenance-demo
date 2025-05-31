import pandas as pd
import json
import tritonclient.http as httpclient
import numpy as np # Needed for TYPE_STRING/TYPE_BYTES in tritonclient
from datetime import datetime # Assuming timestamp might need parsing

# --- Configuration ---
TRITON_SERVER_URL = "localhost:8000" # Use 8001 for gRPC
MODEL_NAME = "anomaly_detector_json"
MODEL_VERSION = "1"
CSV_FILE_PATH = "2020-03-19/S1.csv" # <--- **Updated CSV file path**

# --- Main Script ---
def test_anomaly_json_model(csv_path, server_url, model_name, model_version):
    try:
        # 1. Read CSV data using pandas
        print(f"Reading data from {csv_path}...")
        original_df = pd.read_csv(csv_path)

        # Ensure timestamp column exists
        if 'timestamp' not in original_df.columns:
             print("Error: 'timestamp' column not found in CSV.")
             return

        # Try parsing timestamp column to datetime objects
        try:
            original_df['timestamp_dt'] = pd.to_datetime(original_df['timestamp'])
        except Exception as e:
            print(f"Warning: Could not parse timestamp column as datetime: {e}. Proceeding with original values for merge.")
            original_df['timestamp_dt'] = original_df['timestamp'] # Use original if parsing fails

        # Convert the timestamp column to ISO format strings *before* creating records for JSON
        original_df['timestamp_str'] = original_df['timestamp_dt'].apply(lambda x: x.isoformat() if isinstance(x, datetime) else str(x))

        # Convert pandas DataFrame to a list of dictionaries (records) for JSON input
        # Use the column with string timestamps for the JSON payload
        records_df = original_df.copy()
        records_df['timestamp'] = records_df['timestamp_str'] # Replace original timestamp with string version for JSON
        records_df = records_df.drop(columns=['timestamp_dt', 'timestamp_str']) # Drop temporary columns

        records = records_df.to_dict(orient='records')

        input_json_string = json.dumps(records)
        input_data = np.array([input_json_string.encode('utf-8')], dtype=object)

        # 2. Create Triton HTTP client
        triton_client = httpclient.InferenceServerClient(url=server_url)

        # 3. Prepare input tensor
        input_tensor = httpclient.InferInput(
            "INPUT",
            input_data.shape,
            "BYTES"
        )
        input_tensor.set_data_from_numpy(input_data)

        # 4. Send inference request
        print(f"Sending inference request to model {model_name} version {model_version}...")
        result = triton_client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            model_version=model_version
        )

        # 5. Process the response and merge with original data
        output_tensor = result.as_numpy("OUTPUT")

        if output_tensor is not None and output_tensor.size > 0:
            output_json_string = output_tensor[0].decode('utf-8')
            anomalies_list = json.loads(output_json_string)

            # Convert anomaly results to a DataFrame
            anomalies_df = pd.DataFrame(anomalies_list)

            if not anomalies_df.empty:
                 # Ensure 'ds' column in anomalies_df is datetime for merging if original was parsed
                 # Use the timestamp_dt column from the original_df for merging
                 if original_df['timestamp_dt'].dtype == 'datetime64[ns]':
                      anomalies_df['ds_dt'] = pd.to_datetime(anomalies_df['ds'], errors='coerce')
                      merge_on = 'ds_dt'
                      left_on = 'timestamp_dt'
                 else:
                      merge_on = 'ds'
                      left_on = 'timestamp'

                 # Merge original data with anomaly results
                 merged_df = pd.merge(
                     original_df,
                     anomalies_df[[merge_on, 'anomaly']],
                     left_on=left_on,
                     right_on=merge_on,
                     how='left'
                 )
                 # Fill NaN in 'anomaly' column with False (meaning not detected as anomaly)
                 merged_df['anomaly'] = merged_df['anomaly'].fillna(False)

                 # Clean up temporary datetime and string columns
                 merged_df = merged_df.drop(columns=['timestamp_dt', 'timestamp_str', merge_on], errors='ignore')

                 print("\nAnomaly Status for all data points:")
                 # Display the relevant columns: timestamp, original_value (val1), and anomaly flag
                 print(merged_df[['timestamp', 'val1', 'anomaly']].to_string())


            else:
                 print("\nNo anomalies were detected by the model.")
                 # Display all original data with anomaly=False
                 original_df['anomaly'] = False
                 merged_df = original_df.drop(columns=['timestamp_dt', 'timestamp_str'], errors='ignore')
                 print("\nAnomaly Status for all data points:")
                 print(merged_df[['timestamp', 'val1', 'anomaly']].to_string())


        else:
             print("\nNo anomalies detected or received empty response from Triton.")
             # Display all original data with anomaly=False
             original_df['anomaly'] = False
             merged_df = original_df.drop(columns=['timestamp_dt', 'timestamp_str'], errors='ignore')
             print("\nAnomaly Status for all data points:")
             print(merged_df[['timestamp', 'val1', 'anomaly']].to_string())


    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the test ---
if __name__ == "__main__":
    test_anomaly_json_model(CSV_FILE_PATH, TRITON_SERVER_URL, MODEL_NAME, MODEL_VERSION) 