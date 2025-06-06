import pandas as pd
import numpy as np
import tritonclient.http as httpclient

TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "anomaly_detector_numpy"
MODEL_VERSION = "1"
CSV_FILE_PATH = "2020-03-19/S1.csv"

def test_anomaly_numpy_model(csv_path, server_url, model_name, model_version):
    df = pd.read_csv(csv_path)
    triton_client = httpclient.InferenceServerClient(url=server_url)

    # Prepare input tensors for each field
    inputs = []
    for col, dtype, triton_dtype in [
        ("timestamp", str, "BYTES"),
        ("machine_id", str, "BYTES"),
        ("failure", int, "INT32"),
        ("val1", int, "INT32"),
        ("val2", int, "INT32"),
        ("val3", int, "INT32"),
        ("val4", int, "INT32"),
        ("field7", int, "INT32"),
        ("val5", int, "INT32"),
        ("val6", int, "INT32"),
        ("val7", float, "FP32"),
    ]:
        data = df[col].astype(dtype).tolist()
        if triton_dtype == "BYTES":
            data = [str(x) for x in data]
        input_tensor = httpclient.InferInput(col, [len(df)], triton_dtype)
        input_tensor.set_data_from_numpy(
            np.array(data, dtype=object if triton_dtype == "BYTES" else dtype)
        )
        inputs.append(input_tensor)

    # Send inference request
    result = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        model_version=model_version
    )

    # Get and print anomaly indices
    anomaly_indices = result.as_numpy("anomaly_indices")
    print("Anomaly indices:", anomaly_indices)

if __name__ == "__main__":
    test_anomaly_numpy_model(CSV_FILE_PATH, TRITON_SERVER_URL, MODEL_NAME, MODEL_VERSION) 