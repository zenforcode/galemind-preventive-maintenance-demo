import json
import numpy as np
import triton_python_backend_utils as pb_utils
from src.model.anomaly import SensorData, detect_anomalies

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the input JSON string
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_json = input_tensor.as_numpy()[0].decode("utf-8")
            records = json.loads(input_json)
            # Convert to SensorData
            sensor_data_list = [SensorData(**row) for row in records]
            # Run anomaly detection
            anomalies_df = detect_anomalies(sensor_data_list)
            # Convert anomalies to JSON
            anomalies_json = anomalies_df.to_json(orient="records", date_format="iso")
            output_tensor = pb_utils.Tensor("OUTPUT", np.array([anomalies_json.encode("utf-8")], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses 