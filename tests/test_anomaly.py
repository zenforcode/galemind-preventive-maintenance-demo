import unittest
import pandas as pd
from src.model.anomaly import SensorData, detect_anomalies
from datetime import datetime, timedelta

class TestAnomalyDetection(unittest.TestCase):

    def test_detect_anomalies_no_anomalies(self):
        # Create synthetic data with no anomalies
        # Generating 30 days of hourly data with constant values
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        data = []
        constant_values = {
            'val1': 100,
            'val2': 200,
            'val3': 300,
            'val4': 400,
            'field7': 500,
            'val5': 600,
            'val6': 700,
            'val7': 80.0,
        }
        for i in range(30 * 24): # 30 days * 24 hours
            timestamp = start_time + timedelta(hours=i)

            sensor = SensorData(
                timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'), # Convert to string format expected by SensorData
                machine_id='machine1',
                failure=0,
                val1=constant_values['val1'],
                val2=constant_values['val2'],
                val3=constant_values['val3'],
                val4=constant_values['val4'],
                field7=constant_values['field7'],
                val5=constant_values['val5'],
                val6=constant_values['val6'],
                val7=constant_values['val7'],
            )
            data.append(sensor)

        anomalies = detect_anomalies(data)
        self.assertTrue(anomalies.empty, "Should not detect anomalies in normal data")

    def test_detect_anomalies_with_csv_data(self):
        # Use data from the provided CSV file
        csv_path = "2020-03-19/S1.csv"
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            self.fail(f"Test data file not found at {csv_path}")

        # Convert DataFrame rows to SensorData objects
        data = []
        for index, row in df.iterrows():
            sensor = SensorData(
                timestamp=row['timestamp'],
                machine_id=row['machine_id'],
                failure=int(row['failure']),
                val1=int(row['val1']),
                val2=int(row['val2']),
                val3=int(row['val3']),
                val4=int(row['val4']),
                field7=int(row['field7']),
                val5=int(row['val5']),
                val6=int(row['val6']),
                val7=float(row['val7']),
            )
            data.append(sensor)

        # Get expected anomalies from the CSV data (where 'failure' is 1)
        expected_anomalies_timestamps = set(df[df['failure'] == 1]['timestamp'].tolist())

        # Run anomaly detection
        detected_anomalies_df = detect_anomalies(data)

        # Extract timestamps of detected anomalies
        # Ensure the 'ds' column exists and is in the correct format
        if not 'ds' in detected_anomalies_df.columns:
             self.fail("Expected 'ds' column not found in detected anomalies DataFrame.")

        # Convert detected anomaly timestamps to the same format as in the CSV for comparison
        # Assuming 'ds' is datetime objects from Prophet, convert to string
        detected_anomalies_timestamps = set(detected_anomalies_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist())

        # Assert that all expected anomalies are among the detected anomalies
        # Note: Anomaly detection might find more anomalies than marked in the CSV
        # A perfect match is unlikely due to algorithm sensitivity.
        # We will check if all *known* anomalies from the CSV are *present* in the detected set.
        # This is a basic check; a more robust test might assess overlap percentage.

        missing_anomalies = expected_anomalies_timestamps - detected_anomalies_timestamps
        self.assertTrue(
            len(missing_anomalies) == 0,
            f"Not all expected anomalies were detected. Missing timestamps: {missing_anomalies}"
        )

        # Optional: Add a check for false positives, but this is more complex and depends on acceptable thresholds
        # For now, we focus on ensuring known anomalies are caught.

    # Add more test methods here for different scenarios

if __name__ == '__main__':
    unittest.main() 