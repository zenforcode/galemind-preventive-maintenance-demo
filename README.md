
# Galemind Preventive Maintenance Demo

This repository showcases a demo implementation of preventive maintenance workflows using Galemind and Triton Inference Server.
## Overview
Preventive maintenance (PM) is essential for minimizing equipment downtime, reducing costs, and ensuring operational reliability. This demo illustrates how Galemind can be leveraged to:
    - Trigger preventive maintenance actions based on predefined rules or thresholds.
    - Use an LSTM-based machine learning model to predict potential equipment failures.
    - Serve models using NVIDIA Triton Inference Server for scalable, production-ready inference.

##  Project Structure
```bash
galemind-preventive-maintenance-demo/
├── data/                # Sample equipment and maintenance data
├── model/
│   └── machine/
│       └── 1/           # TorchScript/ONNX model for Triton
│           └── model.pt
├── config/
│   └── config.pbtxt     # Triton model config
├── docker-compose.yml   # Triton service definition
├── main.py              # Data generation and inference workflow
└── README.md

```
# Create virtual enviroment
```bash
uv venv create
source .venv/bin/activate
uv sync
```
# Generate sample data
```bash
python main.py
```
# Run inference Server
Ensure your model is saved in the expected Triton format (TorchScript/ONNX) under model/machine/1/ and that config.pbtxt is present in model/machine/.

Then start server using Docker Compose:
docker-compose up

The server will be accessible at:
- HTTP: http://localhost:8001/v2/models/machine/infer
- gRPC: localhost:8000

