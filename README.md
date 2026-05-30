MedPredict: Generative AI Clinical Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-Optimized-brightgreen.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)

Watch the Demo:
https://github.com/user-attachments/assets/f02d461a-be62-4796-ad11-13e7b373fcdd


Project Overview
MedPredict is an end-to-end Machine Learning Operations (MLOps) architecture designed to solve healthcare data scarcity and deliver real-time diagnostic predictions. 

Instead of relying on limited raw medical data, this system utilizes a **Generative Adversarial Network (CTGAN)** to synthesize privacy-preserving, statistically accurate patient records. Those records are then used to train an optimized **LightGBM classifier**, which is deployed as a containerized RESTful API microservice.

Architecture & Tech Stack
1. **Generative Data Engine (`CTGAN`):** Analyzes raw clinical seed data and hallucinates 1,000+ synthetic patient profiles, retaining feature distributions without exposing real Protected Health Information (PHI).
2. **Predictive Modeling (`LightGBM` & `Scikit-Learn`):** A gradient boosting framework trained on the synthetic dataset, utilizing `GridSearchCV` for hyperparameter tuning.
3. **Inference Server (`FastAPI` & `Uvicorn`):** A high-performance, asynchronous web server that receives patient profiles and returns probability distributions for cardiovascular risks.
4. **Containerization (`Docker`):** The entire pipeline is isolated into reproducible Docker containers to ensure environment consistency.
5. **Interactive Frontend (`Streamlit`):** A clean, Python-based UI allowing clinicians to input demographics and receive instant diagnostic inferences.

 How to Run Locally

1. Build and Run the Inference API (Backend)
Navigate to the predictor service and spin up the Docker container:
```bash
cd synthedata-ops/services/predictor
docker build -t predictor-service .
docker run -d -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" --name clinical-api predictor-service
