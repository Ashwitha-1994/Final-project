# Final-project
Health AI Suite is an end-to-end AI-powered healthcare analytics platform that integrates machine learning, deep learning, and interactive dashboards to support early disease detection, patient risk assessment, and clinical decision support.
Architecture of Health AI suite:
                    ┌────────────────────┐
                    │   Data Sources     │
                    │────────────────────│
                    │ • Clinical Data    │
                    │ • Medical Images   │
                    │ • Time-Series      │
                    │ • Patient Feedback │
                    └─────────┬──────────┘
                              ↓
               ┌────────────────────────────────┐
               │ Data Preprocessing & Engineering │
               │────────────────────────────────│
               │ • Cleaning & Imputation        │
               │ • Encoding & Scaling            │
               │ • Image Normalization           │
               │ • Text Preprocessing            │
               └─────────┬──────────────────────┘
                         ↓
 ┌──────────────────────────────────────────────────────────┐
 │                AI / ML MODEL LAYER                        │
 │──────────────────────────────────────────────────────────│
 │ • Risk Level Prediction (Classification)                 │
 │ • LOS Prediction (Regression)                             │
 │ • Patient Clustering (Unsupervised)                       │
 │ • Association Rule Learning                               │
 │ • CNN – Chest X-Ray Detection                             │
 │ • LSTM – Vital Signs Time-Series Analysis                 │
 │ • NLP – Sentiment Feedback Analysis                       │
 └─────────┬────────────────────────────────────────────────┘
           ↓
 ┌──────────────────────────────────────────────────────────┐
 │            Inference & Integration Layer                  │
 │──────────────────────────────────────────────────────────│
 │ • Model Loaders                                 │
 │ • Serialized Models (.pkl / .joblib / .h5)                │
 │ • Real-Time Prediction Services                           │
 └─────────┬────────────────────────────────────────────────┘
           ↓
 ┌──────────────────────────────────────────────────────────┐
 │              Visualization & UI Layer                     │
 │──────────────────────────────────────────────────────────│
 │ • Streamlit Dashboards                                    │
 │ • Prediction Outputs & Charts                             │
 │ • Risk Scores & Probabilities                             │
 │ • Interactive Analysis                                    │
 └──────────────────────────────────────────────────────────┘

