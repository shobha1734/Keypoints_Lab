# Keypoints_Lab
# J.A.R.V.I.S. — Human Action Recognition & Anomaly Detection
Human action recognition and anomaly detection using MediaPipe pose keypoints + LSTM neural network with Iron Man JARVIS HUD visualization


## 📌 Project Overview

This project detects human body keypoints from video sequences using **MediaPipe Pose**,
feeds them into an **LSTM neural network**, and classifies 7 human actions —
flagging dangerous ones like **Fall Down** and **Lying Down** as anomalies.

The visualization uses an **Iron Man J.A.R.V.I.S. HUD** overlay to display
predictions, confidence scores, and anomaly alerts on video frames.

## 📊 Results

### Test Accuracy: 67%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fall Down | 0.74 | 0.79 | 0.76 | 163 |
| Lying Down | 0.72 | 0.85 | 0.78 | 130 |
| Sit Down | 0.84 | 0.43 | 0.57 | 75 |
| Sitting | 0.65 | 0.74 | 0.69 | 114 |
| Stand Up | 0.80 | 0.64 | 0.71 | 116 |
| Standing | 0.59 | 0.53 | 0.56 | 221 |
| Walking | 0.61 | 0.68 | 0.64 | 253 |


### Key Observations
- **Fall Down** detection recall of **79%** — most critical anomaly class performing well
- **Lying Down** has highest recall at **85%** — reliably detected
- **Standing vs Walking** hardest to distinguish — similar keypoint patterns
- Model trained for **43 epochs** with early stopping restoring best weights from epoch 28

