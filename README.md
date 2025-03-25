
Capstone Project - Real Estate Analysis
# 🏘️ Real Estate Investment Analysis with ML & AI  
**Capstone Project – Deployment of ML and AI (Module 24.1)**

---

## 📌 Project Summary

This project focuses on the analysis and optimization of a real estate portfolio using machine learning (ML) and artificial intelligence (AI) techniques.  
The dataset includes more than **21,000 residential property sales** from **King County, USA (Seattle area)**.

🎯 **Goal**: Help investors make data-driven decisions about property acquisition and management by identifying the most profitable properties and predicting future performance.

---

## 💡 Problem Statement

Investors often struggle to evaluate which real estate properties offer the best return on investment.  
Manually comparing location, price, size, and potential rent is time-consuming and error-prone.

**Objective**:
- Analyze historical housing data  
- Predict property prices using ML models  
- Classify investment quality  
- Detect high-yield clusters  
- Recommend top-performing assets

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `kc_house_data.csv` | Raw dataset (from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)) |
| `Real_Estate_Analysis.ipynb` | Main notebook: EDA, clustering, ML, deep learning |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VincentEquityLab/Capstone-Project-/blob/main/Capstone_Project_Real_Estate_Analysis_ipynb.ipynb)


---

## 📊 Key Results

### 🧠 Machine Learning Models

| Model                          | Metric     | Result     |
|-------------------------------|------------|------------|
| Linear Regression             | RMSE       | ~$171,000  |
| Random Forest Regressor       | RMSE       | ~$103,000  |
| Deep Neural Network (DNN)     | RMSE       | ~$95,000   |

### 💬 Notable Findings

- 📈 Best gross yield: **Over 14%** in certain zipcodes  
- 🏡 Spacious properties under $500K exist in suburban clusters  
- 🧱 Recently built homes can offer **>8% yield**  
- 🧭 **4 investor profiles** identified via clustering

---

## 🔍 Next Steps

- Integrate real-time market data (via APIs)
- Incorporate rental market dynamics to calculate **net yield**
- Deploy as a **Streamlit app** for interactive property exploration
- Include **renovation cost estimates** for ROI optimization

---
