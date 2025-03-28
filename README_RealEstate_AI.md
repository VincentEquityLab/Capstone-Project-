
# 🏠 RealEstate_AI

**Capstone Project – Berkeley HAAS (ML & AI Specialization)**

---

## ❓ Problem Statement

This project aims to assist real estate investors in King County (Seattle area) in identifying the most profitable properties by:

- 📈 Predicting house prices using Machine Learning (ML) and Deep Learning (DL) models  
- 🧠 Classifying investment potential (above/below median)  
- 📊 Detecting high-yield property clusters

---

## 📊 Dataset Overview

- **Source:** Kaggle - King County House Sales  
- **Size:** 21,613 rows × 21 columns  
- **Target Variable:** `price`  

**Key Features:**  
`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `grade`, `yr_built`, `zipcode`, `lat`, `long`

**Derived Features:**  
`gross_yield_%`, `estimated_rent`, `year_sold`, `month_sold`

---

## 🔍 Exploratory Data Analysis (EDA)

### 1. Distribution of House Prices
![Distribution of House Prices](images/price_hist.png)

### 2. Boxplot by Bedrooms
![Boxplot by Bedrooms](images/box_bedrooms_price.png)

### 3. Average Price Over Time
![Average Price Over Time](images/avg_price_year.png)

### 4. Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

---

## 🔄 Clustering Analysis (PCA + KMeans)

Using standardized features and PCA for 2D visualization:

- **Number of Clusters:** 4  
- **Key Insight:** Cluster 2 yields highest rental return (~9%)

### PCA Cluster Visualization
![PCA Cluster Visualization](images/pca_clusters.png)

---

## 🧠 Regression Modeling

We trained five regression models:

| Model              | RMSE (USD) | R² Score |
|-------------------|------------|----------|
| Linear Regression | $205,000   | 0.70     |
| Decision Tree     | $180,000   | 0.75     |
| Random Forest     | $130,000   | 0.88 ✅ |
| KNN Regression    | $220,000   | 0.65     |
| Deep Neural Net   | $160,000   | 0.80     |

### RF Actual vs Predicted
![Actual vs Predicted Prices (Random Forest)](images/actual_vs_rf.png)

---

## 🔐 Classification: Logistic Regression

- **Label:** 1 if price > median, 0 otherwise  
- **Accuracy:** 85%  
- **Precision/Recall/F1:** Balanced around 84–86%

**Confusion Matrix Summary:**  
- True Positives: 1,842  
- True Negatives: 1,789  
- False Positives: 356  
- False Negatives: 301  

---

## 🎯 Strategic Recommendations

### 👨‍💼 For Investors
- Focus on clusters with high yield > 8%
- Prioritize undervalued zipcodes with high rent potential

### 🏘️ For Homeowners
- Choose renovated or newer homes in suburban zones

### 🏙️ For Urban Planners
- Develop clusters with low yield but high population growth
- Improve accessibility to high-yield suburban zones

---

## 🧠 Key Takeaways

- 📌 **Random Forest** and **DNN** are top predictors of price  
- 📌 Logistic regression classifies high/low-value homes with strong accuracy  
- 📌 Price is primarily driven by `sqft_living`, `grade`, `location`

---

## 🔧 Future Work

- Build a Streamlit dashboard for price prediction  
- Add renovation cost estimation  
- Use geospatial analysis (folium, geopandas)  
- Integrate real-time data (Zillow/Redfin APIs)  
- Try advanced models: LightGBM, XGBoost, Stacking

---

## 🗂️ Repository Structure

```
RealEstate_AI/
│
├── data/
│   └── kc_house_data.csv
│
├── images/
│   ├── price_hist.png
│   ├── box_bedrooms_price.png
│   ├── avg_price_year.png
│   ├── correlation_heatmap.png
│   ├── pca_clusters.png
│   └── actual_vs_rf.png
│
└── README.md
```
