
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
## Exploratory Data Analysis
**File**: KC_House_EDA_Final.ipynb

### 1. Cleaning
- The dataset originally had 21,613 rows and 21 columns.
- Removed duplicate entries.
- Dropped rows with 0 bedrooms or 0 bathrooms (considered invalid).
- No missing values found in the cleaned dataset.

**Conclusion:**
- The dataset is clean and ready for analysis.
- All rows contain valid numerical and geographical information.

---

### 2. Univariate Analysis
**Numerical features analyzed:**
- price, sqft_living, sqft_lot, bedrooms, bathrooms, floors, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15

**Distributions:**
- Most variables are right-skewed.
- price, sqft_living, sqft_lot show long right tails.
- bedrooms is mostly between 2 and 4.

**Conclusion:**
- Data is skewed, suggesting transformation may improve modeling.
- Some outliers (e.g. 33 bedrooms) are present.

---

### 3. Bivariate Analysis
**Correlation matrix plotted using heatmap.**

Top correlations with price:
- sqft_living: 0.70
- grade: 0.67
- bathrooms: 0.53
- sqft_above: 0.61
- view, waterfront also show positive influence.

**📊 Correlation Heatmap**
![Correlation Matrix](sandbox:/mnt/data/EDA-Correlation Matrix.png)

**Conclusion:**
- sqft_living, grade, bathrooms, and view are strong predictors of price.

---

### 4. Multivariate Analysis
**Selected features vs price:**
- sqft_living, grade, bathrooms, sqft_above, view, floors

**Observations:**
- Linear or non-linear relationships exist.
- sqft_living and grade have clear positive relationship with price.

**Conclusion:**
- Many predictors have a non-linear relationship with price.
- Feature engineering (e.g., log transformations) could improve performance.

---

### 5. Outlier Detection
**Boxplots created for:**
- price

**📦 Boxplot - Price**
![Boxplot - Price](sandbox:/mnt/data/EDA-Boxplot-Price.png)

**Observations:**
- Price: some listings above $3M are outliers.
- Bedrooms: up to 33 — likely unrealistic.

**Conclusion:**
- Outliers may distort regression models.
- May need removal or capping.

---

### 6. Mapping
**Geographic scatter plot using lat and long, colored by price.**

**📍 House Locations (colored by price)**
![House Locations](sandbox:/mnt/data/EDA-House locations.png)

**Observations:**
- Expensive homes cluster near water and central Seattle.
- Distinct regions can be identified spatially.

**Conclusion:**
- Location plays a critical role in housing prices.
- Spatial clustering or zip-code based modeling could enhance results.

---

### 7. Key Takeaways
- The dataset is clean and contains useful geographic and property features.
- Strong correlation exists between house price and living space, grade, and view.
- Outliers and skewed distributions may require treatment.
- Location is an important predictor and spatial modeling could be relevant.

## Feature Selection
**Notebook**: KC_House_FeatureSelection_Best.ipynb

### 1. Categorical Features
- In the KC House dataset, most features are numerical.
- No significant categorical variables are present (aside from possibly "zipcode" which is numerical but categorical by nature).
- Therefore, no encoding and filtering of categorical variables were required in this analysis.

**Conclusion:**
- Feature selection focused on numerical attributes since categorical ones were either absent or not informative.

---

### 2. Numerical Features
**Normality Check:**
We assessed the distribution of each numerical variable.

📊 **Examples of Distributions:**
- ![sqft_living15](sandbox:/mnt/data/EDA-distribution%20of%20sqft-living15.png)
- ![sqft_above](sandbox:/mnt/data/EDA-distribution%20of%20sqt-above.png)
- ![yr_renovated](sandbox:/mnt/data/EDA-distribution%20of%20yr_renovated.png)
- ![yr_built](sandbox:/mnt/data/EDA-distribution%20of%20yr-built.png)

📝 **Observation:**
- Most variables are right-skewed and not normally distributed.

**Multivariate Relationship with Price:**
- Scatterplots were plotted to assess the relationship between features and the target variable price.

📈 **Examples of Multivariate Visuals:**
- ![Price vs Bathrooms](sandbox:/mnt/data/EDA-price%20vs%20bathrooms.png)
- ![Price vs Floors](sandbox:/mnt/data/EDA-price%20vs%20floors.png)
- ![Price vs Grade](sandbox:/mnt/data/EDA-price%20vs%20grade.png)
- ![Price vs View](sandbox:/mnt/data/EDA-price%20vs%20view.png)

**Mutual Information:**
- We used mutual_info_regression() to assess the influence of numerical variables on the target price.

**Top Features (High Importance):**
- sqft_living
- grade
- sqft_above
- bathrooms
- view

**Low Importance Features:**
- yr_renovated
- long
- zipcode

---

### 3. Model-Based Selection
**Tree-Based Importance (Random Forest / Decision Tree):**
- We used DecisionTreeRegressor and RandomForestRegressor to identify the most influential features.

**Top Influential Features (Tree-Based):**
- sqft_living
- grade
- bathrooms
- view
- sqft_above

**RFE (Recursive Feature Elimination):**
- Using Linear Regression as base model, RFE selected top 10 predictors.

**RFE Selected Features:**
- sqft_living, grade, sqft_above, bathrooms, view, floors, etc.

---

### ✅ Final Conclusion
- Feature selection consistently highlights sqft_living, grade, bathrooms, and view as top predictors.
- Variables like yr_renovated and zipcode show minimal impact.
- A mix of statistical, model-based and information theory methods confirm the top predictors' influence on price.

These results guide dimensionality reduction and model optimization moving forward.



**KC House Classification

---

### Objective:
For the KC House dataset, the goal is to **predict high-priced houses** based on features such as size, location, number of rooms, renovation status, etc.

While this is a regression dataset, for classification purposes, a threshold (e.g., top 20% house prices) has been used to convert this into a binary classification task.

---

### Evaluation Metrics:

The objective of the classification model is to **identify high-priced properties** (positives), while minimizing misclassifications.

Key metrics:

- **Recall (Sensitivity)**: Measures the proportion of correctly identified high-priced homes among all actual high-priced homes.

> Recall = TP / (TP + FN)

- **Precision**: Measures the proportion of predicted high-priced homes that are actually high-priced.

> Precision = TP / (TP + FP)

- **F1 Score**: Harmonic mean of precision and recall, balancing both false positives and false negatives.

> F1 = 2 * (Precision * Recall) / (Precision + Recall)

- **Accuracy**: Proportion of total correct predictions.

- **ROC AUC Score**: Reflects the model's ability to distinguish between classes.

---

### Data Transformation:

| Column          | Transformation     | Notes                           |
|----------------|--------------------|----------------------------------|
| All numerical  | StandardScaler     | Before PCA / KNN / SVM          |
| Categorical    | OneHotEncoding     | For ensemble models             |
| All features   | PCA (10 components)| For dimensionality reduction    |

---

### Modeling Algorithms and Evaluation

#### 1. K-Nearest Neighbor (KNN)
**Notebook**: KC_House_KNN_Final.ipynb

- Accuracy: 87.0%
- Recall: 31%
- Precision: 29%
- F1 Score: 0.33
- ROC AUC Score: ~66%

**Note**: Balanced, but sensitive to scaling. Performed decently.

---

#### 2. KNN + PCA
**Notebook**: KC_House_KNN_PCA_10components.ipynb

- Accuracy: 85.9%
- Recall: 26%
- Precision: 27%
- F1 Score: 0.28

**Note**: PCA slightly reduced performance.

---

#### 3. Logistic Regression + PCA
**Notebook**: KC_House_LogReg_PCA_10components.ipynb

- Accuracy: 88.5%
- Recall: 3%
- Precision: 100%
- F1 Score: 0.07

**Note**: Extremely cautious. Almost no high-price houses predicted.

---

#### 4. SVM + PCA
**Notebook**: KC_House_SVM_PCA_10components.ipynb

- Accuracy: 89.2%
- Recall: 37%
- Precision: 34%
- F1 Score: 0.35

**Note**: Good balance between precision and recall. Scaling essential.

---

#### 5. Random Forest
**Notebook**: KC_House_RandomForest.ipynb

- Accuracy: 88.5%
- Recall: 29%
- Precision: 22%
- F1 Score: 0.23

**Note**: Decent model with moderate performance.

---

#### 6. XGBoost
**Notebook**: KC_House_XGBoost.ipynb

- Accuracy: 88.7%
- Recall: 41%
- Precision: 32%
- F1 Score: 0.36
- ROC AUC Score: 68%

**Note**: One of the best-performing models.

---

### Final Model Comparison

| Model             | Accuracy | Recall | Precision | F1 Score | ROC AUC |
|------------------|----------|--------|-----------|----------|---------|
| XGBoost           | 88.7%    | 41%    | 32%       | **0.36** | **68%** |
| SVM + PCA         | 89.2%    | 37%    | 34%       | 0.35     | 67%     |
| KNN               | 87.0%    | 31%    | 29%       | 0.33     | 66%     |
| Random Forest     | 88.5%    | 29%    | 22%       | 0.23     | 65%     |
| KNN + PCA         | 85.9%    | 26%    | 27%       | 0.28     | 63%     |
| Logistic + PCA    | 88.5%    | 3%     | 100%      | 0.07     | 51%     |

---

🎯 Project Objective
The primary goal is to identify high-priced properties using machine learning models. The original problem is a regression task, predicting price, but here we transform it into a binary classification task:

High-value properties (1): Top 20% most expensive homes

Low-value properties (0): Bottom 80%

This enables real estate professionals and investors to target premium listings more efficiently.

📏 Evaluation Metric Strategy
For businesses targeting high-end clients or planning premium investments, correctly identifying high-value properties is critical. We focus on:

Metric	Purpose
Recall	Ensure we catch most high-priced homes: TP / (TP + FN)
Precision	Ensure predicted high-priced homes are truly high-value: TP / (TP + FP)
F1 Score	Balanced trade-off between Precision and Recall
ROC AUC	Overall class separability capability
Accuracy	General correctness, though less informative in unbalanced datasets
⚙️ Data Transformers
Column Type	Transformation	Notes
Numerical	StandardScaler	Applied for algorithms like SVM & KNN
Categorical	OneHotEncoding	Applied for ensemble models
All Features	PCA (10 comps)	For dimensionality reduction experiments
🤖 Model Comparison Summary
Model	Accuracy	Recall	Precision	F1 Score	ROC AUC
XGBoost	88.7%	41%	32%	0.36	68%
SVM + PCA	89.2%	37%	34%	0.35	67%
KNN	87.0%	31%	29%	0.33	66%
Random Forest	88.5%	29%	22%	0.23	65%
KNN + PCA	85.9%	26%	27%	0.28	63%
Logistic + PCA	88.5%	3%	100%	0.07	51%
✅ Business Recommendations
Recommended Model: XGBoost (from KC_House_XGBoost.ipynb)

Why XGBoost?
Highest F1 Score and ROC AUC — balancing recall and precision.

Captures 41% of high-value homes while maintaining reasonable precision.

Suggested Use Cases:
Real estate agencies can:

Prioritize premium listings for targeted marketing

Identify properties worth renovation

Focus efforts on high-net-worth clientele

Threshold Adjustment:
Adjusting classification threshold from 0.5 to 0.4 can increase recall, identifying more high-value properties.

Ideal when missing a premium listing is costlier than mistakenly flagging a lower-value one.
---
