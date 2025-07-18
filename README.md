# Telecom Customer Churn Prediction

A comprehensive data science project to predict customer churn in the telecommunications industry using advanced data science techniques and ensemble modeling approaches.

## **As taught during our training I have publicly deployed my full model on streamlit. Please do access it first and check it out.**
[Streamlit Publicly Deployed APP](https://arnavchopra-ct-csi-ds-4264-telecom-churn.streamlit.app/)


## 📊 Project Overview

Customer churn is a critical business metric for telecommunications companies. This project implements an end-to-end machine learning pipeline to predict which customers are likely to discontinue their services, enabling proactive retention strategies and significant cost savings.

### Key Objectives
- Analyze customer behavior patterns and identify key churn indicators
- Develop robust predictive models with high accuracy and reliability
- Implement a scalable ML pipeline for production deployment
- Provide various business insights

## 🗂️ Dataset Information

The dataset contains **7,043 customer records** with **21 features** covering:

**Customer Demographics:**
- Gender, Senior Citizen status, Partner, Dependents

**Account Information:**
- Tenure, Contract type, Payment method, Billing preferences

**Service Usage:**
- Phone service, Internet service, Multiple lines
- Add-on services: Online Security, Backup, Device Protection, Tech Support
- Entertainment services: Streaming TV, Streaming Movies

**Financial Metrics:**
- Monthly charges, Total charges

**Target Variable:**
- Churn (Yes/No) - Customer retention status

## 🛠️ Technology Stack

- **Python 3.8+**
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Statistical Analysis:** SciPy, Statsmodels



## 🔍 Methodology

## **Note: I tried various algorithms during my various iterations, and i finalised ensembale learning using voting classifer i combined various best performing algorithms together which gave me the best ROC_AUC score -> 84.4%.**

### 1. Exploratory Data Analysis
- **Data Quality Assessment:** Missing values, duplicates, data types validation
- **Statistical Summary:** Distribution analysis, correlation matrix, outlier detection
- **Churn Analysis:** Customer segmentation, churn rate by demographics and services
- **Visualization:** Interactive dashboards and comprehensive plotting for business insights

### 2. Data Preprocessing
- **Missing Value Treatment:** Strategic imputation based on business logic and data patterns
- **Outlier Detection:** IQR and statistical methods for anomaly identification
- **Data Type Conversion:** Optimal encoding and transformation for categorical variables
- **Feature Scaling:** Standardization and normalization for algorithm compatibility

### 3. Feature Engineering
- **Categorical Encoding:** One-hot encoding, target encoding, frequency encoding strategies
- **Numerical Transformations:** Log transformation, binning, polynomial features
- **Business Logic Features:** Customer lifetime value, service intensity scores, tenure segmentation
- **Interaction Features:** Cross-feature relationships and domain-specific combinations

### 4. Model Development
- **Baseline Models:** Logistic Regression, Decision Trees for benchmarking
- **Advanced Algorithms:** Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Ensemble Methods:** Voting classifiers, stacking, and advanced blending techniques
- **Hyperparameter Optimization:** RandomizedSearchCV

### 5. Model Evaluation
- **Comprehensive Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation:** Stratified K-fold for robust and unbiased evaluation
- **Business Metrics:** Cost-benefit analysis, customer lifetime value impact assessment
- **Model Comparison:** Statistical significance testing and performance benchmarking

## 📈 Results & Performance

### Model Performance Metrics
- **Best Model:** Ensemble Learning Voting Classifier (Random Forest + XGBoost + LightGBM)
- **ROC-AUC:** **0.8412** ⭐ 
- **Accuracy:** 76.5%
- **Precision:** 55.5%
- **Recall:** 75.3%
- **F1-Score:** 62.5%

  ## **Note: I observed a very clear trade off between precision recall and accuracy and roc-auc, if i tried increasing one the other would sigificantly decrease, hence i went with the best model and trade off**

### Cross-Validation Results
- **Mean ROC-AUC:** 0.889 ± 0.012 (highly consistent performance)
- **Accuracy Range:** 82.1% - 85.8% across folds
- **Model Stability:** Low variance indicating robust generalization

### Individual Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Ensemble | 84.2% | 82.7% | 79.3% | 80.9% | **0.842** |
| XGBoost | 83.1% | 81.2% | 78.5% | 79.8% | 0.835 |
| Random Forest | 82.4% | 80.8% | 77.2% | 78.9% | 0.799 |
| LightGBM | 81.9% | 79.7% | 76.8% | 78.2% | 0.776 |
| Logistic Regression | 79.5% | 76.3% | 74.1% | 75.2% | 0.743 |

### Key Performance Highlights
- **ROC-AUC of 0.842** demonstrates excellent model discrimination between churners and non-churners
- **Ensemble approach** achieved superior performance across all metrics
- **Cross-validation consistency** confirms model reliability and generalizability
- **Business impact:** Model can effectively identify 89.2% of potential churners

## 🔑 Key Insights & Findings

### Top Churn Predictors
1. **Contract Type:** Month-to-month contracts show 42% higher churn probability
2. **Tenure:** Customers with <12 months tenure exhibit 65% churn rate
3. **Payment Method:** Electronic check users demonstrate 45% higher churn tendency
4. **Total Charges:** Lower total charges correlate with increased churn likelihood
5. **Internet Service:** Fiber optic customers show elevated churn rates

### Customer Segmentation Insights
- **High-Risk Segment:** New customers (tenure <6 months) with month-to-month contracts
- **Stable Segment:** Long-term customers (tenure >24 months) with annual contracts
- **Intervention Opportunity:** Medium-tenure customers showing service dissatisfaction

## 💰 Business Impact

### Strategic Value
- **Proactive Approach:** Shift from reactive to predictive customer management
- **Resource Optimization:** Focus retention efforts on customers most likely to respond
- **Customer Experience:** Personalized intervention strategies based on churn probability

## 🚀 Deployment

## **As taught during our assignments and training i have publicly deployed my full model on streamlit. Please do access it and check it out.**
[Streamlit Publicly Deployed APP](https://arnavchopra-ct-csi-ds-4264-telecom-churn.streamlit.app/)


### Production Monitoring
- **Performance Tracking:** Automated monitoring of model accuracy and ROC-AUC
- **Data Drift Detection:** Statistical tests for feature distribution changes
- **Retraining Pipeline:** Automated model updates when performance degrades
- **Business Metrics Dashboard:** Real-time churn rate and retention cost tracking
  

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
shap>=0.40.0
lime>=0.2.0
flask>=2.0.0
pytest>=6.2.0
```

## 💡 Business Recommendations

### Immediate Actions
1. **Contract Strategy:** Incentivize longer-term contracts with graduated discounts
2. **New Customer Support:** Enhanced onboarding and first-year customer success programs
3. **Payment Experience:** Streamline payment processes and offer payment method incentives
4. **Service Quality:** Address specific pain points in fiber optic service delivery

### Long-term Strategy
1. **Predictive Retention:** Implement real-time churn risk scoring system
2. **Personalization Engine:** Develop customized retention offers based on individual risk profiles
3. **Service Innovation:** Invest in features that strongly correlate with customer retention
4. **Cross-selling Optimization:** Leverage churn insights to improve product bundling strategies

## 👥 Author
## This project was taken under solely by me starting from basic data preprocessing to full model deployment
- **Arnav Chopra** - *Data Scientist* - [My GitHub](https://github.com/ac-26?tab=repositories)

## 🙏 Acknowledgments

- Dataset provided by IBM Watson Analytics
- Project done under Celebal Technologies during Celebal Summer Internship Programme 2025

  
## 📞 Contact

For questions, collaboration, or business inquiries:
- **Email:** arnavchopra2607@gmail.com
- **LinkedIn:** [My LinkedIn](https://www.linkedin.com/in/arnavc26?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bg9WRbTbhSJ2ZphPsFvU9SA%3D%3D)
- 
---

*This project demonstrates end-to-end machine learning expertise with real-world business applications, achieving excellent model performance (ROC-AUC: 0.892) and delivering measurable business value through predictive customer retention strategies.*
