# Cost-Sensitive Loan Approval Prediction Model

This repository contains the code and analysis for a machine learning model designed to predict loan approval decisions. The primary goal is to minimize financial risk by developing a **cost-sensitive classification model** that accurately identifies high-risk applicants.

---

## Executive Summary

I developed a classification model to predict loan approval decisions using a dataset of applicant financial and credit features. After evaluating four different algorithms (Random Forest, Logistic Regression, XGBoost, and SVM) with a custom cost-sensitive metric, the **Support Vector Machine (SVM) model delivered the best performance**.

The final model focuses on minimizing the high business costs associated with approving bad loans (False Positives) and rejecting good applicants (False Negatives). On the test set, our model achieved an **estimated average cost of $1,325 per applicant**. This figure provides a concrete metric for the finance team to assess the model's viability against projected interest income.

Key insights from the model include:
* **Top Predictors:** `MonthlyIncome`, `LoanAmount`, `TotalAssets`, and `LengthOfCreditHistory` were the most impactful features in determining loan approval.
* **Segmented Risk:** The model's cost-effectiveness varies across different applicant segments. For example, applicants with 'High School' education or who are 'Widowed' represent lower-cost segments, while those with advanced degrees ('Doctorate', 'Master') have a higher average cost of error.

This model provides a robust, data-driven framework to improve loan portfolio profitability, refine credit policies, and implement strategic risk management.

---

## Business Problem

The current manual loan approval process at FinTech Innovations is time-consuming and prone to inconsistencies. Loan officers require a standardized, data-driven tool to improve efficiency and reduce potential bias.

The primary business objective is to create a machine learning model that automates the initial vetting of loan applications. The model must prioritize **correctly identifying potential defaults (True Negatives) over denying good applicants (False Positives)**, as the financial loss from a defaulted loan ($50,000) is significantly higher than the opportunity cost of a missed good loan ($8,000).

A **classification approach** was chosen over regression because the final business decision is binary (approve/deny). This provides clear, actionable outputs and is less sensitive to the numerous outliers present in the financial data.

### Success Criteria
* **Primary Metric:** A custom cost-based score calculated from the confusion matrix: `Cost = (False Positives * $50,000) + (False Negatives * $8,000)`. The goal is to minimize this value.
* **Secondary Metric:** **Precision** will also be monitored to ensure the model is effective at identifying and rejecting bad loans.

---

## Exploratory Data Analysis (EDA)

The analysis was performed on a dataset of 20,000 loan applicants with 36 features.

### Key Findings:
* **Data Quality:** The dataset is complete with **no missing values**.
* **Target Imbalance:** The target variable `LoanApproved` is imbalanced, with **76.1% of loans being denied (0)** and **23.9% approved (1)**.
* **High Correlation:** Several feature pairs were found to be highly correlated (e.g., `AnnualIncome` and `MonthlyIncome`; `Age` and `Experience`). To reduce multicollinearity, redundant features with higher variance like `AnnualIncome`, `NetWorth`, `MonthlyLoanPayment`, and `InterestRate` were dropped in favor of their counterparts.



---

## Modeling Pipeline

A sophisticated preprocessing and modeling pipeline was constructed using `scikit-learn`.

### 1. Preprocessing
A `ColumnTransformer` was used to apply different preprocessing steps to different feature types:
* **Numerical Features:** Imputed with the `median` value and scaled using `StandardScaler`.
* **Nominal Categorical Features:** Imputed with the `most_frequent` value and encoded using `OneHotEncoder`.
* **Ordinal Features (`EducationLevel`):** Imputed with the `most_frequent` value and encoded using `OrdinalEncoder` to preserve the inherent order.

### 2. Model Selection & Tuning
The modeling process involved two main phases:
1.  **Randomized Search:** A `RandomizedSearchCV` was run on four candidate models (Random Forest, Logistic Regression, XGBoost, SVM) to quickly identify the best-performing algorithm based on our custom cost scorer. **SVM emerged as the clear winner.**

| Model | Best Average Cost |
| :--- | :--- |
| **SVM** | **-$3,920,800** |
| Logistic Regression | -$5,222,000 |
| Random Forest | -$5,798,400 |
| XGBoost | -$5,844,000 |

2.  **Grid Search:** The SVM model was further fine-tuned with a `GridSearchCV` on a narrower set of hyperparameters to optimize its performance, resulting in the final model.

**Best SVM Parameters:**
```
classifier__C: 0.05
classifier__class_weight: None
classifier__gamma: auto
classifier__kernel: rbf
preprocessor__num__imputer__strategy: median
```

---

## Final Model Evaluation

The optimized SVM model demonstrates strong and balanced performance on the unseen test data.

* **Average Cost Per Applicant:** **$1,325**
* **Accuracy:** **92%**
* **ROC AUC Score:** **0.95**

### Performance Visuals
The confusion matrix shows a high number of True Negatives (correctly rejected bad loans), which aligns with our primary business goal.

 

### Classification Report
```
              precision    recall  f1-score   support

           0       0.92      0.98      0.95      3805
           1       0.92      0.73      0.81      1195

    accuracy                           0.92      5000
   macro avg       0.92      0.85      0.88      5000
weighted avg       0.92      0.92      0.92      5000
```
The model achieves a high precision of **92%** for class 1, indicating that when it predicts a loan should be approved, it is correct 92% of the time.

---

## Feature Importance & Business Insights

Permutation importance was used to identify the most influential features in the final model's decisions.



### Key Drivers of Loan Approval:
1.  **MonthlyIncome:** The most critical factor.
2.  **LoanAmount:** The size of the loan is the second most important feature.
3.  **TotalAssets:** Applicant's total assets play a significant role.
4.  **LengthOfCreditHistory:** A longer history positively influences the decision.
5.  **EducationLevel:** Higher education levels are correlated with higher approval rates.

### Cost Analysis by Segment
The model's average cost of error differs across customer segments. For instance, the cost is highest for applicants with **Master's degrees ($2,103)** and lowest for those with **High School education ($854)**. This suggests that the model finds it more difficult to classify applicants with advanced degrees, leading to more costly errors. This insight can be used to apply additional scrutiny or different business rules to higher-cost segments.



---

## Conclusion & Recommendations

The developed SVM model provides a powerful, cost-aware tool for standardizing and improving loan approval decisions. With a test set accuracy of 92% and an average cost per applicant of $1,325, it offers a strong baseline for deployment.

**Recommendations:**
1.  **Implementation:** The model can be implemented as a decision-support tool for loan officers, flagging high-risk applications for manual review while fast-tracking low-risk ones.
2.  **Risk-Based Pricing:** Instead of a simple approve/deny output, the model's probability scores can be used to segment applicants into risk tiers, allowing for dynamic interest rate pricing.
3.  **Policy Refinement:** The feature importance results (e.g., the high impact of `MonthlyIncome`) can be used to validate and refine existing credit policies.

Future improvements could involve exploring more complex, non-linear models like Gradient Boosting or neural networks to potentially capture more nuanced patterns and further reduce the cost of errors.

---
