# Customer Churn Prediction — Summary Report

## Business problem and dataset
The goal is to identify telecom customers likely to leave the service so the retention team can contact them before churn. The IBM Telco Customer Churn dataset contains 7,043 customer records and 21 original columns, including services, contracts, billing, tenure, and the `Churn` target. Retention can protect Customer Lifetime Value (CLV) and reduce repeated Customer Acquisition Cost (CAC).

## Data preparation
`customerID` is excluded from the model inputs because it is an identifier. `TotalCharges` is converted from text to numeric, with whitespace converted to missing values. The notebook imputes missing numeric values, engineers `tenure_group`, `num_services`, and `AutoPay`, scales numeric features, and one-hot encodes categorical features. The split is 80% training and 20% testing, stratified with `random_state=42`.

## Class imbalance
The target contains more retained customers than churned customers. SMOTE is applied only to training rows; the test set keeps the original class distribution. This avoids evaluating on synthetic test customers.

## Model comparison and recommendation
The table below reports this run's test-set metrics. Results can vary slightly with package versions.

| Model         |   Accuracy |   Precision |   Recall |   F1-Score |   AUC-ROC |   Training Time (seconds) |
|:--------------|-----------:|------------:|---------:|-----------:|----------:|--------------------------:|
| GaussianNB    |     0.7062 |      0.4696 |   0.8262 |     0.5988 |    0.8153 |                    0.0039 |
| Decision Tree |     0.7523 |      0.5229 |   0.7647 |     0.6211 |    0.8329 |                    0.0173 |
| KNN           |     0.7012 |      0.4602 |   0.7273 |     0.5637 |    0.7651 |                    0.0009 |
| SVM           |     0.7601 |      0.5354 |   0.7273 |     0.6168 |    0.8195 |                    9.2605 |

**Recommendation:** Gaussian Naive Bayes had the highest churn recall in this run (0.826), meaning it identified the largest share of actual churners among these four tested models. Its precision was 0.470, so some customers it flags will not churn. Before deployment, the retention team should review whether this volume fits the daily call budget, and consider threshold tuning and a cost-based evaluation. The Decision Tree offers more interpretability; the SVM took longer to train in this run.

## Key churn signals and actions
Use the notebook's EDA, contract churn rates, false-negative profile, and Decision Tree feature importance to confirm the strongest signals. Month-to-month contracts and shorter tenure are plausible churn-risk patterns, but final claims should match the measured charts and feature importance. The retention team can test targeted onboarding, proactive service support, and contract-specific offers, then measure whether these actions reduce churn.

## Next steps
Tune the probability threshold around daily call capacity, use leakage-safe cross-validation with preprocessing and SMOTE inside a pipeline, monitor drift, collect retention-campaign outcomes, and expose the saved pipeline through a controlled prediction API.
