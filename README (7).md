<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0f2027,50:203a43,100:2c5364&height=260&section=header&text=Telco%20Customer%20Churn&fontSize=60&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Predict%20%E2%80%A2%20Prevent%20%E2%80%A2%20Retain&descAlignY=60&descSize=22" width="100%" alt="header"/>

<a href="https://github.com/YOUR-USERNAME/telco-churn-supervised-learning">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=00F5D4&center=true&vCenter=true&width=760&lines=Who+will+leave+next%3F+Let+the+model+tell+you+%F0%9F%94%AE;SMOTE+on+train+only+%E2%80%94+zero+test+leakage+%E2%9A%96%EF%B8%8F;GaussianNB+%7C+Decision+Tree+%7C+KNN+%7C+SVM+%F0%9F%A4%96;From+prediction+to+retention+action+%F0%9F%93%9E" alt="typing"/>
</a>

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-imbalanced--learn-2E86DE?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)

![Customers](https://img.shields.io/badge/Customers-7%2C043-00f5d4?style=flat-square)
![Columns](https://img.shields.io/badge/Columns-21-4cc9f0?style=flat-square)
![Models](https://img.shields.io/badge/Models-4-f9c74f?style=flat-square)
![Best Recall](https://img.shields.io/badge/Best%20Churn%20Recall-0.826-f72585?style=flat-square)

</div>

---

## 📌 Contents

[Overview](#-overview) · [Business Problem](#-business-problem) · [Pipeline](#-pipeline) · [SMOTE](#-class-imbalance--smote) · [Results](#-model-results) · [Findings](#-key-findings) · [Structure](#-repository-structure) · [Quick Start](#-quick-start) · [Saved Model](#-use-the-saved-model) · [Roadmap](#-roadmap) · [Video](#-video-walkthrough)

---

## 🎯 Overview

An end-to-end supervised learning project that predicts **telecom customer churn** using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

> [!NOTE]
> **`Churn = Yes`** means the customer left. **Recall on the churn class is the headline metric**, because every false negative is a customer the model failed to flag.

---

## 💼 Business Problem

| Metric | Why it matters |
|:--|:--|
| **CLV** — Customer Lifetime Value | Retention protects recurring revenue |
| **CAC** — Customer Acquisition Cost | Keeping a customer avoids paying to replace them |

Flag likely churners early, so the retention team can contact them **before** they leave.

---

## 🔄 Pipeline

```mermaid
flowchart LR
    A[📥 Load Data<br/>7,043 rows] --> B[🧹 Clean &<br/>Engineer Features]
    B --> C[✂️ Stratified<br/>80/20 Split]
    C --> D[⚖️ SMOTE<br/>train only]
    D --> E[🤖 Train<br/>4 Models]
    E --> F[📊 Evaluate on<br/>untouched test set]
    F --> G[🏆 Save<br/>Pipeline]
    G --> H[📞 Retention<br/>Actions]

    style A fill:#0077b6,color:#fff
    style B fill:#00b4d8,color:#fff
    style C fill:#2a9d8f,color:#fff
    style D fill:#e67e22,color:#fff
    style E fill:#9b59b6,color:#fff
    style F fill:#e63946,color:#fff
    style G fill:#2ecc71,color:#fff
    style H fill:#f72585,color:#fff
```

<details>
<summary><b>🧹 Data preparation details</b></summary>

<br/>

- `customerID` dropped (identifier, not a predictor)
- `TotalCharges`: text → numeric, blank strings → missing → imputed
- Engineered features: `tenure_group`, `num_services`, `AutoPay`
- Numeric features scaled, categorical features one-hot encoded
- Stratified 80 / 20 split, `random_state=42`

</details>

---

## ⚖️ Class Imbalance & SMOTE

More customers stay than leave, so a naive model can ignore churners. **SMOTE** creates synthetic churn examples, but **only in the training set**.

```text
BEFORE (train)                       AFTER SMOTE (train only)
🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵              🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵
🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵     ──▶      🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵
🔴🔴🔴                                🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴  (synthetic added)

TEST SET  →  untouched, original class distribution
```

> [!IMPORTANT]
> Never oversample the test set. Evaluating on synthetic customers would inflate every score.

---

## 🤖 Model Results

<div align="center">
<img src="https://quickchart.io/chart?w=800&h=400&bkg=transparent&c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22Gaussian%20NB%22%2C%22Decision%20Tree%22%2C%22KNN%22%2C%22SVM%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Recall%20%28%25%29%22%2C%22data%22%3A%5B82.6%2C76.5%2C72.7%2C72.7%5D%2C%22backgroundColor%22%3A%22%2300b4d8%22%7D%2C%7B%22label%22%3A%22Precision%20%28%25%29%22%2C%22data%22%3A%5B47.0%2C52.3%2C46.0%2C53.5%5D%2C%22backgroundColor%22%3A%22%23f77f00%22%7D%2C%7B%22label%22%3A%22F1%20%28%25%29%22%2C%22data%22%3A%5B59.9%2C62.1%2C56.4%2C61.7%5D%2C%22backgroundColor%22%3A%22%23a78bfa%22%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Churn-class%20metrics%20on%20the%20test%20set%22%2C%22fontColor%22%3A%22%23888%22%7D%2C%22legend%22%3A%7B%22labels%22%3A%7B%22fontColor%22%3A%22%23888%22%7D%7D%2C%22scales%22%3A%7B%22yAxes%22%3A%5B%7B%22ticks%22%3A%7B%22beginAtZero%22%3Atrue%2C%22max%22%3A100%2C%22fontColor%22%3A%22%23888%22%7D%7D%5D%2C%22xAxes%22%3A%5B%7B%22ticks%22%3A%7B%22fontColor%22%3A%22%23888%22%7D%7D%5D%7D%2C%22plugins%22%3A%7B%22datalabels%22%3A%7B%22anchor%22%3A%22end%22%2C%22align%22%3A%22top%22%2C%22color%22%3A%22%23888%22%2C%22font%22%3A%7B%22size%22%3A10%7D%7D%7D%7D%7D" width="80%" alt="Model comparison chart"/>
</div>

<br/>

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Train Time (s) |
|:--|--:|--:|--:|--:|--:|--:|
| **Gaussian NB** 🏆 | 0.7062 | 0.4696 | **0.8262** | 0.5988 | 0.8153 | 0.0039 |
| Decision Tree | 0.7523 | 0.5229 | 0.7647 | 0.6211 | **0.8329** | 0.0173 |
| KNN | 0.7012 | 0.4602 | 0.7273 | 0.5637 | 0.7651 | 0.0009 |
| SVM | **0.7601** | **0.5354** | 0.7273 | 0.6168 | 0.8195 | 9.2605 |

**Churn recall at a glance**

```text
Gaussian NB    ██████████████████████████████████████████  82.6%  🏆
Decision Tree  ██████████████████████████████████████░░░░  76.5%
KNN            ████████████████████████████████████░░░░░░  72.7%
SVM            ████████████████████████████████████░░░░░░  72.7%
```

<details>
<summary><b>🏆 Recommendation</b></summary>

<br/>

**Gaussian Naive Bayes** caught the largest share of real churners (recall **0.826**). Trade-off: precision is **0.470**, so some flagged customers would have stayed anyway.

- Check flagged volume against the team's **daily call budget**
- Tune the probability threshold and run a **cost-based evaluation** before deployment
- **Decision Tree** → most interpretable, best AUC
- **SVM** → best accuracy and precision, but slowest (~9 s)

</details>

---

## 🔍 Key Findings

| Signal | Where to confirm it in the notebook |
|:--|:--|
| 📅 Month-to-month contracts | Contract-wise churn rate chart |
| ⏳ Short tenure | Tenure-group churn chart |
| 🧬 Top drivers | Decision Tree feature importance |
| ❌ Missed churners | False-negative profile |

Suggested retention experiments: targeted onboarding, proactive support, contract-specific offers, then **measure** whether churn actually drops.

---

## 📁 Repository Structure

```text
telco-churn-supervised-learning/
├── CustomerChurn_SupervisedLearning.ipynb   # Tasks, explanations, code, plots
├── Telco-Customer-Churn.csv                 # Source dataset
├── churn_model.pkl                          # Fitted prediction pipeline
├── model_comparison.csv                     # Metrics table
├── summary_report.md                        # Business-facing report
├── requirements.txt
├── outputs/                                 # Saved comparisons and sample predictions
└── README.md
```

> [!WARNING]
> Make sure the CSV filename inside the notebook matches the file in this repo.

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR-USERNAME/telco-churn-supervised-learning.git
cd telco-churn-supervised-learning
```

<table>
<tr><td><b>🪟 Windows</b></td><td><b>🍎 macOS / 🐧 Linux</b></td></tr>
<tr>
<td>

```bash
python -m venv .venv
.venv\Scripts\activate
```

</td>
<td>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</td>
</tr>
</table>

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
python -m notebook
```

Open `CustomerChurn_SupervisedLearning.ipynb` and **Run All** cells top to bottom.

---

## 🧠 Use the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

new_customers = pd.read_csv("new_customers.csv")   # same raw columns as training

churn_probability = model.predict_proba(new_customers)[:, 1]
churn_flag = model.predict(new_customers)
```

---

## 🧪 Methodology Notes

> [!IMPORTANT]
> For the most rigorous evaluation, put **feature engineering → preprocessing → SMOTE → estimator** in a single `imbalanced-learn` pipeline and cross-validate on the original training split. This prevents leakage across folds.

---

## 🛣 Roadmap

- [x] Clean data and engineer features
- [x] SMOTE on training set only
- [x] Compare 4 models and save the pipeline
- [ ] Tune threshold around daily call capacity
- [ ] Leakage-safe cross-validation with SMOTE inside a pipeline
- [ ] Cost-based evaluation (missed churner vs wasted call)
- [ ] Drift monitoring and campaign outcome tracking
- [ ] Serve the pipeline via a controlled prediction API

---

## 🎥 Video Walkthrough

<!-- Replace with your Google Drive / unlisted YouTube URL -->
▶️ **[Watch the 5–10 minute project explanation](PASTE-YOUR-VIDEO-URL-HERE)**

---

<div align="center">

**Made with ❤️, Python and scikit-learn**

⭐ Star the repo if it helped you!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=120&section=footer" width="100%" alt="footer"/>

</div>
