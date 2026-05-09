# Hospital Length-of-Stay Prediction

**CSE271 — Data Science Methodology, Spring 2026, Egypt University of Informatics**

Live demo: <https://dsc.gabr.online>

End-to-end machine-learning framework for predicting hospital Length of Stay (LOS) on the Microsoft Hospital LOS dataset (~100,000 patient encounters). Eight models, two prediction tasks, an interactive Streamlit dashboard, and an IEEE-format technical report.

## Models

| Task | Models |
|------|--------|
| Regression — exact LOS in days | Linear Regression, Decision Tree, Random Forest★, XGBoost★ |
| Classification — Short / Medium / Long | Logistic Regression, k-NN, SVM★, MLP★ |

★ = bonus model beyond the lecture syllabus.

## Headline results

XGBoost regressor: **R² = 0.9624, RMSE = 0.454 days**.
MLP classifier: **accuracy = 0.908, weighted F1 = 0.908**.

## Repository layout

```
hospital-los/
├── models.py                       # Full training pipeline (CLI)
├── dashboard.py                    # Streamlit dashboard with real model inference
├── requirements.txt                # Python dependencies
├── LengthOfStay.csv                # Microsoft LOS dataset
├── models/                         # Pickled trained models (gitignored)
├── report_ieee/                    # Technical report
│   ├── main.tex                    # IEEE single-column report
│   ├── references.bib              # IEEE bibliography
│   ├── regenerate_figures.py       # Regenerates figures and pickles models
│   ├── inject_metrics.py           # Pipes metrics.json into LaTeX macros
│   ├── figures/                    # Vector PDF figures
│   └── main.pdf                    # Compiled report
└── deploy/                         # AWS deployment scripts
    ├── bootstrap.sh                # EC2 first-run setup
    ├── streamlit.service           # systemd unit
    └── Caddyfile                   # Reverse proxy + auto-HTTPS
```

## Quick start — local

```bash
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python report_ieee/regenerate_figures.py             # trains models, writes models/*.pkl
streamlit run dashboard.py                           # opens http://localhost:8501
```

## Team

| Member | ID | Contribution |
|--------|----|--------------|
| Yousef Alaa (lead) | 23-101037 | Project coordination, MLP, presentation |
| Mohamed Sameh | 23-101051 | Regression models, feature engineering |
| Yassin Sherif | 23-101269 | Classification models, dashboard |
| Abdelrahman Gabr | 23-101040 | Technical report, repo, audit, deployment |
