# Machine Learning for Data Scientists Masterclass

A comprehensive, hands-on + theory repository for data scientists covering core machine learning concepts from fundamentals through advanced topics. Every concept is taught via **Jupyter notebooks** combining theory (Markdown + LaTeX), runnable code, visualizations, and exercises.

---

## Repository Purpose

This masterclass provides:
- **Structured learning path** from data splitting basics to model deployment
- **Runnable notebooks** with synthetic + real datasets (sklearn built-ins)
- **Industry best practices**: pipelines, leakage prevention, reproducibility
- **Exercises + capstone projects** for hands-on practice
- **Reusable utility functions** for data generation, plotting, and evaluation

---

## Quick Start

### Option 1: Using `conda` (Recommended)

```bash
git clone https://github.com/<your-username>/Machine-Learning-for-Data-Scientists-Masterclass.git
cd Machine-Learning-for-Data-Scientists-Masterclass
conda env create -f environment.yml
conda activate ml-masterclass
jupyter lab
```

### Option 2: Using `venv` + `pip`

```bash
git clone https://github.com/<your-username>/Machine-Learning-for-Data-Scientists-Masterclass.git
cd Machine-Learning-for-Data-Scientists-Masterclass
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

### Option 3: Using `make`

```bash
make setup       # creates venv and installs deps
make notebooks   # launches Jupyter Lab
```

---

## Recommended Learning Path

| Order | Module | Topic | Notebooks |
|-------|--------|-------|-----------|
| 1 | `00` | Course Orientation | 1 |
| 2 | `ML050` | Prerequisites Quick Reference | 1 |
| 3 | `ML100` | Data Splitting & Feature Fundamentals | 6 |
| 4 | `ML200` | Linear Regression | 4 |
| 5 | `ML300` | Logistic Regression & Classification | 4 |
| 6 | `ML400` | KNN & Clustering | 5 |
| 7 | `ML500` | Trees, Ensembles & Boosting | 5 |
| 8 | `ML600` | Optimization, Regularization & Model Selection | 6 |
| 9 | `ML700` | Advanced Topics (Optional) | 4 |
| 10 | Projects | Capstone Projects | 3 |

**Total: 39 notebooks + 6 exercise notebooks + 3 projects**

---

## Repository Structure

```
Machine-Learning-for-Data-Scientists-Masterclass/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── Makefile
├── data/
│   ├── raw/           # Raw data placeholder
│   └── processed/     # Processed data placeholder
├── assets/
│   └── images/        # Exported diagrams
├── src/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── data_generation.py
│       ├── plotting.py
│       ├── metrics_helpers.py
│       └── preprocessing_helpers.py
├── notebooks/
│   ├── 00_Course_Orientation.ipynb
│   ├── ML050_ML_Prerequisites_Quick_Ref.ipynb
│   ├── ML100_Data_Splitting_and_Feature_Fundamentals/ (6 notebooks)
│   ├── ML200_Linear_Regression/ (4 notebooks)
│   ├── ML300_Logistic_Regression_and_Classification/ (4 notebooks)
│   ├── ML400_KNN_and_Clustering/ (5 notebooks)
│   ├── ML500_Trees_Ensembles_and_Boosting/ (5 notebooks)
│   ├── ML600_Optimization_Regularization_and_Model_Selection/ (6 notebooks)
│   └── ML700_Advanced_Topics_Optional/ (4 notebooks)
├── projects/
│   ├── Project_01_Regression_Price_Prediction.ipynb
│   ├── Project_02_Classification_Churn_or_Fraud.ipynb
│   └── Project_03_Clustering_Customer_Segmentation.ipynb
└── exercises/
    ├── README.md
    ├── ML100_exercises.ipynb
    ├── ML200_exercises.ipynb
    ├── ML300_exercises.ipynb
    ├── ML400_exercises.ipynb
    ├── ML500_exercises.ipynb
    └── ML600_exercises.ipynb
```

---

## Dependencies

**Core** (required):
- Python >= 3.9
- numpy, pandas, matplotlib, seaborn
- scikit-learn, scipy, jupyter/jupyterlab

**Optional** (for specific notebooks, gracefully handled with try/except):
- xgboost
- statsmodels
- imbalanced-learn (SMOTE)
- shap
- umap-learn

---

## Reproducibility

- All notebooks use `random_state=42` (or `np.random.seed(42)`)
- Synthetic datasets are generated with fixed seeds
- Each notebook is self-contained and runnable end-to-end

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Follow the existing notebook style (learning objectives, TOC, exercises)
4. Ensure notebooks run end-to-end without errors
5. Submit a pull request

---

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this material for any purpose, including commercially, as long as you give appropriate credit.
