# 🏥 Patient 30-Day Readmission Risk Analysis

> **Healthcare Data Analytics Project** | Python · Pandas · Matplotlib · Seaborn

---

## 📌 Project Overview

Hospital readmissions within 30 days are a critical quality metric — they cost the U.S. healthcare system **$26 billion annually** and often signal gaps in patient care. This project performs a comprehensive analysis of 2,000 synthetic patient records to identify **key risk factors** driving 30-day readmissions and builds a **rule-based risk stratification model**.

---

## 🎯 Business Questions Answered

| # | Question |
|---|----------|
| 1 | Which diagnoses have the highest readmission rates? |
| 2 | How does age impact readmission risk? |
| 3 | Does discharge type affect the likelihood of returning? |
| 4 | Are uninsured patients at higher risk? |
| 5 | What patient profile is highest risk? |

---

## 📊 Key Findings

- **Heart Failure (33.8%)** and **COPD (31.8%)** drive the highest readmission rates
- Patients **75+ years old** are ~70% more likely to be readmitted than 18-40 year olds
- **Discharge Against Medical Advice** carries a **37.5% readmission rate** — the highest of all discharge types
- **Uninsured patients** have the highest readmission rate (31.3%) among insurance types
- Patients with **2+ prior admissions** show sharply increasing readmission risk

---

## 🗂️ Project Structure

```
patient_readmission_analysis/
│
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   └── patients.csv            # Raw dataset (2,000 records)
│
├── outputs/
│   ├── readmission_analysis.png  # 9-panel visualization dashboard
│   └── patients_with_risk.csv    # Enriched dataset with risk scores
│
├── analysis.py                 # Full EDA + risk scoring pipeline
└── README.md
```

---

## 🔑 Features Engineered

| Feature | Description |
|---------|-------------|
| `age_group` | Binned age into clinical segments (18-40, 41-60, 61-75, 75+) |
| `high_risk_diag` | Flag for Heart Failure / COPD diagnoses |
| `long_stay` | Flag for LOS > 7 days |
| `poly_pharmacy` | Flag for 10+ medications |
| `repeat_patient` | Flag for 2+ prior admissions |
| `risk_score` | Composite risk score (0–18 scale) |
| `risk_level` | Low / Moderate / High / Critical stratification |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation & SQL-style analysis
- **Matplotlib / Seaborn** — multi-panel visualization
- **NumPy** — synthetic data generation & feature engineering

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/patient-readmission-analysis.git
cd patient-readmission-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn

# Generate dataset
python data/generate_data.py

# Run analysis
python analysis.py
```

---

## 📈 Output Dashboard

The script generates a 9-panel dashboard (`outputs/readmission_analysis.png`) including:
- Readmission rate by diagnosis, age group, discharge type, insurance
- Heatmap of Diagnosis × Age Group risk matrix
- Length of Stay distribution (readmitted vs not)
- Prior admissions vs readmission risk trend
- Summary KPI panel

---

## 💡 Business Impact / Recommendations

1. **Target Heart Failure & COPD patients** for post-discharge follow-up programs
2. **Flag patients discharged AMA** for immediate outreach within 48 hours
3. **Prioritize elderly (75+) uninsured patients** for social worker intervention
4. **Implement polypharmacy review** — patients on 10+ medications need medication reconciliation

---

## 📬 Contact

Created by **[Your Name]**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
