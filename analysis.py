"""
=======================================================
  Project 1: Patient Readmission Risk Analysis
  Author: [Your Name]
  Tools: Python, Pandas, Matplotlib, Seaborn
  Dataset: Synthetic Hospital Patient Records (2000 patients)
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#F8FAFB',
    'axes.facecolor': '#F8FAFB',
})
PALETTE = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED']
RED     = '#DC2626'
BLUE    = '#2563EB'

# ── 1. Load & Inspect ──────────────────────────────────────────────────────
df = pd.read_csv('data/patients.csv')
print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nReadmission Rate: {df['readmitted_30days'].mean():.1%}")
print(f"\nData Types:\n{df.dtypes}")

# ── 2. Feature Engineering ─────────────────────────────────────────────────
df['age_group']       = pd.cut(df['age'], bins=[17,40,60,75,100],
                                labels=['18-40','41-60','61-75','75+'])
df['high_risk_diag']  = df['diagnosis'].isin(['Heart Failure','COPD']).astype(int)
df['long_stay']       = (df['length_of_stay'] > 7).astype(int)
df['poly_pharmacy']   = (df['num_medications'] >= 10).astype(int)
df['repeat_patient']  = (df['num_prev_admissions'] >= 2).astype(int)

print("\n\nFEATURE ENGINEERING DONE")
print(df[['age_group','high_risk_diag','long_stay','poly_pharmacy','repeat_patient']].head())

# ── 3. SQL-style Analysis with Pandas ──────────────────────────────────────
print("\n\n" + "="*60)
print("SQL-STYLE ANALYSIS")
print("="*60)

# Q1: Readmission rate by diagnosis
q1 = (df.groupby('diagnosis')['readmitted_30days']
        .agg(['sum','count','mean'])
        .rename(columns={'sum':'readmissions','count':'total','mean':'rate'})
        .sort_values('rate', ascending=False))
q1['rate_pct'] = (q1['rate'] * 100).round(1)
print("\n[Q1] Readmission Rate by Diagnosis:")
print(q1)

# Q2: Age group vs readmission
q2 = df.groupby('age_group')['readmitted_30days'].mean().reset_index()
q2.columns = ['age_group','readmission_rate']
print("\n[Q2] Readmission Rate by Age Group:")
print(q2)

# Q3: Discharge type impact
q3 = (df.groupby('discharge_type')['readmitted_30days']
        .agg(['mean','count'])
        .rename(columns={'mean':'rate','count':'total'})
        .sort_values('rate', ascending=False))
print("\n[Q3] Readmission Rate by Discharge Type:")
print(q3)

# Q4: Insurance type
q4 = df.groupby('insurance_type')['readmitted_30days'].mean().sort_values(ascending=False)
print("\n[Q4] Readmission Rate by Insurance:")
print(q4)

# Q5: Top high-risk patient profile
q5 = (df[df['readmitted_30days']==1]
        .groupby(['diagnosis','age_group','discharge_type'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(5))
print("\n[Q5] Top High-Risk Profiles:")
print(q5)

# ── 4. Visualization ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Patient 30-Day Readmission Risk Analysis', 
             fontsize=22, fontweight='bold', y=0.98, color='#1E293B')
fig.text(0.5, 0.955, '2,000 Patient Records  |  Healthcare Analytics Dashboard',
         ha='center', fontsize=12, color='#64748B')

# ── Plot 1: Readmission by Diagnosis ──
ax1 = fig.add_subplot(3, 3, 1)
colors_diag = [RED if r > 0.30 else BLUE for r in q1['rate']]
bars = ax1.barh(q1.index, q1['rate_pct'], color=colors_diag, height=0.6, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Readmission Rate (%)', fontsize=9)
ax1.set_title('Readmission Rate by Diagnosis', fontweight='bold', fontsize=11)
for bar, val in zip(bars, q1['rate_pct']):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=8.5, color='#1E293B')
ax1.set_xlim(0, 55)

# ── Plot 2: Age Group ──
ax2 = fig.add_subplot(3, 3, 2)
q2_sorted = q2.sort_values('readmission_rate', ascending=False)
clr = [RED if r > 0.3 else BLUE for r in q2_sorted['readmission_rate']]
ax2.bar(q2_sorted['age_group'], q2_sorted['readmission_rate']*100, color=clr, edgecolor='white')
ax2.set_ylabel('Readmission Rate (%)', fontsize=9)
ax2.set_title('Rate by Age Group', fontweight='bold', fontsize=11)
for i, v in enumerate(q2_sorted['readmission_rate']):
    ax2.text(i, v*100 + 0.5, f'{v:.1%}', ha='center', fontsize=8.5)

# ── Plot 3: Discharge Type ──
ax3 = fig.add_subplot(3, 3, 3)
q3_reset = q3.reset_index()
clr3 = [RED if r > 0.3 else BLUE for r in q3_reset['rate']]
ax3.barh(q3_reset['discharge_type'], q3_reset['rate']*100, color=clr3, height=0.5)
ax3.set_xlabel('Readmission Rate (%)', fontsize=9)
ax3.set_title('Rate by Discharge Type', fontweight='bold', fontsize=11)
for i, (_, row) in enumerate(q3_reset.iterrows()):
    ax3.text(row['rate']*100 + 0.5, i, f"{row['rate']:.1%}", va='center', fontsize=8)

# ── Plot 4: Heatmap – Diagnosis x Age Group ──
ax4 = fig.add_subplot(3, 3, (4, 5))
pivot = df.pivot_table(values='readmitted_30days', index='diagnosis',
                        columns='age_group', aggfunc='mean') * 100
sns.heatmap(pivot, ax=ax4, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, cbar_kws={'label': 'Readmission Rate %'}, annot_kws={'size': 9})
ax4.set_title('Readmission Heatmap: Diagnosis × Age Group', fontweight='bold', fontsize=11)
ax4.set_xlabel('Age Group', fontsize=9)
ax4.set_ylabel('Diagnosis', fontsize=9)

# ── Plot 5: Length of Stay Distribution ──
ax5 = fig.add_subplot(3, 3, 6)
read_yes = df[df['readmitted_30days']==1]['length_of_stay']
read_no  = df[df['readmitted_30days']==0]['length_of_stay']
ax5.hist(read_no,  bins=20, alpha=0.6, color=BLUE, label='Not Readmitted', density=True)
ax5.hist(read_yes, bins=20, alpha=0.6, color=RED,  label='Readmitted',     density=True)
ax5.set_xlabel('Length of Stay (days)', fontsize=9)
ax5.set_ylabel('Density', fontsize=9)
ax5.set_title('Length of Stay Distribution', fontweight='bold', fontsize=11)
ax5.legend(fontsize=8)

# ── Plot 6: Insurance Type ──
ax6 = fig.add_subplot(3, 3, 7)
ins_rates = df.groupby('insurance_type')['readmitted_30days'].mean().sort_values(ascending=False)
clr6 = [RED if r > 0.28 else BLUE for r in ins_rates]
ax6.bar(ins_rates.index, ins_rates*100, color=clr6, edgecolor='white')
ax6.set_ylabel('Readmission Rate (%)', fontsize=9)
ax6.set_title('Rate by Insurance Type', fontweight='bold', fontsize=11)
for i, v in enumerate(ins_rates):
    ax6.text(i, v*100 + 0.3, f'{v:.1%}', ha='center', fontsize=8.5)

# ── Plot 7: Prev Admissions vs Readmission ──
ax7 = fig.add_subplot(3, 3, 8)
prev_grp = df.groupby('num_prev_admissions')['readmitted_30days'].mean()
prev_grp = prev_grp[prev_grp.index <= 6]
ax7.plot(prev_grp.index, prev_grp*100, marker='o', color=RED, linewidth=2.5, markersize=7)
ax7.fill_between(prev_grp.index, prev_grp*100, alpha=0.15, color=RED)
ax7.set_xlabel('Number of Previous Admissions', fontsize=9)
ax7.set_ylabel('Readmission Rate (%)', fontsize=9)
ax7.set_title('Prior Admissions vs Readmission Risk', fontweight='bold', fontsize=11)

# ── Plot 8: Key KPI Summary ──
ax8 = fig.add_subplot(3, 3, 9)
ax8.axis('off')
kpis = [
    ('Total Patients',        f"{len(df):,}"),
    ('Overall Readmission Rate', f"{df['readmitted_30days'].mean():.1%}"),
    ('Highest Risk Diagnosis', 'Heart Failure'),
    ('Highest Risk Age Group', '75+'),
    ('Avg Stay – Readmitted', f"{df[df['readmitted_30days']==1]['length_of_stay'].mean():.1f} days"),
    ('Avg Stay – Not Readmitted', f"{df[df['readmitted_30days']==0]['length_of_stay'].mean():.1f} days"),
    ('High-Risk Discharge', 'Against Med. Advice'),
]
ax8.text(0.5, 1.0, '📋  Key Findings', ha='center', va='top',
         fontsize=13, fontweight='bold', color='#1E293B', transform=ax8.transAxes)
for i, (label, val) in enumerate(kpis):
    y = 0.88 - i * 0.12
    ax8.text(0.02, y, label, transform=ax8.transAxes,
             fontsize=8.5, color='#64748B')
    ax8.text(0.98, y, val, transform=ax8.transAxes,
             fontsize=9, fontweight='bold', color='#1E293B', ha='right')
    if i < len(kpis)-1:
        line = plt.Line2D([0.02, 0.98], [y - 0.04, y - 0.04],
                          transform=ax8.transAxes, color='#E2E8F0', linewidth=0.8)
        ax8.add_line(line)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('outputs/readmission_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Chart saved → outputs/readmission_analysis.png")

# ── 5. Risk Score ──────────────────────────────────────────────────────────
diag_map = {'Heart Failure':4,'COPD':3,'Diabetes':2,'Pneumonia':2,'Hip Fracture':1}
disch_map = {'Against Medical Advice':4,'Home':1,'Home Health Care':2,'Skilled Nursing Facility':3}
ins_map   = {'Uninsured':3,'Medicaid':2,'Medicare':1,'Private':1}

df['risk_score'] = (
    df['diagnosis'].map(diag_map) +
    df['discharge_type'].map(disch_map) +
    df['insurance_type'].map(ins_map) +
    (df['age'] >= 65).astype(int) * 2 +
    df['repeat_patient'] * 3 +
    df['long_stay'].astype(int) * 1 +
    df['poly_pharmacy'].astype(int) * 1
)
df['risk_level'] = pd.cut(df['risk_score'],
                           bins=[0,5,9,13,100],
                           labels=['Low','Moderate','High','Critical'])

print("\n\nRISK STRATIFICATION SUMMARY:")
print(df['risk_level'].value_counts())
print("\nReadmission Rate by Risk Level:")
print(df.groupby('risk_level')['readmitted_30days'].mean().apply(lambda x: f"{x:.1%}"))

# Save enriched data
df.to_csv('outputs/patients_with_risk.csv', index=False)
print("\n✅ Enriched dataset → outputs/patients_with_risk.csv")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
