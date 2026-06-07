# cardio AI - phase 1: EDA
# Author : prince-gupta79
# Dataset : UCI Heart Disease
# Goal : understand the data before building any models


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# loading the dataset
df = pd.read_csv('heart (1).csv')

print("=" * 50)
print("Cardiosense AI - Data Intelligence Report")
print("=" * 50)

# 2. basic info
print(f"\n Dataset shape: {df.shape[0]} patients, {df.shape[1]} features")
print(f"\n Columns Names:\n{list(df.columns)}")
print(f"\n Data Type:\n{df.dtypes}")
print(f"\n Missing Values: \n {df.isnull().sum()}")
print(f"\n Basic Statistics:\n {df.describe()}")

# 3. target variable distribution
print(f"\n Target Distribution:")
print(df['target'].value_counts())
print(f"Heart Disease Rate: {df['target'].mean() * 100:.1f}%")

# 4. visualization
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(24, 18))
fig.suptitle("CardioSense AI - EDA",
               fontsize=20, fontweight='bold', y=1.01)

# plot 1 - Target distribution
ax1 = fig.add_subplot(3, 3, 1)
colors = ['#2ecc71', '#e74c3c']
df['target'].value_counts().plot(kind='bar', color=colors, ax=ax1, edgecolor='black')
ax1.set_title('Heart Disease Distribution', fontweight='bold')
ax1.set_xlabel('0 = No Disease, 1 = Disease')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No Disease', 'Disease'], rotation=0)

# plot 2 - Age distribution by target
ax2 = fig.add_subplot(3, 3, 2)
df[df['target'] == 0]['age'].hist(alpha=0.7, color='#2ecc71',
                                  label='No Disease', ax=ax2, bins=20)
df[df['target'] == 1]['age'].hist(alpha=0.7, color='#e74c3c',
                                  label='Disease', ax=ax2, bins=20)
ax2.set_title('Age Distribution by Condition', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')
ax2.legend()

# plot 3 - Max heart rate vs age
ax4 = fig.add_subplot(3, 3, 4)
scatter = ax4.scatter(df['age'], df['thalach'],
                      c=df['target'],
                      cmap='RdYlGn_r',
                      alpha=0.7, s=60)
ax4.set_title('Age vs Max Heart Rate', fontweight='bold')
ax4.set_xlabel('Age')
ax4.set_ylabel('Max Heart Rate')
plt.colorbar(scatter, ax=ax4, label='Disease')

# plot 5 - Cholesterol distribution
ax5 = fig.add_subplot(3, 3, 5)
sns.boxplot(data=df, x='target', y='chol',
            hue='target', palette=['#2ecc71', '#e74c3c'],
            ax=ax5, legend=False)
ax5.set_title('Cholesterol by Condition', fontweight='bold')
ax5.set_ylabel('Cholesterol (mg/dl)')

# plot 6 - Blood Pressure
ax6 = fig.add_subplot(3, 3, 6)
sns.boxplot(data=df, x='target', y='trestbps',
            hue='target', palette=['#2ecc71', '#e74c3c'],
            ax=ax6, legend=False)
ax6.set_title('Blood Pressure by Condition', fontweight='bold')
ax6.set_ylabel('Blood Pressure (mmHg)')

# plot 7 - Gender Distribution
ax7 = fig.add_subplot(3, 3, 7)
gender_disease = df.groupby(['sex', 'target']).size().unstack()
gender_disease.plot(kind='bar', color=['#2ecc71', '#e74c3c'],
                    ax=ax7, edgecolor='black')
ax7.set_title('Gender vs Heart Disease', fontweight='bold')
ax7.set_xticklabels(['Female', 'Male'], rotation=0)
ax7.legend(['No Disease', 'Disease'])

# plot 8 - Correlation heatmap
ax8 = fig.add_subplot(3, 3, 8)
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', ax=ax8,
            annot_kws={"size": 7})
ax8.set_title('Feature Correlation Matrix', fontweight='bold')

# plot 9 - ST depression
ax9 = fig.add_subplot(3, 3, 9)
sns.violinplot(data=df, x='target', y='oldpeak',
               hue='target', palette=['#2ecc71', '#e74c3c'],
               ax=ax9, legend=False)
ax9.set_title('ST Depression by Condition', fontweight='bold')
ax9.set_ylabel('ST Depression (mm)')

plt.tight_layout()
plt.savefig('EDA_report.png', dpi=150, bbox_inches='tight')
plt.subplots_adjust(hspace = 0.4, wspace = 0.3)
plt.show()

print(f"\n EDA Complete - Report saved as EDA_report.png")
print(f"\n Key Findings:")
print(f"   • {df.shape[0]} patients analysed")
print(f"   • {df['target'].sum()} patients have heart disease ({df['target'].mean()*100:.1f}%)")
print(f"   • Average age: {df['age'].mean():.1f} years")
print(f"   • Age range: {df['age'].min()} to {df['age'].max()} years")
print(f"   • Average cholesterol: {df['chol'].mean():.1f} mg/dl")
print(f"   • Most correlated feature with disease: "
      f"{corr['target'].drop('target').abs().idxmax()}")