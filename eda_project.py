import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("student_data.csv")

print("="*50)
print("FIRST 5 RECORDS")
print("="*50)
print(df.head())

# ---------------------------
# Dataset Information
# ---------------------------
print("\nDataset Shape:", df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ---------------------------
# Statistical Summary
# ---------------------------
print("\nStatistical Summary")
print(df.describe())

# ---------------------------
# Correlation Matrix
# ---------------------------
numeric_df = df.select_dtypes(include=np.number)

print("\nCorrelation Matrix")
print(numeric_df.corr())

# ---------------------------
# Visualization Settings
# ---------------------------
sns.set_style("whitegrid")

# ---------------------------
# Histogram
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Marks"], bins=8)
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.savefig("histogram_marks.png")
plt.show()

# ---------------------------
# Box Plot
# ---------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Marks"])
plt.title("Boxplot of Marks")
plt.savefig("boxplot_marks.png")
plt.show()

# ---------------------------
# Scatter Plot
# ---------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Study_Hours",
    y="Marks",
    data=df
)
plt.title("Study Hours vs Marks")
plt.savefig("scatterplot.png")
plt.show()

# ---------------------------
# Correlation Heatmap
# ---------------------------
plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# ---------------------------
# Gender-wise Marks
# ---------------------------
plt.figure(figsize=(8,5))
sns.barplot(
    x="Gender",
    y="Marks",
    data=df
)
plt.title("Average Marks by Gender")
plt.savefig("gender_marks.png")
plt.show()

# ---------------------------
# Attendance vs Marks
# ---------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Attendance",
    y="Marks",
    data=df
)
plt.title("Attendance vs Marks")
plt.savefig("attendance_marks.png")
plt.show()

# ---------------------------
# Insights
# ---------------------------
print("\nKEY INSIGHTS")
print("-"*50)

print("Average Marks:",
      round(df["Marks"].mean(),2))

print("Highest Marks:",
      df["Marks"].max())

print("Lowest Marks:",
      df["Marks"].min())

print("\nAverage Marks by Gender:")
print(df.groupby("Gender")["Marks"].mean())

print("\nCorrelation with Marks:")
print(numeric_df.corr()["Marks"].sort_values(ascending=False))

print("\nEDA Completed Successfully!")
print("Graphs saved as PNG files.")