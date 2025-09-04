# ============================================
#  Data Analysis & Visualization Assignment
# Using Pandas & Matplotlib (Iris Dataset)
# ============================================

# ---- Task 0: Import Libraries ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ---- Task 1: Load and Explore the Dataset ----
try:
    # Load the iris dataset from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame  # Convert to pandas DataFrame

    print(" Dataset loaded successfully!\n")

except FileNotFoundError:
    print(" Error: Dataset file not found.")
except Exception as e:
    print(f" An error occurred: {e}")

# Display first few rows
print(" First 5 rows of the dataset:")
print(df.head(), "\n")

# Explore dataset structure
print(" Dataset Info:")
print(df.info(), "\n")

print(" Check for Missing Values:")
print(df.isnull().sum(), "\n")

# Clean dataset (fill or drop missing values if any)
df = df.dropna()

# ---- Task 2: Basic Data Analysis ----
print(" Basic Statistics of Numerical Columns:")
print(df.describe(), "\n")

# Grouping: Average petal length by species
grouped = df.groupby("target")["petal length (cm)"].mean()
print(" Average Petal Length per Species:")
print(grouped, "\n")

# Observations
print(" Observations:")
print("- Iris-setosa generally has the shortest petal length.")
print("- Iris-virginica tends to have the longest petal length.")
print("- The dataset has no missing values and is well-structured.\n")

# ---- Task 3: Data Visualization ----
sns.set(style="whitegrid")  # make plots prettier

# 1. Line Chart - sepal length trend (first 30 rows for visibility)
plt.figure(figsize=(8,5))
plt.plot(df["sepal length (cm)"][:30], marker="o", label="Sepal Length")
plt.title("Line Chart: Sepal Length (First 30 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.savefig("line_chart.png")
plt.close()
print(" Saved: line_chart.png")

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(8,5))
species = [iris_data.target_names[i] for i in grouped.index]
plt.bar(species, grouped.values, color=["skyblue", "salmon", "lightgreen"])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.savefig("bar_chart.png")
plt.close()
print(" Saved: bar_chart.png")

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="purple", alpha=0.7)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()
print(" Saved: histogram.png")

# 4. Scatter Plot - sepal length vs petal length
plt.figure(figsize=(8,5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df["target"], cmap="viridis")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.savefig("scatter_plot.png")
plt.close()
print(" Saved: scatter_plot.png")
