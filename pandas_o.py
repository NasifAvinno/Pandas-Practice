
# PANDAS

# loading a dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


# Load Iris dataset
iris = load_iris(as_frame=True)

# Store the raw dataframe
df_raw = iris.frame

# Make a copy
df = df_raw.copy()



# Basic DataFrame Operations


print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataFrame Information:")
df.info()

print("\nStatistical Description:")
print(df.describe())

print("\nShape:")
print(df.shape)

print("\nNumber of columns:")
print(len(df.columns))



# Selecting a Column


sepal_length = df['sepal length (cm)']

print("\nSepal Length:")
print(sepal_length.to_list())



# Filtering Rows


large_sepal = df.loc[df['sepal length (cm)'] > 5.0]

print("\nRows where sepal length > 5.0:")
print(large_sepal.head(3))


# Setosa = target 0
setosas = df[df['target'] == 0]

print("\nSetosa:")
print(setosas.head(3))



# Selecting Rows and Columns


subset = df.iloc[0:5, 0:3]

print("\nSubset:")
print(subset)



# GroupBy


print("\nMean values grouped by target:")
print(df.groupby('target').mean())


print("\nMean and Standard Deviation:")
print(df.groupby('target').agg(['mean', 'std']))



# Histogram


df['sepal width (cm)'].hist()

plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Width')

plt.show()



# Bar Chart


df['target'].value_counts().plot(kind='bar')

plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Number of Samples per Target')

plt.show()



# Create New Column


df['Petal ratio'] = (
    df['petal length (cm)'] /
    df['petal width (cm)']
)

print("\nDataFrame with Petal Ratio:")
print(df.head())



# Missing Values


print("\nMissing values:")
print(df.isnull().sum())


# Fill missing numeric values with mean
numeric_columns = df.select_dtypes(include='number').columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].mean()
)


# Check again
print("\nMissing values after filling:")
print(df.isnull().sum())



# Convert Target to Species


df['species'] = df['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})


print("\nSpecies:")
print(df[['target', 'species']].head())



# Encode Species


df['species encoded'] = df['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})


print("\nSpecies Encoding:")
print(df[['species', 'species encoded']].head())


# Crosstab


print("\nCrosstab:")
print(
    pd.crosstab(
        df['species'],
        df['species encoded']
    )
)



# Histogram for Each Species


plt.figure(figsize=(10, 6))

for species in df['species'].unique():

    subset = df[df['species'] == species]

    plt.hist(
        subset['sepal length (cm)'],
        alpha=0.5,
        bins=15,
        label=species
    )


plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length by Species')

plt.legend()

plt.show()

