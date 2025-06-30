#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# In[5]:


# Load the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/student_feedback.csv")


# In[6]:


df.head()


# In[7]:


# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'Student ID'], inplace=True)


# In[8]:


# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())


# In[9]:


# Fill missing values with column means (or use df.dropna() to drop rows)
df.fillna(df.mean(numeric_only=True), inplace=True)


# In[10]:


avg_scores = df.mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_scores.values, y=avg_scores.index, palette="viridis")
plt.title("Average Feedback Scores per Attribute")
plt.xlabel("Average Score")
plt.tight_layout()
plt.show()


# In[11]:


# 2. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Survey Attributes")
plt.tight_layout()
plt.show()


# In[12]:


# 3. Key Driver Analysis
# Predicting Recommendation
X = df.drop(columns=["Course recommendation based on relevance"])
y = df["Course recommendation based on relevance"]


# In[13]:


model = LinearRegression()
model.fit(X, y)


# In[14]:


# Coefficients
importance = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind='barh', color='skyblue')
plt.title("Factors Influencing Course Recommendation")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()


# In[19]:


# 4. Clustering Students Based on Feedback
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters


# In[16]:


# Cluster summary
cluster_summary = df.groupby('Cluster').mean()
print("Average scores per cluster:")
print(cluster_summary)


# In[17]:


# 5. Optional: Visualizing Clusters with PCA (if desired)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)


# In[18]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=clusters, palette='Set2')
plt.title("Student Feedback Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()

