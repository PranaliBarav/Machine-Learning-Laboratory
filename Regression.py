#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


# Dataset: Study Hours vs Exam Scores
X = np.array([1, 2, 3, 4, 5, 6, 7, 8,9]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 60, 65, 70, 78, 85])


# In[3]:


print(X)


# In[4]:


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


# In[5]:


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


print("Linear Regression R² Score:", r2_score(y, y_pred))


# In[7]:


from sklearn.preprocessing import PolynomialFeatures


# In[8]:


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


# In[9]:


poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)


# In[10]:


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Fit')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


print("Polynomial Regression R² Score:", r2_score(y, y_poly_pred))


# In[12]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[13]:


iris = load_iris()
X = iris.data[:, :2]  # Use first 2 features for simplicity
y = (iris.target == 0).astype(int)  # Setosa = 1, Others = 0


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)


# In[16]:


print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

