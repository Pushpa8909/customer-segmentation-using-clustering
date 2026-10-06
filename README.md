# Customer Segmentation Using Clustering

## Project Overview

Customer segmentation is the process of dividing customers into groups based on their characteristics and purchasing behavior.

This project uses **Machine Learning and K-Means Clustering** to identify meaningful customer segments based on their **Annual Income** and **Spending Score**.

The project also includes a Flask-based web application for analyzing customers and predicting their cluster.

---

## Objectives

- Analyze customer purchasing behavior.
- Group customers using K-Means clustering.
- Identify meaningful customer segments.
- Visualize customer clusters.
- Provide a web application for customer segmentation.
- Support targeted marketing and personalized customer strategies.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS
- Jupyter Notebook
- Joblib

---

## Machine Learning Method

The project uses **K-Means Clustering**, an unsupervised machine learning algorithm.

The main features used for clustering are:

- Annual Income
- Spending Score

The **Elbow Method** is used to determine a suitable number of clusters.

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Elbow Method
6. K-Means Clustering
7. Cluster Visualization
8. Cluster Analysis
9. Customer Prediction
10. Web Application

---

## Project Structure

```text
customer-segmentation-using-clustering/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── css/
│   │       └── style.css
│   └── templates/
│       ├── dashboard.html
│       ├── index.html
│       └── predict.html
│
├── data/
│   └── customers_1500.xlsx
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
└── requirements.txt
