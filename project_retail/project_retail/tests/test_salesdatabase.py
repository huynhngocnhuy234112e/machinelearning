
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from project_retail.project_retail.connectors.connector import Connector
from project_retail.project_retail.models.customer_clustering import (
    get_cluster_details,
    print_customers_by_cluster,
    export_clusters_to_excel,
)

# Chọn chế độ thực thi: all | console | excel | web
MODE = 'all'
args = [a.lower() for a in sys.argv[1:]]
if '--web' in args:
    MODE = 'web'
elif '--excel' in args:
    MODE = 'excel'
elif '--console' in args:
    MODE = 'console'

conn=Connector(database="salesdatabase")
conn.connect()
sql="select * from customer"
df=conn.queryDataset(sql)
print(df)

sql2=("select distinct customer.CustomerId, Age, Annual_Income,Spending_Score from customer, customer_spend_score "
      "where customer.CustomerId=customer_spend_score.CustomerID")
df2=conn.queryDataset(sql2)
print(df2)

df2.columns = ['CustomerId', 'Age', 'Annual Income', 'Spending Score']

print(df2)
print(df2.head())
print(df2.describe())

def showHistogram(df, columns):
    plt.figure(1, figsize=(7, 8))
    n = 0
    for column in columns:
        n += 1
        plt.subplot(3, 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32)  # Lưu ý: distplot đã deprecated; có thể đổi sang histplot
        plt.title(f'Histogram of {column}')
    plt.show()

# Bỏ qua CustomerId
showHistogram(df2, df2.columns[1:])

def elbowMethod(df, columnsForElbow):
    X = df.loc[:, columnsForElbow].values
    inertia = []
    for n in range(1, 11):
        model = KMeans(n_clusters=n, init='k-means++', max_iter=500, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(figsize=(15, 6))          # đừng tái dùng figure(1)
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-.', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster sum of squared distances')
    plt.title('Elbow Method')
    plt.show()

# GỌI HÀM Ở NGOÀI:
elbowMethod(df2, ['Age', 'Spending Score'])

def runKMeans(X, cluster):
    model = KMeans(
        n_clusters=cluster,
        init='k-means++',
        max_iter=500,
        random_state=42
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)
    return y_kmeans, centroids, labels

# Chọn 2 trục để vẽ: Age & Spending Score
columns = ['Age', 'Spending Score']
X = df2.loc[:, columns].values

cluster = 4
colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]

y_kmeans, centroids, labels = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)

df2['cluster'] = labels

# In chi tiết và xuất Excel cho kịch bản 2D
df2_2d = df2[['CustomerId', 'Age', 'Annual Income', 'Spending Score']].copy()
df2_2d['cluster'] = labels.astype(int)
details_2d = get_cluster_details(conn, df2_2d)
if MODE in ('console', 'all'):
    print("\n=== Chi tiết khách theo cụm (2D: Age + Spending Score, k=4) ===")
    print_customers_by_cluster(details_2d)
if MODE in ('excel', 'all'):
    export_clusters_to_excel(details_2d, "customer_clusters_2d_test_salesdatabase.xlsx")
    print("Excel (2D) saved:", os.path.abspath("customer_clusters_2d_test_salesdatabase.xlsx"))

def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(
            X[y_kmeans == i, 0],
            X[y_kmeans == i, 1],
            s=100,
            c=colors[i],
            label='Cluster %i' % (i + 1)
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

visualizeKMeans(
    X,
    y_kmeans,
    cluster,
    "Clusters of Customers - Age X Spending Score",
    "Age",
    "Spending Score",
    colors
)

import plotly.express as px
from sklearn.preprocessing import StandardScaler

# 1) Chọn 3 biến để vẽ 3D
columns = ['Age', 'Annual Income', 'Spending Score']

# 2) (khuyến nghị) scale trước khi KMeans
X_raw = df2.loc[:, columns].values
X = StandardScaler().fit_transform(X_raw)

# 3) Set k = 5 và train lại
cluster = 5
y_kmeans, centroids, labels = runKMeans(X, cluster)
df2['cluster'] = labels.astype(int)

# 4) Vẽ 3D với Plotly
def visualize3DKmeans(df, columns, hover_data, cluster):
    fig = px.scatter_3d(
        df, x=columns[0], y=columns[1], z=columns[2],
        color='cluster',
        hover_data=list(hover_data),
        category_orders={'cluster': list(range(cluster))}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

visualize3DKmeans(df2, columns, df2.columns, cluster)

# In chi tiết và xuất Excel cho kịch bản 3D
df2_3d = df2[['CustomerId', 'Age', 'Annual Income', 'Spending Score']].copy()
df2_3d['cluster'] = labels.astype(int)
details_3d = get_cluster_details(conn, df2_3d)
if MODE in ('console', 'all'):
    print("\n=== Chi tiết khách theo cụm (3D: Age + Annual Income + Spending Score, k=5, scaled) ===")
    print_customers_by_cluster(details_3d)
if MODE in ('excel', 'all'):
    export_clusters_to_excel(details_3d, "customer_clusters_3d_test_salesdatabase.xlsx")
    print("Excel (3D) saved:", os.path.abspath("customer_clusters_3d_test_salesdatabase.xlsx"))

# Chạy web mode nếu chọn --web
if MODE == 'web':
    from project_retail.project_retail.ui.cluster_web import app
    app.run(host="127.0.0.1", port=5000, debug=True)