# 202311000226张崇艺扬
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# 1. 加载 Iris 数据集
iris = datasets.load_iris()
X = iris.data  
y = iris.target 

# 2. 留出法拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
random_state=42, stratify=y)

# 3. 使用 K 均值聚类算法进行训练，类别数设置为 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
print("====================================================")
print("               202311000226-张崇艺扬")
print("====================================================")

# 4. 使用五折交叉验证进行模型性能评估
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
for score in scoring:
    cv_score = cross_val_score(kmeans, X_train, y_train, cv=5, scoring=score)
    print(f'\nCross-validation {score}: {cv_score.mean():.4f} ± {cv_score.std():.4f}')

# 5. 使用测试集评估模型性能
y_pred = kmeans.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
print("\n")