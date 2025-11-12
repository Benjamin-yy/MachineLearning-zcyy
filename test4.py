# 202311000226-张崇艺扬
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# 3. 创建并训练带预剪枝的决策树模型
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=5, random_state=42)
clf.fit(X_train, y_train)

# 4. 使用五折交叉验证评估模型性能 (在训练集上)
print("====================================================")
print("               202311000226-张崇艺扬")
print("====================================================")

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print("--- 训练集5折交叉验证评估 ---")
print("Accuracy scores:", scores)
print("Accuracy mean:", scores.mean())
print("-" * 20)

precision = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision_weighted').mean()
recall = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall_weighted').mean()
fl = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted').mean()
print("精度 (weighted precision mean):", precision)
print("召回率 (weighted recall mean):", recall)
print("F1 值 (weighted f1 mean):", fl)
print("\n")

# 5. 使用测试集评估模型性能
y_pred = clf.predict(X_test)
print("--- 测试集性能评估 ---")
print("测试集上的分类报告:")
print(classification_report(y_test, y_pred))

test_accuracy = accuracy_score(y_test, y_pred)
print("测试集上的准确度:", test_accuracy)