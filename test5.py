#202311000226-张崇艺扬
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import numpy as np
print("====================================================")
print("               202311000226-张崇艺扬")
print("====================================================")

#1. 加载 Iris 数据集
iris = load_iris()
X = iris.data 
y = iris.target 

#2.使用留出法划分训练集和测试集(留出 1/3 数据作为测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42, stratify=y)

#3.使用训练集训练 BP神经网络
bp_model = MLPClassifier(
    hidden_layer_sizes=(10,10), 
    activation='relu',
    solver='adam',
    max_iter=1000, 
    random_state=42
)

bp_model.fit(X_train, y_train)

#4.使用五折交叉验证评估模型性能
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='accuracy')
precision_scores=cross_val_score(bp_model,X,y,cv=cv,scoring='precision_weighted')
recall_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='recall_weighted')
f1_scores = cross_val_score(bp_model, X, y, cv=cv, scoring='f1_weighted')

print("交叉验证性能:")
print(f" 准确度 (Accuracy): {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f}")
print(f" 精确率 (Precision): {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
print(f"召回率 (Recall): {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
print(f"F1 值(F1-score): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

#5.使用测试集评估模型性能
y_pred = bp_model.predict(X_test)

print("\n 测试集性能:")
print(f"准确度(Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"召回率(Recall): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 值(F1-score): {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\n 分类报告:")
print(classification_report(y_test, y_pred))