print("\n"+"=" * 60)
print("                    202311000226-张崇艺扬")
print("=" * 60)
import numpy as np
import warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning
#自定义逻辑回归类
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'n_iterations': self.n_iterations}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

#加载 iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

#使用留出法留出 1/3 的样本作为测试集(同分布取样)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

#创建自定义逻辑回归模型并训练
custom_lr = CustomLogisticRegression()
custom_lr.fit(X_train, y_train)

#创建 scikit-learn 逻辑回归模型并训练,处理ConvergenceWarning 警告
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning) #忽略 Future Warning,可能是因为
    #scikit-learn 版本问题导致的警告
    warnings.filterwarnings('error', category=ConvergenceWarning) #将 Convergence Warning 转
    #换为错误,以便进行处理
    try:
        sklearn_lr = LogisticRegression(max_iter=10000) #增加最大迭代次数
        sklearn_lr.fit(X_train, y_train)
    except ConvergenceWarning as e:
        print(f"Scikit-learn 逻辑回归模型未能收敛:{e}")

#定义评估函数
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    accuracy = np.mean(scores)
    print(f'五折交叉验证准确度:{accuracy}')

    y_pred = model.predict(X)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)#处理
    #UndefinedMetricWarning 警告
    print(f"精度:{precision}")

    recall = recall_score(y, y_pred, average='macro')
    print(f'召回率:{recall}')

    f1 = f1_score(y, y_pred, average='macro')
    print(f'F1值:{f1}\n')

#评估自定义逻辑回归模型
print("\n自定义逻辑回归模型评估:")
evaluate_model(custom_lr, X_train, y_train)

#评估 scikit-learn 逻辑回归模型
print("scikit-learn 逻辑回归模型评估:")
if 'sklearn_lr' in locals(): #确保模型已成功训练
    evaluate_model(sklearn_lr, X_train, y_train)

#使用测试集测试自定义逻辑回归模型
y_pred_test_custom = custom_lr.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_test_custom)
print("\n"f'自定义逻辑回归模型在测试集上的准确度:{accuracy_custom}')

precision_custom = precision_score(y_test, y_pred_test_custom, average='macro', zero_division=0)
print(f"精度:{precision_custom}")

recall_custom = recall_score(y_test, y_pred_test_custom, average='macro')
print(f'召回率: {recall_custom}')

f1_custom = f1_score(y_test, y_pred_test_custom, average='macro')
print(f'F1值:{f1_custom}\n')

#使用测试集测试 scikit-learn 逻辑回归模型
if 'sklearn_lr' in locals(): #确保模型已成功训练
    y_pred_test_sklearn = sklearn_lr.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_test_sklearn)
    print(f"scikit-learn 逻辑回归模型在测试集上的准确度:{accuracy_sklearn}")

    precision_sklearn = precision_score(y_test, y_pred_test_sklearn, average='macro',
    zero_division=0)
    print(f"精度:{precision_sklearn}")

    recall_sklearn = recall_score(y_test, y_pred_test_sklearn, average='macro')
    print(f'召回率:{recall_sklearn}')

    f1_sklearn = f1_score(y_test, y_pred_test_sklearn, average='macro')
    print(f"F1值:{f1_sklearn}""\n")