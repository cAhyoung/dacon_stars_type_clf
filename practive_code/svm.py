from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


# svm 모델 불러오기 7가지 모델 중 svm.SVC로 불러옴
svm_clf = svm.SVC(decision_function_shape="ovo")

# training dataset 불러오기
train_data = pd.read_csv("./dataset/train.csv")

X = train_data.iloc[:, 2:]
y = train_data.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# 모델 학습
svm_clf.fit(X_train, y_train)

# 모델 예측
svm_pred = svm_clf.predict(X_test)

# 정확도 출력
print(accuracy_score(y_test, svm_pred))
