from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 이건 왜 불러오는걸까?
import warnings

warnings.filterwarnings('ignore')


# Decision Tree, Bagging Clf 모델 가져오기
dt_clf = DecisionTreeClassifier()
bag_clf = BaggingClassifier(dt_clf)

# training dataset 불러오기
train_data = pd.read_csv("./dataset/train.csv")

# 맞춰야 하는 것은 type이기 때문에 type과 나머지 데이터들을 분리해줌
X = train_data.iloc[:, 2:]
y = train_data.iloc[:, 1]

# training dataset과 test dataset으로 쪼개기
# training과 test의 비율은 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# 분리시켜준 dataset을 모델에 적용, 학습시킴
dt_clf.fit(X_train, y_train)
bag_clf.fit(X_train, y_train)

# 학습된 각 모델들을 test dataset을 이용해 분류
pred_dt = dt_clf.predict(X_test)
pred_bg = bag_clf.predict(X_test)

# 정확도 추출
accuracy_dt = accuracy_score(y_test, pred_dt)
accuracy_bg = accuracy_score(y_test, pred_bg)

# 정확도 출력
print('정확도 {0:.4f}'.format(accuracy_dt))
print('정확도 {0:.4f}'.format(accuracy_bg))
