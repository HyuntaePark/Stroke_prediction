import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# 데이터프레임 모든열 출력
pd.set_option('display.max_columns', None)

# 범주형 feature와 stroke과의 관계를 그래프로 표현
def graphs(feature):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 4)
    sns.set_style("white")
    sns.set_context("poster", font_scale=0.5)

    ax_stroke = fig.add_subplot(gs[:2, :2])
    sns.countplot(x=feature, hue='stroke', data=df, ax=ax_stroke, palette='coolwarm')
    sns.despine()

    ax_stroke = fig.add_subplot(gs[:2, 2:4], sharey=ax_stroke)
    sns.countplot(x='stroke', hue=feature, data=df, ax=ax_stroke, palette='coolwarm')
    sns.despine()
    plt.show()

# 수치형 feature와 stroke의 관계를 그래프로 표현
def num_graphs(feature, start, end, partial):
    f, ax = plt.subplots(1, 2, figsize=(20, 10))

    df.loc[df['stroke'] == 0][feature].plot.hist(ax=ax[0], bins=20, edgecolor='black', color='skyblue')
    ax[0].set_title('stroke = 0')
    ax1 = list(range(start, end, partial))
    ax[0].set_xticks(ax1)

    df[df['stroke'] == 1][feature].plot.hist(ax=ax[1], color='red', bins=20, edgecolor='black')
    ax[1].set_title('stroke=1')
    x2 = list(range(start, end, partial))
    ax[1].set_xticks(x2)
    plt.show();


def labelEnc(df):
    label = LabelEncoder()
    categorical_df = df.select_dtypes(include='object')
    numerical_df = df.select_dtypes(exclude='object')
    for i in range(0, len(categorical_df.columns)):
        df[categorical_df.columns[i]] = label.fit_transform(categorical_df.iloc[:, [i]].values)

    return df

def showHeatmap(df):
    heatmap_data = df
    colormap = plt.cm.PuBu
    plt.figure(figsize=(15, 15))
    plt.title("Correlation of Features", y=1.05, size=15)
    sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
                annot=True, annot_kws={"size": 8})
    plt.show()

def logistic_regression(x_train,y_train, x_test, y_test):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    pred = lr.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print('Logistic Regression을 이용한 데이터 예측 및 정확도')
    print('예측 데이터 : ', pred)
    print('정확도 : ', acc)
    plot_confusion_matrix(lr, x_test, y_test, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    return acc
# entropy 의사결정나무(Information Gain을 이용, 트리의 최대차수 10)
def decistion_tree_classifier(x_train, y_train, x_test, y_test):
    dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print('Decision Tree를 이용한 데이터 예측 및 정확도')
    print('예측 데이터 : ', pred)
    print('정확도 : ', acc)
    plot_confusion_matrix(dtc, x_test, y_test, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    return acc
def support_vector_machine(x_train,y_train,x_test,y_test):
    svc = SVC(probability=True)
    svc.fit(x_train, y_train)
    pred = svc.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print('Support Vector Machine을 이용한 데이터 예측 및 정확도')
    print('예측 데이터 : ', pred)
    print('정확도 : ', acc)
    plot_confusion_matrix(svc, x_test, y_test, cmap=plt.cm.Blues, normalize='all')
    plt.show()
    return acc
def k_neighbors_classifier(k, x, y):
    knn = KNeighborsClassifier(n_neighbors = k)
    pred = cross_val_predict(knn, x, y, cv=5)
    acc = cross_val_score(knn, x, y, cv=5, scoring="accuracy")
    acc_mean = acc.mean()

    return acc_mean


# 1. 데이터 수집 및 데이터 특성 확인
# 데이터셋 로드
# 데이터수집과 데이터분석을 위한 데이터프레임 df
df = pd.read_csv('C:\\Users\\Tae\\PycharmProjects\\Stroke\\stroke.csv')

# 데이터의 행렬 사이즈 확인
print(df.shape)
print()
# 상위 5개의 데이터 확인
print(df.head())
print()
# 데이터셋 통계치 확인
print(df.describe().T)
print()
# 데이터의 기본정보 확인
print(df.info())
print()

# 컬럼에서 탈락시킬 변수 결정 : 'id' 뇌졸중 유무랑 상관없다고 판단
# id 컬럼 탈락
df.drop(['id'], axis=1, inplace=True)

# 널 데이터 확인
print(df.isnull().sum()) # bmi에서 201개의 널값 발견

# bmi의 null값을 평균값으로 채움
df["bmi"].fillna(df["bmi"].mean(), inplace = True)

# 각 컬럼별 데이터타입 확인
print(df.dtypes)
# gender                object
# age                  float64
# hypertension           int64
# heart_disease          int64
# ever_married          object
# work_type             object
# Residence_type        object
# avg_glucose_level    float64
# bmi                  float64
# smoking_status        object
# stroke                 int64

## 뇌졸중과 범주형 feature와의 상관관계 그래프
# 성별 & 뇌졸중
graphs('gender')
# 성별 안에 other이라는 항목이 존재.
# 이상치로 판별 가능성이 있으므로 지원준다.
data_delete = df[df['gender'] == 'Other'].index
df = df.drop(data_delete)
# 결혼여부 & 뇌졸중
graphs('ever_married')
# 직업형태 & 뇌졸중
graphs('work_type')
# 주거지 & 뇌졸중
graphs('Residence_type')
# 흡연여부 & 뇌졸중
graphs('smoking_status')
# 고혈압 & 뇌졸중
graphs('hypertension')
# 심장병 & 뇌졸중
graphs('heart_disease')

# 뇌졸중과 수치형 feature와의 상관관계 그래프
# 나이 & 뇌졸중
num_graphs('age', 0, 85, 5)
# 글루코스 레벨 & 뇌졸중
num_graphs('avg_glucose_level',30 ,300, 20)
# bmi & 뇌졸중
num_graphs('bmi', 0, 70, 5)

# 종속변수와 독립변수와의 상관관계 분석을 위해 데이터셋 복사
corr_df = df.copy()
# 카테고리컬 데이터를 레이블인코딩을 통해 수치로 변환
# 오로지 상관도를 보기위해 레이블인코딩을 복사본에 진행
labelEnc(corr_df)
# 수치변환된 데이터셋을 이용해 히트맵 출력
showHeatmap(corr_df)

# 2. 데이터 클리닝 및 feature engineering
# 카테코리컬 데이터를 수치형 데이터로 변환
# pandas내 get_dummies 함수 활용 결과는 one-hotencoding 한 것처럼 나옴
df = pd.get_dummies(df)
print(df)

# 훈련 데이터와 테스트 데이터 나누기
x = df.drop('stroke', axis=1).values
y = df['stroke'].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=13,shuffle=True)

# 훈련 데이터에 대하여 MinMax Scaling진행
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 최적 알고리즘, 정확도를 구하기위한 변수
best_alg = ''
best_arg_score = 0
# 1. LogisticRegression
lr_score = logistic_regression(x_train, y_train, x_test, y_test)
if lr_score > best_arg_score:
    best_arg = 'Logistic Regression'
    best_arg_score = lr_score
print()
# 2. DecisionTreeClassifier
dtc_score = decistion_tree_classifier(x_train, y_train, x_test, y_test)
if dtc_score > best_arg_score:
    best_arg = 'Decision Tree'
    best_arg_score = dtc_score
print()
# 3. Support Vector Machine
svm_score = support_vector_machine(x_train,y_train, x_test, y_test)
if svm_score > best_arg_score:
    best_arg = 'Support Vector Machine'
    best_arg_score = svm_score
print()

# KNN 시각화는 plot말고 출력물로 결과를 대신함
# 4. K Neighbors Clasifier
knn_best_score = 0
# k를 1~20개까지 돌려보고 k가 몇일때 가장 정확도가 높은지 구함
for i in range(1, 21):
    score = k_neighbors_classifier(i, x, y)
    print('정확도 ( k = ', i, ') :', score)
    # Store the best parameter and score
    if score > knn_best_score:
        knn_best_parameter = i
        knn_best_score = score

print('KNN 최적 이웃 숫자 : ', knn_best_parameter)
print('KNN 최고 정확도 : ', knn_best_score)
print()
if knn_best_score > best_arg_score:
    best_arg = 'K Nearest Neighbor'
    best_arg_score = knn_best_score

# 위에 사용한 알고리즘에 대하여 최적 알고리즘과 최적 알고리즘 정확도를 구함
print('최적 알고리즘 : ', best_arg)
if best_arg == 'K Nearest Neighbor':
    print('KNN 최적 이웃 숫자 : ', knn_best_parameter)
print('최적 알고리즘 정확도 : ', best_arg_score)

