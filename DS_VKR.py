import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
# Импорт моделей
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from tensorflow import keras

df_1 = pd.read_excel("E:/Data Science/ВКР/X_bp.xlsx", index_col=0, engine='openpyxl')
df_2 = pd.read_excel("E:/Data Science/ВКР/X_nup.xlsx", index_col=0, engine='openpyxl')

df = pd.concat([df_1, df_2], axis=1, join='inner')

# Сводка по датасету
df.info()
# Считаем кол-во пропусков
df.isna().sum()
# Параметры распределения датасета
df.describe()
print(df.describe())


# Построение графиков
fig, axes = plt.subplots(2, 3)
i = 0
m = df.shape[1]
for column in df.columns:
    sns.histplot(df[column], ax=axes[0, i])
    sns.boxplot(data=df, x=column, ax=axes[1, i])
    i = i + 1
    if i > 2 or column == df.columns[m - 1]:
        plt.show()
        fig, axes = plt.subplots(2, 3)
        i = 0


i = 1
for column1 in df.columns:
    for column2 in df.columns[i:]:
        sns.jointplot(data=df, x=df[column1], y=df[column2], hue="Угол нашивки, град")
        plt.show()
    i = i + 1

# Удаление выбросов межкваритльными растояниями
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print('\n q1 =  \n', q1)
print('\n q3 =  \n', q3)
print('\n lower = \n', lower_bound, '\n'
                                    '\n upper = \n', upper_bound)
df_clean = df[(df > lower_bound) & (df < upper_bound)]

print('\n None = \n', df_clean.isna().sum())
df_clean.fillna(df_clean.median(), inplace=True)

df_in = df[(df < lower_bound) | (df > upper_bound)]
for col in df_in.columns:
    print('\n Выбросы: \n', df_in[col].dropna())

# Сводка по датасету
df_clean.info()
# Параметры распределения датасета
print(df_clean.describe())


# Построим корреляционную матрицу
plt.figure(figsize=(24, 12))
corr_heatmap = sns.heatmap(df_clean.corr(), vmin=-1, vmax=1, center=0, cmap="BrBG",
                           linewidths=0.1, annot=True)
corr_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()
#plt.savefig('corr_heatmap.png')
# Максимальные корреляции
print(df_clean.corr().abs().apply(lambda x: sorted(x)[-2]))
# Отсутствие линейной статистической зависимости

# Подготовка данных к обучению
# Нормализация MinMax, LabelEncoder для Угла нашивки можно не выполнять по формуле MinMax
y1 = df_clean[["Модуль упругости при растяжении, ГПа"]]
y2 = df_clean[["Прочность при растяжении, МПа"]]
X = df_clean.drop(columns=["Модуль упругости при растяжении, ГПа",
                           "Прочность при растяжении, МПа"])
print(df_clean.describe())
X_scaler = preprocessing.MinMaxScaler()
y1_scaler = preprocessing.MinMaxScaler()
y2_scaler = preprocessing.MinMaxScaler()
X = X_scaler.fit_transform(X)
y1 = y1_scaler.fit_transform(y1)
y2 = y2_scaler.fit_transform(y2)
# Разделяем датасет на тестовые и обучающие выборки
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3, random_state=22)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.3, random_state=22)


def test1_result(model, X_test, y_test, scaler, name):
    y_predict = model.predict(X_test)
    y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_mean = [y_test.mean()] * y_test.shape[0]
    rmse_0 = metrics.mean_squared_error(y_test, y_mean, squared=False)
    rmse = metrics.mean_squared_error(y_test, y_predict, squared=False)
    r2 = metrics.r2_score(y_test, y_predict)
    print('\nМодель ' + name + '\t E(x)')
    print('RMSE: {:0.4f}'.format(rmse) + '\t\t{:0.4f}'.format(rmse_0))
    print('R2: {:0.4f}'.format(r2))
    # plt.plot(y_test.values, label='y_test')
    plt.plot(y_test, label='y_test')
    plt.plot(y_predict, label='y_predict')
    plt.plot(y_mean, label='y_mean')
    plt.title(name, fontdict={'fontsize': 18}, pad=12)
    plt.grid()
    plt.legend()
    plt.show()
    return {'RMSE': rmse, 'R2': r2}


def design_models(X_train, X_test, y_train, y_test, y_scaler):
    # Инициализация сеток параметров моделей, по которым будет выполняться поиск
    LinReg = LinearRegression()
    LinReg_params = {'fit_intercept': [True, False],
                     'positive': [True, False]}
    SupVecR = SVR()
    SVR_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': range(1, 5),
                  'gamma': ['scale', 'auto']}
    RandForest = RandomForestRegressor()
    RandForest_params = {'n_estimators': range(10, 51, 10),
                         'max_depth': range(1, 10)}
    MLP = MLPRegressor()
    MLP_params = {'hidden_layer_sizes': [(7, 7, 1)],
                  'activation': ['tanh', 'relu'],
                  'alpha': [0.001],
                  'solver': ['adam'],
                  'max_iter': range(1000, 10001, 1000),
                  'learning_rate_init': [0.001, 0.01]}

    # Линейная регрессия
    model_LinReg = GridSearchCV(estimator=LinReg, scoring='neg_root_mean_squared_error',
                                param_grid=LinReg_params, cv=10)
    model_LinReg.fit(X_train, y_train)
    print(model_LinReg.best_estimator_)
    print(model_LinReg.best_score_)
    print(model_LinReg.best_params_)
    model_LinReg = LinearRegression(fit_intercept=model_LinReg.best_params_['fit_intercept'],
                                    positive=model_LinReg.best_params_['positive'])
    model_LinReg.fit(X_train, y_train)

    # Метод опорных векторов
    model_SVR = GridSearchCV(estimator=SupVecR, scoring='neg_root_mean_squared_error',
                             param_grid=SVR_params, cv=10)
    model_SVR.fit(X_train, np.ravel(y_train))
    print(model_SVR.best_estimator_)
    print(model_SVR.best_score_)
    print(model_SVR.best_params_)
    model_SVR = SVR(kernel=model_SVR.best_params_['kernel'],
                    gamma=model_SVR.best_params_['gamma'],
                    degree=model_SVR.best_params_['degree'])
    model_SVR.fit(X_train, np.ravel(y_train))

    # Случайный лес
    model_RandForest = GridSearchCV(estimator=RandForest, scoring='neg_root_mean_squared_error',
                                    param_grid=RandForest_params, cv=10, n_jobs=-1)
    model_RandForest.fit(X_train, np.ravel(y_train))
    print(model_RandForest.best_estimator_)
    print(model_RandForest.best_score_)
    print(model_RandForest.best_params_)
    model_RandForest = RandomForestRegressor(n_estimators=model_RandForest.best_params_['n_estimators'],
                                             max_depth=model_RandForest.best_params_['max_depth'])
    model_RandForest.fit(X_train, np.ravel(y_train))

    # Многослойный персептрон
    model_MLP = GridSearchCV(estimator=MLP, scoring='neg_root_mean_squared_error',
                             param_grid=MLP_params, cv=10, n_jobs=-1)
    model_MLP.fit(X_train, np.ravel(y_train))
    print(model_MLP.best_estimator_)
    print(model_MLP.best_score_)
    print(model_MLP.best_params_)
    model_MLP = MLPRegressor(activation=model_MLP.best_params_['activation'],
                             max_iter=model_MLP.best_params_['max_iter'],
                             learning_rate_init=model_MLP.best_params_['learning_rate_init'])
    model_MLP.fit(X_train, np.ravel(y_train))

    # Оцениваем точность на обучающей выборке
    model_LR_result = test1_result(model_LinReg, X_train, y_train, y_scaler, 'Linear Regression')
    model_SVR_result = test1_result(model_SVR, X_train, y_train, y_scaler, 'SVR')
    model_RF_result = test1_result(model_RandForest, X_train, y_train, y_scaler, 'Random Forest')
    model_MLP_result = test1_result(model_MLP, X_train, y_train, y_scaler, 'MLP')
    # Оцениваем точность на тестовой выборке
    model_LR_result = test1_result(model_LinReg, X_test, y_test, y_scaler, 'Linear Regression')
    model_SVR_result = test1_result(model_SVR, X_test, y_test, y_scaler, 'SVR')
    model_RF_result = test1_result(model_RandForest, X_test, y_test, y_scaler, 'Random Forest')
    model_MLP_result = test1_result(model_MLP, X_test, y_test, y_scaler, 'MLP')


# Модели прогноза Модуля упругости при растяжении
design_models(X1_train, X1_test, y1_train, y1_test, y1_scaler)
# Модели прогноза Прочность при растяжении
design_models(X2_train, X2_test, y2_train, y2_test, y2_scaler)


# Многослойный перцептрон на TensorFlow
df3 = df_clean.copy()
y = df3[["Соотношение матрица-наполнитель"]]
X = df3.drop(columns=["Соотношение матрица-наполнитель"])
# X = X.to_numpy()

X_scaler = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler()
X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

print(X1_train.shape, y1_train.shape)
print(X1_test.shape, y1_test.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

matrix_scaler = preprocessing.MinMaxScaler()
X_train = matrix_scaler.fit_transform(X_train)
X_test = matrix_scaler.fit_transform(X_test)
y_train = matrix_scaler.fit_transform(y_train)
y_test = matrix_scaler.fit_transform(y_test)
print(X_train.shape)



def test2_result(model, X_test, y_test, scaler, name):
    y_predict = model.predict(X_test)
    y_predict = scaler.inverse_transform(y_predict)
    y_test = scaler.inverse_transform(y_test)
    y_mean = [y_test.mean()] * y_test.shape[0]
    rmse_0 = metrics.mean_squared_error(y_test, y_mean, squared=False)
    rmse = metrics.mean_squared_error(y_test, y_predict, squared=False)
    r2 = metrics.r2_score(y_test, y_predict)
    print('\nМодель ' + name + '\t E(x)')
    print('RMSE: {:0.4f}'.format(rmse) + '\t\t{:0.4f}'.format(rmse_0))
    print('R2: {:0.4f}'.format(r2))
    # plt.plot(y_test.values, label='y_test')
    plt.plot(y_test, label='y_test')
    plt.plot(y_predict, label='y_predict')
    plt.plot(y_mean, label='y_mean')
    plt.title(name, fontdict={'fontsize': 18}, pad=12)
    plt.grid()
    plt.legend()
    plt.show()
    return {'RMSE': rmse, 'R2': r2}


def seq_model(X):
    # Создаем модель из последовательных слоев с указанными параметрами
    input_shape = (X.shape[1])
    inputs = keras.Input(shape=input_shape, name='inputs')
    x = keras.layers.Dense(10, activation='sigmoid', name='dense_1')(inputs)
    #x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(10, activation='sigmoid', name='dense_2')(x)
    outputs = keras.layers.Dense(1, name='outputs')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='matrix_predict')
    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='mse', metrics=["mae"])
    model.summary()
    return model


matrix_model = seq_model(X_train)

history = matrix_model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# Графики обучения модели
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label="val_loss")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

# Результаты обучения модели
score = matrix_model.evaluate(X_test, y_test, verbose=1)
y_max = y_scaler.data_max_[0]
y_min = y_scaler.data_min_[0]
print("Ошибка на тестовой выборке:", score[0] * (y_max - y_min) + y_min)
print("Абсолютная ошибка на тестовой выборке:", score[1] * (y_max - y_min) + y_min)
test2_result(matrix_model, X_train, y_train, y_scaler, 'SeqLayers')
test2_result(matrix_model, X_test, y_test, y_scaler, 'SeqLayers')

matrix_model.save("E:/PyCharm/PyCharmProjects/TryFlask/models")
matrix_model = keras.models.load_model('E:/PyCharm/PyCharmProjects/TryFlask/models')


