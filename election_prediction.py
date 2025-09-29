# -*- coding: utf-8 -*-
"""
Этот скрипт реализует полный конвейер для задачи предсказания результатов выборов
на основе нейронной сети.
Он выполняет следующие шаги:
1. Генерация данных (`dataIn.txt`, `dataOut.txt`).
2. Загрузка и предобработка данных с проверкой ориентации.
3. Разделение данных на обучающую и тестовую выборки.
4. Построение нейронной сети с одним скрытым слоем и активацией 'logsig' (сигмоида).
5. Обучение модели с использованием EarlyStopping.
6. Оценка производительности модели с помощью различных метрик (точность, матрица ошибок, ROC-AUC).
7. Визуализация результатов (кривые обучения, матрица ошибок, PCA-проекция).
8. Сравнение производительности нейросети с классическими моделями (Logistic Regression, Random Forest).
9. Сохранение обученной модели.
"""
# --- Импорт необходимых библиотек ---
import numpy as np  # Для работы с массивами и математических операций
import matplotlib.pyplot as plt  # Для создания графиков и визуализаций
import seaborn as sns  # Для более красивых визуализаций (например, тепловой карты)
from sklearn.model_selection import train_test_split  # Для разделения данных на обучающие и тестовые
from sklearn.metrics import (
    accuracy_score,  # Метрика: точность
    classification_report,  # Отчет с precision, recall, f1-score
    confusion_matrix,  # Матрица ошибок
    roc_curve,  # Для построения ROC-кривой
    auc,  # Для вычисления площади под ROC-кривой (AUC)
)
from sklearn.decomposition import PCA  # Для метода главных компонент (снижение размерности)
from sklearn.linear_model import LogisticRegression  # Классическая модель: Логистическая регрессия
from sklearn.ensemble import RandomForestClassifier  # Классическая модель: Случайный лес
from tensorflow import keras  # Основная библиотека для создания нейронных сетей
from keras._tf_keras.keras import layers  # Слои для построения архитектуры сети
from keras._tf_keras.keras.callbacks import EarlyStopping  # Коллбэк для ранней остановки обучения
import os  # Для работы с операционной системой (создание папок, проверка файлов)

# --- Глобальная конфигурация скрипта ---
N_SAMPLES = 200         # Общее количество избирателей (образцов данных)
INPUT_FEATURES = 12     # Количество вопросов в анкете (признаков)
TEST_SIZE = 0.3         # Доля данных, которая будет использоваться для тестирования (30%)
RANDOM_STATE = 42       # Фиксация генератора случайных чисел для воспроизводимости результатов
HIDDEN_UNITS = 16       # Количество нейронов в скрытом слое нейронной сети
EPOCHS = 100            # Максимальное количество эпох для обучения модели
BATCH_SIZE = 16         # Размер пакета (batch): сколько образцов обрабатывается за один шаг обучения
PATIENCE = 10           # "Терпение" для EarlyStopping: сколько эпох ждать улучшения, прежде чем остановить обучение
MODEL_FILENAME = "mlp_elections_model.h5"  # Имя файла для сохранения обученной модели
PLOTS_DIR = "plots"     # Название директории для сохранения всех графиков


def generate_data(n_samples=N_SAMPLES, input_features=INPUT_FEATURES):
    """
    Генерирует и сохраняет синтетические данные об ответах избирателей и результатах выборов.
    - dataIn.txt: Входные данные (признаки). Матрица размером (input_features x n_samples),
                  где каждый столбец - ответы одного избирателя.
    - dataOut.txt: Выходные данные (метки). Матрица размером (2 x n_samples),
                   где результаты закодированы в формате one-hot.
    """
    print("--- 1. Генерация данных ---")
    # Установка seed для генератора случайных чисел, чтобы результаты были одинаковыми при каждом запуске
    np.random.seed(RANDOM_STATE)

    # Генерация матрицы признаков X: (12 x N_SAMPLES), значения 0 или 1 (ответ "нет" или "да")
    X = np.random.randint(0, 2, size=(input_features, n_samples))

    # Генерация вектора меток (за кого проголосовал избиратель: 0 или 1)
    labels = np.random.randint(0, 2, size=(n_samples,))
    # Преобразование меток в формат one-hot encoding
    # Например, метка 0 -> [1, 0], метка 1 -> [0, 1]
    Y = np.zeros((2, n_samples), dtype=int)
    Y[labels, np.arange(n_samples)] = 1

    # Сохранение сгенерированных матриц в текстовые файлы
    np.savetxt("dataIn.txt", X, fmt="%d")
    np.savetxt("dataOut.txt", Y, fmt="%d")
    print(f"Сгенерирован dataIn.txt с формой: {X.shape}")
    print(f"Сгенерирован dataOut.txt с формой: {Y.shape}")
    print("-" * 25)
    return X, Y


def load_and_preprocess_data():
    """
    Загружает данные из файлов 'dataIn.txt' и 'dataOut.txt'.
    Проверяет ориентацию матриц и транспонирует их, если необходимо,
    чтобы данные имели формат (количество_образцов, количество_признаков).
    Возвращает X (N, 12) и Y (N, 2).
    """
    print("--- 2. Загрузка и предобработка данных ---")
    # Загрузка данных из файлов
    X_raw = np.loadtxt("dataIn.txt", dtype=int)
    Y_raw = np.loadtxt("dataOut.txt", dtype=int)

    # --- Проверка и исправление ориентации матриц ---
    # Нейросети в Keras ожидают данные в формате (количество_образцов, количество_признаков).
    # Наши сгенерированные данные имеют формат (количество_признаков, количество_образцов).
    # Поэтому их нужно транспонировать (поменять строки и столбцы местами).

    # Обработка матрицы признаков X
    if X_raw.shape[0] == INPUT_FEATURES and X_raw.shape[1] != INPUT_FEATURES:
        X = X_raw.T  # Транспонируем, если строки - это признаки
    else:
        X = X_raw

    # Обработка матрицы меток Y
    if Y_raw.shape[0] == 2 and Y_raw.shape[1] != 2:
        Y = Y_raw.T  # Транспонируем, если строки - это классы
    else:
        Y = Y_raw

    # Преобразование типов данных в float32, который является стандартным для TensorFlow/Keras
    X = X.astype("float32")
    Y = Y.astype("float32")

    print(f"Загружен X с итоговой формой (N x признаки): {X.shape}")
    print(f"Загружен Y с итоговой формой (N x классы): {Y.shape}")
    print("-" * 25)
    return X, Y


def build_nn_model(input_dim, hidden_units):
    """
    Создает модель нейронной сети Keras.
    Архитектура:
    - Входной слой
    - Один скрытый слой с функцией активации 'sigmoid' (в Matlab называется 'logsig').
    - Выходной слой с функцией активации 'softmax' для бинарной классификации.
    """
    print("--- 3. Построение модели нейронной сети ---")
    # Создаем последовательную модель
    model = keras.Sequential(
        [
            # Входной слой. Его форма определяется количеством признаков.
            layers.Input(shape=(input_dim,)),
            # Скрытый слой (Dense - полносвязный).
            # `hidden_units` - количество нейронов.
            # `activation="sigmoid"` - функция активации, преобразует выход нейрона в диапазон (0, 1).
            layers.Dense(
                hidden_units, activation="sigmoid", name="hidden_logsig_layer"
            ),
            # Выходной слой. 2 нейрона, так как у нас 2 класса (Правящая партия, Оппозиция).
            # `activation="softmax"` - преобразует выходы в вероятности принадлежности к каждому классу.
            # Сумма вероятностей на выходе всегда равна 1.
            layers.Dense(2, activation="softmax", name="output_layer"),
        ]
    )
    # Компиляция модели: определение оптимизатора, функции потерь и метрик
    model.compile(
        optimizer="adam",  # Адаптивный алгоритм оптимизации, хорошо подходит для большинства задач
        loss="categorical_crossentropy",  # Функция потерь для многоклассовой (в т.ч. бинарной с one-hot) классификации
        metrics=["accuracy"],  # Метрика, которую нужно отслеживать во время обучения - точность
    )
    # Вывод структуры модели в консоль
    model.summary()
    print("-" * 25)
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Обучает переданную модель на обучающих данных с использованием валидационных данных
    для контроля переобучения с помощью коллбэка EarlyStopping.
    """
    print("--- 4. Обучение модели ---")
    # Создание коллбэка EarlyStopping.
    # Он остановит обучение, если `val_loss` (потери на валидационной выборке)
    # не будет улучшаться в течение `PATIENCE` эпох.
    # `restore_best_weights=True` вернет веса модели к тому состоянию, где `val_loss` был минимальным.
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    # Запуск процесса обучения
    history = model.fit(
        X_train,  # Обучающие признаки
        y_train,  # Обучающие метки
        validation_data=(X_test, y_test),  # Данные для валидации после каждой эпохи
        epochs=EPOCHS,  # Максимальное количество эпох
        batch_size=BATCH_SIZE,  # Размер пакета
        callbacks=[early_stopping],  # Подключение коллбэков
        verbose=1,  # Режим вывода логов (1 - показывать прогресс-бар)
    )
    print("Обучение завершено.")
    print("-" * 25)
    # Возвращаем объект history, который содержит данные о процессе обучения (потери, точность на каждой эпохе)
    return history


def evaluate_and_report(model, X_test, y_test):
    """
    Оценивает производительность обученной модели на тестовых данных
    и выводит подробный отчет: точность, матрицу ошибок, отчет о классификации и ROC-AUC.
    """
    print("--- 5. Оценка производительности модели ---")
    # Получение предсказаний модели в виде вероятностей для каждого класса
    y_prob = model.predict(X_test)
    # Преобразование вероятностей в предсказанные классы (выбираем индекс с максимальной вероятностью)
    y_pred = np.argmax(y_prob, axis=1)
    # Получение истинных классов из one-hot формата
    y_true = np.argmax(y_test, axis=1)

    # Расчет и вывод точности (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Точность (Accuracy): {accuracy:.4f}\n")

    # Вывод отчета о классификации (precision, recall, f1-score для каждого класса)
    print("Отчет о классификации:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Правящая_Партия", "Оппозиция"]
        )
    )

    # Расчет и вывод матрицы ошибок (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    print("Матрица ошибок:\n", cm)

    # Расчет и вывод ROC-AUC метрики
    # Мы вычисляем ее для класса "Оппозиция" (индекс 1)
    # `pos_label=1` указывает, какой класс считать "положительным"
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC (для класса Оппозиция): {roc_auc:.4f}")
    print("-" * 25)
    return cm, y_true, y_pred


def visualize_results(history, cm, X_test, y_true):
    """
    Создает и сохраняет набор графиков для анализа результатов обучения и производительности модели.
    - Кривые обучения (Loss, Accuracy)
    - Тепловая карта матрицы ошибок
    - PCA-проекция тестовых данных
    """
    print("--- 6. Визуализация результатов ---")
    # Проверка, существует ли папка для графиков. Если нет - создаем ее.
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 1. Графики потерь (Loss) и точности (Accuracy) во время обучения
    plt.figure(figsize=(14, 6))
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Потери на обучении (Train Loss)")
    plt.plot(history.history["val_loss"], label="Потери на валидации (Validation Loss)")
    plt.title("Динамика потерь модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Точность на обучении (Train Accuracy)")
    plt.plot(history.history["val_accuracy"], label="Точность на валидации (Validation Accuracy)")
    plt.title("Динамика точности модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    # Сохранение фигуры в файл
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"))
    plt.show()

    # 2. Тепловая карта матрицы ошибок (Confusion Matrix)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,  # Показать значения в ячейках
        fmt="d",  # Формат целых чисел
        cmap="Blues",  # Цветовая схема
        xticklabels=["Правящая_Партия", "Оппозиция"],
        yticklabels=["Правящая_Партия", "Оппозиция"],
    )
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанная метка")
    plt.ylabel("Истинная метка")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()

    # 3. PCA-проекция тестовых данных для визуализации разделимости классов
    # Снижаем размерность с 12 признаков до 2 для построения 2D-графика
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 7))
    # Строим диаграмму рассеяния, где цвет точек определяется их истинным классом
    scatter = plt.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1], c=y_true, cmap="coolwarm", alpha=0.8
    )
    plt.title("PCA-проекция тестовых данных (раскрашено по истинным меткам)")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    # Добавляем легенду для соответствия цветов и классов
    plt.legend(handles=scatter.legend_elements()[0], labels=["Правящая_Партия", "Оппозиция"])
    plt.savefig(os.path.join(PLOTS_DIR, "pca_projection.png"))
    plt.show()
    print(f"Графики сохранены в директорию '{PLOTS_DIR}'.")
    print("-" * 25)

def compare_with_classical_models(X_train, y_train, X_test, y_test):
    """
    Обучает и оценивает две классические модели машинного обучения (Logistic Regression, Random Forest)
    на тех же данных для сравнения их производительности с нейронной сетью.
    """
    print("--- 7. Сравнение с классическими моделями ---")
    # Модели из scikit-learn требуют метки в формате (N,), а не one-hot (N, 2).
    # Преобразуем метки обратно из one-hot формата.
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # --- Логистическая регрессия ---
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train_labels)  # Обучение модели
    y_pred_lr = lr_model.predict(X_test)  # Получение предсказаний
    lr_accuracy = accuracy_score(y_test_labels, y_pred_lr)  # Расчет точности
    print(f"Точность Logistic Regression: {lr_accuracy:.4f}")

    # --- Случайный лес (Random Forest) ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train_labels)  # Обучение модели
    y_pred_rf = rf_model.predict(X_test)  # Получение предсказаний
    rf_accuracy = accuracy_score(y_test_labels, y_pred_rf)  # Расчет точности
    print(f"Точность Random Forest: {rf_accuracy:.4f}")
    print("-" * 25)


def main():
    """
    Основная функция, которая запускает все шаги конвейера последовательно.
    """
    # Шаг 1: Генерация данных (только если файлы 'dataIn.txt' и 'dataOut.txt' отсутствуют)
    if not (os.path.exists("dataIn.txt") and os.path.exists("dataOut.txt")):
        generate_data()

    # Шаг 2: Загрузка и предварительная обработка данных
    X, Y = load_and_preprocess_data()

    # Шаг 3: Разделение данных на обучающую и тестовую выборки
    # `stratify=np.argmax(Y, axis=1)` гарантирует, что пропорции классов
    # в обучающей и тестовой выборках будут такими же, как в исходном наборе данных.
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=np.argmax(Y, axis=1)
    )
    print(f"Данные разделены на: Обучающие {X_train.shape}, Тестовые {X_test.shape}")
    print("-" * 25)

    # Шаг 4: Построение и обучение нейронной сети
    model = build_nn_model(input_dim=X.shape[1], hidden_units=HIDDEN_UNITS)
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Шаг 5: Оценка производительности модели
    cm, y_true, _ = evaluate_and_report(model, X_test, y_test)

    # Шаг 6: Визуализация результатов
    visualize_results(history, cm, X_test, y_true)

    # Шаг 7: Сравнение с другими моделями
    compare_with_classical_models(X_train, y_train, X_test, y_test)

    # Шаг 8: Сохранение итоговой обученной модели
    print("--- 8. Сохранение итоговой модели ---")
    model.save(MODEL_FILENAME)
    print(f"Модель сохранена в {MODEL_FILENAME}")
    print("-" * 25)

    print("Конвейер успешно завершил работу!")


# Эта конструкция гарантирует, что функция main() будет вызвана только тогда,
# когда скрипт запускается напрямую, а не когда он импортируется как модуль.
if __name__ == "__main__":
    main()