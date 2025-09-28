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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- Конфигурация ---
N_SAMPLES = 200         # Количество избирателей
INPUT_FEATURES = 12     # Количество вопросов (признаков)
TEST_SIZE = 0.3         # Доля данных для тестирования
RANDOM_STATE = 42       # Фиксация случайности для воспроизводимости
HIDDEN_UNITS = 16       # Количество нейронов в скрытом слое
EPOCHS = 100            # Максимальное количество эпох обучения
BATCH_SIZE = 16         # Размер пакета (batch)
PATIENCE = 10           # Терпение для EarlyStopping (сколько эпох ждать улучшения)
MODEL_FILENAME = "mlp_elections_model.h5"  # Имя файла для сохранения модели
PLOTS_DIR = "plots"     # Директория для сохранения графиков


def generate_data(n_samples=N_SAMPLES, input_features=INPUT_FEATURES):
    """
    Генерирует и сохраняет синтетические данные о выборах.
    - dataIn.txt: (input_features x n_samples) ответы избирателей.
    - dataOut.txt: (2 x n_samples) one-hot закодированные результаты.
    """
    print("--- 1. Генерация данных ---")
    np.random.seed(RANDOM_STATE)

    # Генерация признаков (12xN)
    X = np.random.randint(0, 2, size=(input_features, n_samples))

    # Генерация меток (one-hot 2xN)
    labels = np.random.randint(0, 2, size=(n_samples,))
    Y = np.zeros((2, n_samples), dtype=int)
    Y[labels, np.arange(n_samples)] = 1

    np.savetxt("dataIn.txt", X, fmt="%d")
    np.savetxt("dataOut.txt", Y, fmt="%d")
    print(f"Сгенерирован dataIn.txt с формой: {X.shape}")
    print(f"Сгенерирован dataOut.txt с формой: {Y.shape}")
    print("-" * 25)
    return X, Y


def load_and_preprocess_data():
    """
    Загружает данные из файлов, проверяет ориентацию и выполняет предобработку.
    Возвращает X (N, 12) и Y (N, 2).
    """
    print("--- 2. Загрузка и предобработка данных ---")
    X_raw = np.loadtxt("dataIn.txt", dtype=int)
    Y_raw = np.loadtxt("dataOut.txt", dtype=int)

    # --- Транспонирование при необходимости, чтобы получить формат (N, признаки) ---
    # Обработка признаков
    if X_raw.shape[0] == INPUT_FEATURES and X_raw.shape[1] != INPUT_FEATURES:
        X = X_raw.T
    else:
        X = X_raw

    # Обработка меток
    if Y_raw.shape[0] == 2 and Y_raw.shape[1] != 2:
        Y = Y_raw.T
    else:
        Y = Y_raw

    # Преобразование в float32 для TensorFlow
    X = X.astype("float32")
    Y = Y.astype("float32")

    print(f"Загружен X с итоговой формой (N x признаки): {X.shape}")
    print(f"Загружен Y с итоговой формой (N x классы): {Y.shape}")
    print("-" * 25)
    return X, Y


def build_nn_model(input_dim, hidden_units):
    """
    Создает модель нейронной сети Keras с одним скрытым слоем 'logsig'.
    """
    print("--- 3. Построение модели нейронной сети ---")
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            # Скрытый слой с активацией 'sigmoid' (logsig)
            layers.Dense(
                hidden_units, activation="sigmoid", name="hidden_logsig_layer"
            ),
            # Выходной слой с активацией 'softmax' для классификации на 2 класса
            layers.Dense(2, activation="softmax", name="output_layer"),
        ]
    )
    # Компиляция модели
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    print("-" * 25)
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Обучает модель с использованием колбэка EarlyStopping для предотвращения переобучения.
    """
    print("--- 4. Обучение модели ---")
    # EarlyStopping прекратит обучение, если val_loss не улучшается в течение 'patience' эпох
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )
    print("Обучение завершено.")
    print("-" * 25)
    return history


def evaluate_and_report(model, X_test, y_test):
    """
    Оценивает модель и выводит подробный отчет о производительности.
    """
    print("--- 5. Оценка производительности модели ---")
    y_prob = model.predict(X_test)  # Вероятности для каждого класса
    y_pred = np.argmax(y_prob, axis=1)  # Предсказанные метки
    y_true = np.argmax(y_test, axis=1)  # Истинные метки

    # Точность (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Точность (Accuracy): {accuracy:.4f}\n")

    # Отчет о классификации (precision, recall, f1-score)
    print("Отчет о классификации:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Правящая партия", "Оппозиция"]
        )
    )

    # Матрица ошибок (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    print("Матрица ошибок:\n", cm)

    # ROC-AUC метрика
    # Для класса "Оппозиция" (индекс 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC (для класса Оппозиция): {roc_auc:.4f}")
    print("-" * 25)
    return cm, y_true, y_pred


def visualize_results(history, cm, X_test, y_true):
    """
    Генерирует и сохраняет все необходимые графики.
    """
    print("--- 6. Визуализация результатов ---")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 1. Графики потерь (Loss) и точности (Accuracy)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Потери на обучении (Train Loss)")
    plt.plot(history.history["val_loss"], label="Потери на валидации (Validation Loss)")
    plt.title("Потери модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Точность на обучении (Train Accuracy)")
    plt.plot(history.history["val_accuracy"], label="Точность на валидации (Validation Accuracy)")
    plt.title("Точность модели")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"))
    plt.show()

    # 2. Тепловая карта матрицы ошибок
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Правящая партия", "Оппозиция"],
        yticklabels=["Правящая партия", "Оппозиция"],
    )
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанная метка")
    plt.ylabel("Истинная метка")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()

    # 3. PCA-проекция данных
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1], c=y_true, cmap="coolwarm", alpha=0.8
    )
    plt.title("PCA-проекция тестовых данных (раскрашено по истинным меткам)")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Правящая партия", "Оппозиция"])
    plt.savefig(os.path.join(PLOTS_DIR, "pca_projection.png"))
    plt.show()
    print(f"Графики сохранены в директорию '{PLOTS_DIR}'.")
    print("-" * 25)

def compare_with_classical_models(X_train, y_train, X_test, y_test):
    """
    Обучает и оценивает модели Logistic Regression и Random Forest для сравнения.
    """
    print("--- 7. Сравнение с классическими моделями ---")
    # Преобразуем one-hot метки в обычные (0, 1) для моделей sklearn
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Логистическая регрессия
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train_labels)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test_labels, y_pred_lr)
    print(f"Точность Logistic Regression: {lr_accuracy:.4f}")

    # Случайный лес (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train_labels)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test_labels, y_pred_rf)
    print(f"Точность Random Forest: {rf_accuracy:.4f}")
    print("-" * 25)


def main():
    """
    Основная функция для запуска всего конвейера.
    """
    # Шаг 1: Генерация данных (если файлы отсутствуют)
    if not (os.path.exists("dataIn.txt") and os.path.exists("dataOut.txt")):
        generate_data()

    # Шаг 2: Загрузка и предобработка данных
    X, Y = load_and_preprocess_data()

    # Шаг 3: Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=np.argmax(Y, axis=1)
    )
    print(f"Данные разделены на: Обучающие {X_train.shape}, Тестовые {X_test.shape}")
    print("-" * 25)

    # Шаг 4: Построение и обучение нейронной сети
    model = build_nn_model(input_dim=X.shape[1], hidden_units=HIDDEN_UNITS)
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Шаг 5: Оценка модели
    cm, y_true, _ = evaluate_and_report(model, X_test, y_test)

    # Шаг 6: Визуализация результатов
    visualize_results(history, cm, X_test, y_true)

    # Шаг 7: Сравнение с другими моделями
    compare_with_classical_models(X_train, y_train, X_test, y_test)

    # Шаг 8: Сохранение итоговой модели
    print("--- 8. Сохранение модели ---")
    model.save(MODEL_FILENAME)
    print(f"Модель сохранена в {MODEL_FILENAME}")
    print("-" * 25)

    print("Конвейер успешно завершил работу!")


if __name__ == "__main__":
    main()