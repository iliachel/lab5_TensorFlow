# -*- coding: utf-8 -*-
"""
Etot skript realizuet polnyy konveyer dlya zadachi predskazaniya rezultatov vyborov
na osnove neyronnoy seti.
On vypolnyaet sleduyushchie shagi:
1. Generatsiya dannykh (`dataIn.txt`, `dataOut.txt`).
2. Zagruzka i predobrabotka dannykh s proverkoy orientatsii.
3. Razdelenie dannykh na obuchayushchuyu i testovuyu vyborki.
4. Postroenie neyronnoy seti s odnim skrytym sloem i aktivatsiey 'logsig' (sigmoida).
5. Obuchenie modeli s ispolzovaniem EarlyStopping.
6. Otsenka proizvoditelnosti modeli s pomoshchyu razlichnykh metrik (tochnost, matritsa oshibok, ROC-AUC).
7. Vizualizatsiya rezultatov (krivye obucheniya, matritsa oshibok, PCA-proektsiya).
8. Sravnenie proizvoditelnosti neyroseti s klassicheskimi modelyami (Logistic Regression, Random Forest).
9. Sokhranenie obuchennoy modeli.
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
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.callbacks import EarlyStopping
import os

# --- Konfiguratsiya ---
N_SAMPLES = 200         # Kolichestvo izbirateley
INPUT_FEATURES = 12     # Kolichestvo voprosov (priznakov)
TEST_SIZE = 0.3         # Dolya dannykh dlya testirovaniya
RANDOM_STATE = 42       # Fiksatsiya sluchaynosti dlya vosproizvodimosti
HIDDEN_UNITS = 16       # Kolichestvo neyronov v skrytom sloe
EPOCHS = 100            # Maksimalnoe kolichestvo epokh obucheniya
BATCH_SIZE = 16         # Razmer paketa (batch)
PATIENCE = 10           # Terpenie dlya EarlyStopping (skolko epokh zhdat uluchsheniya)
MODEL_FILENAME = "mlp_elections_model.h5"  # Imya fayla dlya sokhraneniya modeli
PLOTS_DIR = "plots"     # Direktoriya dlya sokhraneniya grafikov


def generate_data(n_samples=N_SAMPLES, input_features=INPUT_FEATURES):
    """
    Generiruet i sokhranyaet sinteticheskie dannye o vyborakh.
    - dataIn.txt: (input_features x n_samples) otvety izbirateley.
    - dataOut.txt: (2 x n_samples) one-hot zakodirovannye rezultaty.
    """
    print("--- 1. Generatsiya dannykh ---")
    np.random.seed(RANDOM_STATE)

    # Generatsiya priznakov (12xN)
    X = np.random.randint(0, 2, size=(input_features, n_samples))

    # Generatsiya metok (one-hot 2xN)
    labels = np.random.randint(0, 2, size=(n_samples,))
    Y = np.zeros((2, n_samples), dtype=int)
    Y[labels, np.arange(n_samples)] = 1

    np.savetxt("dataIn.txt", X, fmt="%d")
    np.savetxt("dataOut.txt", Y, fmt="%d")
    print(f"Sgenerirovan dataIn.txt s formoy: {X.shape}")
    print(f"Sgenerirovan dataOut.txt s formoy: {Y.shape}")
    print("-" * 25)
    return X, Y


def load_and_preprocess_data():
    """
    Zagruzhaet dannye iz faylov, proveryaet orientatsiyu i vypolnyaet predobrabotku.
    Vozvrashchaet X (N, 12) i Y (N, 2).
    """
    print("--- 2. Zagruzka i predobrabotka dannykh ---")
    X_raw = np.loadtxt("dataIn.txt", dtype=int)
    Y_raw = np.loadtxt("dataOut.txt", dtype=int)

    # --- Transponirovanie pri neobkhodimosti, chtoby poluchit format (N, priznaki) ---
    # Obrabotka priznakov
    if X_raw.shape[0] == INPUT_FEATURES and X_raw.shape[1] != INPUT_FEATURES:
        X = X_raw.T
    else:
        X = X_raw

    # Obrabotka metok
    if Y_raw.shape[0] == 2 and Y_raw.shape[1] != 2:
        Y = Y_raw.T
    else:
        Y = Y_raw

    # Preobrazovanie v float32 dlya TensorFlow
    X = X.astype("float32")
    Y = Y.astype("float32")

    print(f"Zagruzhen X s itogovoy formoy (N x priznaki): {X.shape}")
    print(f"Zagruzhen Y s itogovoy formoy (N x klassy): {Y.shape}")
    print("-" * 25)
    return X, Y


def build_nn_model(input_dim, hidden_units):
    """
    Sozdaet model neyronnoy seti Keras s odnim skrytym sloem 'logsig'.
    """
    print("--- 3. Postroenie modeli neyronnoy seti ---")
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            # Skrytyy sloy s aktivatsiey 'sigmoid' (logsig)
            layers.Dense(
                hidden_units, activation="sigmoid", name="hidden_logsig_layer"
            ),
            # Vykhodnoy sloy s aktivatsiey 'softmax' dlya klassifikatsii na 2 klassa
            layers.Dense(2, activation="softmax", name="output_layer"),
        ]
    )
    # Kompilyatsiya modeli
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    print("-" * 25)
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Obuchaet model s ispolzovaniem kolbeka EarlyStopping dlya predotvrashcheniya pereobucheniya.
    """
    print("--- 4. Obuchenie modeli ---")
    # EarlyStopping prekratit obuchenie, esli val_loss ne uluchshaetsya v techenie 'patience' epokh
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
    print("Obuchenie zaversheno.")
    print("-" * 25)
    return history


def evaluate_and_report(model, X_test, y_test):
    """
    Otsenivaet model i vyvodit podrobnyy otchet o proizvoditelnosti.
    """
    print("--- 5. Otsenka proizvoditelnosti modeli ---")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Tochnost (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Tochnost (Accuracy): {accuracy:.4f}\n")

    # Otchet o klassifikatsii (precision, recall, f1-score)
    print("Otchet o klassifikatsii:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Pravyashchaya_Partiya", "Oppozitsiya"]
        )
    )

    # Matritsa oshibok (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    print("Matritsa oshibok:\n", cm)

    # ROC-AUC metrika
    # Dlya klassa "Oppozitsiya" (indeks 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC (dlya klassa Oppozitsiya): {roc_auc:.4f}")
    print("-" * 25)
    return cm, y_true, y_pred


def visualize_results(history, cm, X_test, y_true):
    """
    Generiruet i sokhranyaet vse neobkhodimye grafiki.
    """
    print("--- 6. Vizualizatsiya rezultatov ---")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 1. Grafiki poter (Loss) i tochnosti (Accuracy)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Poteri na obuchenii (Train Loss)")
    plt.plot(history.history["val_loss"], label="Poteri na validatsii (Validation Loss)")
    plt.title("Poteri modeli")
    plt.xlabel("Epokha")
    plt.ylabel("Poteri")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Tochnost na obuchenii (Train Accuracy)")
    plt.plot(history.history["val_accuracy"], label="Tochnost na validatsii (Validation Accuracy)")
    plt.title("Tochnost modeli")
    plt.xlabel("Epokha")
    plt.ylabel("Tochnost")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"))
    plt.show()

    # 2. Teplovaya karta matritsy oshibok
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pravyashchaya_Partiya", "Oppozitsiya"],
        yticklabels=["Pravyashchaya_Partiya", "Oppozitsiya"],
    )
    plt.title("Matritsa oshibok")
    plt.xlabel("Predskazannaya metka")
    plt.ylabel("Istinnaya metka")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()

    # 3. PCA-proektsiya dannykh
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1], c=y_true, cmap="coolwarm", alpha=0.8
    )
    plt.title("PCA-proektsiya testovykh dannykh (raskrasheno po istinnym metkam)")
    plt.xlabel("Glavnaya komponenta 1")
    plt.ylabel("Glavnaya komponenta 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Pravyashchaya_Partiya", "Oppozitsiya"])
    plt.savefig(os.path.join(PLOTS_DIR, "pca_projection.png"))
    plt.show()
    print(f"Grafiki sokhraneny v direktoriyu '{PLOTS_DIR}'.")
    print("-" * 25)

def compare_with_classical_models(X_train, y_train, X_test, y_test):
    """
    Obuchaet i otsenivaet modeli Logistic Regression i Random Forest dlya sravneniya.
    """
    print("--- 7. Sravnenie s klassicheskimi modelyami ---")
    # Preobrazuem one-hot metki v obychnye (0, 1) dlya modeley sklearn
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Logisticheskaya regressiya
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train_labels)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test_labels, y_pred_lr)
    print(f"Tochnost Logistic Regression: {lr_accuracy:.4f}")

    # Sluchaynyy les (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train_labels)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test_labels, y_pred_rf)
    print(f"Tochnost Random Forest: {rf_accuracy:.4f}")
    print("-" * 25)


def main():
    """
    Osnovnaya funktsiya dlya zapuska vsego konveyera.
    """
    # Shag 1: Generatsiya dannykh (esli fayly otsutstvuyut)
    if not (os.path.exists("dataIn.txt") and os.path.exists("dataOut.txt")):
        generate_data()

    # Shag 2: Zagruzka i predobrabotka dannykh
    X, Y = load_and_preprocess_data()

    # Shag 3: Razdelenie dannykh na obuchayushchuyu i testovuyu vyborki
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=np.argmax(Y, axis=1)
    )
    print(f"Dannye razdeleny na: Obuchayushchie {X_train.shape}, Testovye {X_test.shape}")
    print("-" * 25)

    # Shag 4: Postroenie i obuchenie neyronnoy seti
    model = build_nn_model(input_dim=X.shape[1], hidden_units=HIDDEN_UNITS)
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Shag 5: Otsenka modeli
    cm, y_true, _ = evaluate_and_report(model, X_test, y_test)

    # Shag 6: Vizualizatsiya rezultatov
    visualize_results(history, cm, X_test, y_true)

    # Shag 7: Sravnenie s drugimi modelyami
    compare_with_classical_models(X_train, y_train, X_test, y_test)

    # Shag 8: Sokhranenie itogovoy modeli
    print("--- 8. Sokhranenie itogovoy modeli ---")
    model.save(MODEL_FILENAME)
    print(f"Model sokhranena v {MODEL_FILENAME}")
    print("-" * 25)

    print("Konveyer uspeshno zavershil rabotu!")


if __name__ == "__main__":
    main()