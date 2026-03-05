import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report


# Config
IMAGE_SHAPE = (64, 64, 3)
METADATA_PATH = "metadata.csv"
DATA_PATH = "image_data.npy"
SAVE_PATH = "cnn_section2_best.keras"

EPOCHS = 20
BATCH_SIZE = 32
SEED = 0
AUTOTUNE = tf.data.AUTOTUNE


# Reproducibility
def set_seeds(seed: int = 0) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


# Data
def load_local_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH} in current folder.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH} in current folder.")

    all_data = np.load(DATA_PATH).astype("float32")
    metadata = pd.read_csv(METADATA_PATH)

    # Normalize to 0..1 if needed
    if all_data.max() > 1.5:
        all_data /= 255.0

    return all_data, metadata


def get_split(all_data, metadata, split_name: str):
    sub = metadata[metadata["split"] == split_name]
    if sub.empty:
        return None, None

    X = all_data[sub["index"].values]
    y = sub["class"].values.astype("float32").reshape(-1, 1)
    return X, y


def make_ds(X, y, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    return ds

# Model
 
def build_cnn():
    reg = tf.keras.regularizers.l2(1e-4)

    augment = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")
    
    inputs = tf.keras.layers.Input(shape=IMAGE_SHAPE)


    x = augment(inputs)

    # Slightly stronger CNN than your starter version: add BatchNorm
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
         
         ],
    )
    return model

def get_class_weights(y_train):
    y_int = y_train.reshape(-1).astype(int)
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_int)
    return {0: float(weights[0]), 1: float(weights[1])}


# Train / Eval

def summarize_history(history):
    val_auc = history.history.get("val_auc", [])
    if not val_auc:
        return None, None
    best_epoch = int(np.argmax(val_auc) + 1)
    best_val = float(np.max(val_auc))
    return best_epoch, best_val

def evaluate_and_report(model, X, y, name="set"):
    probs = model.predict(X, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    y_true = y.reshape(-1).astype(int)

    cm = confusion_matrix(y_true, preds)
    print(f"\n=== {name.upper()} CONFUSION MATRIX ===\n{cm}")
    print(f"\n=== {name.upper()} REPORT ===")
    print(classification_report(y_true, preds, digits=3))




def debug_predictions(model, X, y, name="set", n=400):
    # Small sanity check: are we predicting all one class?
    n = min(n, len(X))
    probs = model.predict(X[:n], verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    pct_ones = float(preds.mean())
    true_pct_ones = float(y[:n].mean())

    print(f"[debug] {name} prob mean/min/max: {probs.mean():.3f} {probs.min():.3f} {probs.max():.3f}")
    print(f"[debug] {name} preds % ones: {pct_ones:.3f} | true % ones: {true_pct_ones:.3f}")


 
# Main
 
def main():

    set_seeds(SEED)

    all_data, metadata = load_local_data()
    metadata = metadata[metadata["split"].isin(["train", "test"])].copy()

    X_train_full, y_train_full = get_split(all_data, metadata, "train")
    X_test, y_test = get_split(all_data, metadata, "test")
    X_field, y_field = get_split(all_data, metadata, "field")  # may be None

    if X_train_full is None or X_test is None:
        raise ValueError("Your metadata.csv must include at least train and test splits.")


    print(f"Train(full): {X_train_full.shape} | Test: {X_test.shape}")
    print("Train labels (mean):", float(y_train_full.mean()), "| Test labels (mean):", float(y_test.mean()))
    print("X range:", float(X_train_full.min()), "to", float(X_train_full.max()))

    
    X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=SEED,
            stratify=y_train_full,
    )

    train_ds = make_ds(X_train, y_train, training=True)
    val_ds = make_ds(X_val, y_val, training=False)

    class_weights = get_class_weights(y_train)
    print("Class weights:", class_weights)


    model = build_cnn()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    best_epoch, best_val = summarize_history(history)
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch} | Best val_auc: {best_val:.3f}")

    # Evaluate best saved model on the REAL test set
    best_model = tf.keras.models.load_model(SAVE_PATH)
    test_metrics = best_model.evaluate(X_test, y_test, verbose=0)
    evaluate_and_report(best_model, X_test, y_test, name="test")

    if X_field is not None:
        print("\n--- FINAL EVAL: FIELD (Section 3 idea) ---")
        field_metrics = best_model.evaluate(X_field, y_field, verbose=0)
        print(dict(zip(best_model.metrics_names, [float(x) for x in field_metrics])))
        evaluate_and_report(best_model, X_field, y_field, name="field")
    else:
        print("\n(no field split found in metadata.csv — skipping field evaluation)")

    print(f"\nSaved best model to: {SAVE_PATH}")

if __name__ == "__main__":
    main()