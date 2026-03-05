import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Config
IMAGE_SHAPE = (64, 64, 3)
METADATA_PATH = "metadata.csv"
DATA_PATH = "image_data.npy"
SAVE_PATH = "cnn_section2_best.keras"

EPOCHS = 20
BATCH_SIZE = 32
SEED = 0


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
        raise ValueError(f"No rows found for split='{split_name}'")

    X = all_data[sub["index"].values]
    y = sub["class"].values.astype("float32").reshape(-1, 1)
    return X, y


#
# Model
 
def build_cnn():
    # A solid “starter CNN” that generalizes reasonably well
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=IMAGE_SHAPE),

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D(2),

        # IMPORTANT: GAP tends to overfit less than Flatten
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


 
# Train / Eval
 
def train_model(model, X_train, y_train, X_val, y_val, save_path: str):
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cb,
        verbose=2,
    )
    return history


def summarize_history(history):
    val_acc = history.history.get("val_accuracy", [])
    if not val_acc:
        return None, None
    best_epoch = int(np.argmax(val_acc) + 1)
    best_val = float(np.max(val_acc))
    return best_epoch, best_val


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

    print(f"Train(full): {X_train_full.shape} | Test: {X_test.shape}")
    print("Train labels (mean):", float(y_train_full.mean()), "| Test labels (mean):", float(y_test.mean()))
    print("X range:", float(X_train_full.min()), "to", float(X_train_full.max()))

    
    X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=SEED,
            stratify=y_train_full,
    )

    model = build_cnn()
    history = train_model(model, X_train, y_train, X_val, y_val, save_path=SAVE_PATH)

    best_epoch, best_val = summarize_history(history)
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch} | Best val_accuracy: {best_val:.3f}")

    # Evaluate best saved model on the REAL test set
    best_model = tf.keras.models.load_model(SAVE_PATH)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final test accuracy: {float(test_acc):.3f}")
    print(f"Saved best model to: {SAVE_PATH}")


if __name__ == "__main__":
    main()