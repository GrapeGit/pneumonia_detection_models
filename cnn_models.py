import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Config
IMAGE_SHAPE = (64, 64, 3)
METADATA_PATH = "metadata.csv"
DATA_PATH = "image_data.npy"
SAVE_PATH = "cnn_section2_best.keras"

EPOCHS = 30
BATCH_SIZE = 32
SEED = 0
FINAL_THRESHOLD = 0.95
AUTOTUNE = tf.data.AUTOTUNE


# Reproducibility
def set_seeds(seed: int = 0) -> None:
    """Set random seeds for TensorFlow and NumPy to improve reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


# Data
def load_local_data():
    """Load image data and metadata from local files."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH} in current folder.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH} in current folder.")

    all_data = np.load(DATA_PATH).astype("float32")
    metadata = pd.read_csv(METADATA_PATH)

    print(metadata.groupby(["split", "class"]).size())

    # Normalize to 0..1 if needed
    if all_data.max() > 1.5:
        all_data /= 255.0

    return all_data, metadata


def get_split(all_data, metadata, split_name: str):
    """Extract images and labels for a specific dataset split."""
    sub = metadata[metadata["split"] == split_name]
    if sub.empty:
        return None, None

    X = all_data[sub["index"].values]
    y = sub["class"].values.astype("float32").reshape(-1, 1)
    return X, y


def make_ds(X, y, training: bool):
    """Convert NumPy image and label arrays into a TensorFlow Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    return ds


def threshold_sweep(model, X, y, name="set"):
    """Evaluate model performance across several decision thresholds."""
    probs = model.predict(X, verbose=0).reshape(-1)
    y_true = y.reshape(-1).astype(int)

    print(f"\n    {name.upper()} THRESHOLD SWEEP     ")
    print("threshold  acc   bal_acc  macro_f1  pred_%_ones")

    for t in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]:
        preds = (probs >= t).astype(int)

        acc = accuracy_score(y_true, preds)
        bal = balanced_accuracy_score(y_true, preds)
        macro = f1_score(y_true, preds, average="macro", zero_division=0)
        pct_ones = preds.mean()

        print(f"{t:8.2f}  {acc:.3f}  {bal:.3f}    {macro:.3f}     {pct_ones:.3f}")

# Model
 
def build_cnn():
    """    Build and compile a convolutional neural network for binary image classification."""
    reg = tf.keras.regularizers.l2(1e-5)
    augment = tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.05),
    ], name="augmentation")
    inputs = tf.keras.layers.Input(shape=IMAGE_SHAPE)


    x = augment(inputs)

    # add BatchNorm
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
    x = tf.keras.layers.Dropout(0.2)(x)
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
    """Compute balanced class weights from training labels."""
    y_int = y_train.reshape(-1).astype(int)
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_int)
    return {0: float(weights[0]), 1: float(weights[1])}


# Train / Eval

def summarize_history(history):
    """Find the epoch with the best validation AUC."""
    val_auc = history.history.get("val_auc", [])
    if not val_auc:
        return None, None
    best_epoch = int(np.argmax(val_auc) + 1)
    best_val = float(np.max(val_auc))
    return best_epoch, best_val

def evaluate_and_report(model, X, y, name="set", threshold=0.5):
    """Evaluate a trained model using a custom decision threshold."""
    probs = model.predict(X, verbose=0).reshape(-1)
    preds = (probs >= threshold).astype(int)
    y_true = y.reshape(-1).astype(int)

    cm = confusion_matrix(y_true, preds)
    print(f"\n   {name.upper()} CONFUSION MATRIX      \n{cm}")
    print(f"\n      {name.upper()} REPORT   ")
    print(f"Threshold used: {threshold:.2f}")
    print(classification_report(y_true, preds, digits=3, zero_division=0))

def find_best_threshold(model, X_val, y_val):
    """Search for the decision threshold that gives the best balanced accuracy."""
    probs = model.predict(X_val, verbose=0).reshape(-1)
    y_true = y_val.reshape(-1).astype(int)

    print("\nClass 0 prob range:",
          probs[y_true == 0].min(),
          probs[y_true == 0].mean(),
          probs[y_true == 0].max())

    print("Class 1 prob range:",
          probs[y_true == 1].min(),
          probs[y_true == 1].mean(),
          probs[y_true == 1].max())

    best_t = 0.5
    best_score = -1.0

    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= t).astype(int)

        # Better for balanced binary classification
        #score = f1_score(y_true, preds, average="macro", zero_division=0)
        score = balanced_accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_t = t

    final_preds = (probs >= best_t).astype(int)

    print(f"\nBest threshold from validation set: {best_t:.2f}")
    print(f"Best validation balanced acc: {best_score:.3f}")
    print(f"Validation accuracy at best threshold: {accuracy_score(y_true, final_preds):.3f}")
    print(f"Validation balanced accuracy at best threshold: {balanced_accuracy_score(y_true, final_preds):.3f}")

    return best_t


def debug_predictions(model, X, y, name="set", n=400):
    """Print quick diagnostic statistics about predicted probabilities"""
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
    #metadata = metadata[metadata["split"].isin(["train", "test"])].copy()

    X_train_full, y_train_full = get_split(all_data, metadata, "train")
    X_test, y_test = get_split(all_data, metadata, "test")
    X_field, y_field = get_split(all_data, metadata, "field")

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
        verbose=2,
    )

    best_epoch, best_val = summarize_history(history)
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch} | Best val_auc: {best_val:.3f}")

    # Evaluate best saved model on the REAL test set
    best_model = tf.keras.models.load_model(SAVE_PATH)

    # debugging
    #threshold_sweep(best_model, X_val, y_val, name="val")
    #threshold_sweep(best_model, X_test, y_test, name="test")
    #threshold_sweep(best_model, X_field, y_field, name="field")


    debug_predictions(best_model, X_val, y_val, name="val")
    debug_predictions(best_model, X_test, y_test, name="test")

    #best_threshold = find_best_threshold(best_model, X_val, y_val)

    evaluate_and_report(
        best_model,
        X_test,
        y_test,
        name="test",
        threshold=FINAL_THRESHOLD
    )

    if X_field is not None:
        print("\nFINAL EVAL: FIELD")
        field_metrics = best_model.evaluate(X_field, y_field, verbose=0)
        print(dict(zip(best_model.metrics_names, [float(x) for x in field_metrics])))

        evaluate_and_report(
            best_model,
            X_field,
            y_field,
            name="field",
            threshold=FINAL_THRESHOLD
        )
    else:
        print("\n(no field split found in metadata.csv — skipping field evaluation)")

    print(f"\nSaved best model to: {SAVE_PATH}")

if __name__ == "__main__":
    main()