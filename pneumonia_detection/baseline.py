import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

IMAGE_SHAPE = (64, 64, 3) # Each image is 64 by 64 pixels with 3 colour channels

METADATA_PATH = "metadata.csv"
DATA_PATH = "image_data.npy"


def get_split(all_data, metadata, split_name, flatten=False):
    """Extracts either train or test set"""
    sub = metadata[metadata["split"] == split_name]

    X = all_data[sub["index"].values]
    y = sub["class"].values.astype(np.int32)

    if flatten:
        X = X.reshape((X.shape[0], np.prod(IMAGE_SHAPE)))

    return X, y


def main():

    # load data
    all_data = np.load(DATA_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # sklearn needs flattened images
    X_train, y_train = get_split(all_data, metadata, "train", flatten=True)
    X_test, y_test = get_split(all_data, metadata, "test", flatten=True)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=2),
    }

    print("\nTraining baseline models...\n")

    best_model = None
    best_score = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"{name}: {acc:.3f}")

        if acc > best_score:
            best_score = acc
            best_model = name

    print("\nBest model:", best_model)
    print("Best accuracy:", round(best_score, 3))


if __name__ == "__main__":
    main()