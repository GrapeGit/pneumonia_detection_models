# Pneumonia Detection CNN
Last updated: May 3, 2026

Built this project wayyyy back when to learn more about convolutional neural network (CNN) to classify chest X-ray images as either **Pneumonia** or **No Pneumonia**. I've spruced it up a little since, but it's still a work in progress :) the model was trained using TensorFlow/Keras and can be used through a simple Streamlit web app

## Project Files

- `cnn_models.py` — trains and evaluates the CNN model
- `cnn_section2_best.keras` — saved best-performing CNN model
- `baseline.py` — early draft script used to test simple baseline models before the CNN. 
- `metadata.csv` — contains image labels, dataset split names, and image indices. Feel free to use !
- `image_data.npy` — NumPy array containing the image data. Also feel free to use!
- `app.py` — Streamlit app for uploading an image and getting a prediction

## Model Overview

The CNN takes chest X-ray images resized to `64 x 64 x 3` as input.

The model includes:

- Data augmentation
  - Random rotation
  - Random zoom
  - Random contrast
- Three convolutional blocks
- Batch normalization
- ReLU activation
- Max pooling
- Global average pooling
- Dropout
- Sigmoid output for binary classification

The model outputs a probability between `0` and `1`, where higher values indicate a higher predicted likelihood of pneumonia.

## Decision Threshold

I'm currently using a threshold of **.95** for classification. 


Images with a predicted probability greater than or equal to `0.95` are classified as:

```text
Pneumonia
```

Images with a predicted probability below `0.95` are classified as:

```text
No Pneumonia
```


## Accuracy Report

The final CNN achieved the following performance on the test set:

| Metric | Score |
|---|---:|
| Test Accuracy | 83.3% |
| Test Macro F1 | 83.2% |
| Class 0 Recall | 79.0% |
| Class 1 Recall | 87.5% |

Test confusion matrix:

```text
[[158  42]
 [ 25 175]]
```

This means the model correctly classified:

```text
158 class 0 images
175 class 1 images
```

Out of 400 total test images:

```text
(158 + 175) / 400 = 0.8325
```

So the final test accuracy is approximately:

```text
83.3%
```


## How to Run the Training Script


Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure the following files are in the same folder:

```text
cnn_models.py
metadata.csv
image_data.npy
```

Then run:

```bash
python3 cnn_models.py
```

The script will:

1. Load the image data and metadata
2. Split the training data into training and validation sets
3. Train the CNN model
4. Save the best model as `cnn_section2_best.keras`
5. Print the test and field evaluation reports

The saved model file will be:

```text
cnn_section2_best.keras
```

## How to Run the Streamlit App

Make sure the following files are in the same folder:

```text
app.py
cnn_section2_best.keras
```


Then run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser. You can upload a chest X-ray image, and the app will display:

- The uploaded image
- The predicted diagnosis
- The pneumonia probability

## Streamlit App Usage

After running:

```bash
streamlit run app.py
```

Upload a `.png`, `.jpg`, or `.jpeg` image.

The app will preprocess the image by:

1. Converting it to RGB
2. Resizing it to `64 x 64`
3. Normalizing pixel values to the range `0..1`
4. Passing it into the trained CNN model

The app then displays the prediction result.

Thanks! 