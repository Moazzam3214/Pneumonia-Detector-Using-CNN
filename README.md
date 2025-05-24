# 🫁 Pneumonia Detection Using Deep Learning

A web-based application to detect **Pneumonia** from chest X-ray images using a Convolutional Neural Network (CNN) with over 90% accuracy.

---


## 🧠 Model Overview

* **Architecture**: Custom CNN with 5 convolutional blocks and regularization layers
* **Frameworks**: TensorFlow, Keras
* **Accuracy**: Achieved 90%+ on the test dataset
* **Input**: Grayscale chest X-ray resized to `150x150`

---

## 📂 Files in This Repo

| File                                    | Description                                    |
| --------------------------------------- | ---------------------------------------------- |
| `app.py`                                | Streamlit app for real-time image prediction   |
| `pneumonia_detection_model.h5`          | Trained CNN model file                         |
| `pneumonia-detection-using-cnn-*.ipynb` | Full model training, evaluation & EDA notebook |
| `requirements.txt`                      | Requirements for the project                   |

---

## 🧪 Dataset

Chest X-ray dataset from **Kaggle**:

* Classes: `NORMAL`, `PNEUMONIA`
* Subsets: `train`, `val`, `test`
* Link: [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## ▶️ How to Run Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/pneumonia-detector.git
   cd pneumonia-detector
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**

   ```bash
   streamlit run app.py
   ```

---

## 📝 Notebook Highlights

* Data preprocessing (resizing, grayscale)
* Data augmentation to reduce overfitting
* Training using both **RMSprop** and **Adam**
* Accuracy >90% with balanced precision & recall

---

## 📦 Requirements

* `streamlit`
* `tensorflow`
* `opencv-python`
* `numpy`

You can install everything using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* Add support for multi-class lung disease classification
* Integrate with cloud for storage and larger model hosting
* Improve UI and mobile responsiveness

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---
