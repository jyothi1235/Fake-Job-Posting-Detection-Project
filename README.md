# 🕵️ Fake Job Posting Detection

## 📌 Project Overview

This project focuses on detecting **fraudulent job postings** using Machine Learning techniques.
It analyzes job-related text data such as descriptions, requirements, and company profiles to classify jobs as **Real or Fake**.

---

## 🎯 Objectives

* Identify fake job postings using ML models
* Perform text preprocessing and feature extraction
* Compare multiple classification algorithms
* Build a real-time prediction system using Streamlit

---

## 🌐 Live Demo

👉 **Try the app here:**
https://jyothi1235-fake-job-posting-detection-project-app-zbyonq.streamlit.app/

---

## 📂 Dataset

* Source: Fake Job Postings Dataset (Kaggle)
* Features used:

  * Title
  * Description
  * Requirements
  * Company Profile
* Target:

  * `fraudulent` (0 = Real, 1 = Fake)

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Streamlit

---

## 🔄 Workflow

1. Data Loading
2. Data Cleaning (handling missing values)
3. Text Preprocessing
4. Feature Extraction using TF-IDF
5. Handling Imbalanced Data
6. Model Training
7. Model Evaluation (Precision, Recall, F1-score)
8. Model Comparison
9. Deployment using Streamlit

---

## 🤖 Models Used

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)
* Gradient Boosting

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

> Note: F1-score and Recall are important due to imbalanced dataset.

---

## 🚀 Streamlit Application

### Features:

* User-friendly interface
* Input job details
* Real-time prediction (Fake / Real)
* Confidence score
* Example input auto-fill

### Run locally:

```bash
streamlit run app.py
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🧪 Example Prediction

Input:

```
Earn money quickly with no experience required
```

Output:

```
⚠️ Fake Job Posting
```

---

## ⚠️ Limitations

* Model depends on training data quality
* Cannot detect completely new fraud patterns
* Focuses mainly on textual features

---

## 🔮 Future Improvements

* Use Deep Learning (LSTM/BERT)
* Add keyword highlighting
* Improve UI and deployment
* Use real-time job data

---

## 👩‍💻 Author

Marreddy Jyothi Reddy

---

## ⭐ Conclusion

This project demonstrates how machine learning can effectively detect fraudulent job postings by analyzing textual patterns and improving decision-making in recruitment systems.
