# Task 5 – Data Science Example: Text Classification on Consumer Complaint Dataset

### Candidate: **Niranjana S**    
---

## 📘 Project Overview

This project focuses on **text classification** of the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) into the following **four categories**:

| Label | Category |
|-------|-----------|
| 0 | Credit reporting, repair, or other |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

The goal is to build a supervised machine learning model that classifies consumer complaints into these categories based on their text descriptions.

---

## 🧩 Steps Followed

### **1. Exploratory Data Analysis (EDA) and Feature Engineering**
- Loaded the dataset and explored data structure (columns, null values, sample entries).
- Analyzed the distribution of complaints across categories.
- Created new feature **`clean_text`** for preprocessed complaint messages.

### **2. Text Pre-Processing**
Performed text cleaning and normalization using Python NLP libraries:
- Lowercasing  
- Removing punctuation, stopwords, and special symbols  
- Lemmatization for better generalization  

### **3. Model Selection – Multi-Class Classification**
Trained multiple models for comparison:
- Logistic Regression  
- Naive Bayes  
- **Linear SVM (Selected as Best Performing Model)**  

### **4. Model Performance Comparison**
Evaluated each model using metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Linear SVM achieved the **highest accuracy** among the compared models.

### **5. Model Evaluation**
Generated a **confusion matrix** and calculated **weighted** and **macro averages** for performance validation.

### **6. Prediction**
Tested the final model on new complaint texts.  
The model successfully classified complaints into their respective categories.

---

## ⚙️ Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- nltk  
- re  
- string  

---

## 🧠 Model Used
**Linear Support Vector Machine (Linear SVM)** was chosen for final prediction due to its strong generalization capability on textual data.

---

## 📊 Results Summary

| Metric | Value (approx.) |
|--------|-----------------|
| Accuracy | ~0.85 |
| Precision (weighted) | 0.84 |
| Recall (weighted) | 0.85 |
| F1-score (macro avg) | 0.83 |

---

## 🖼️ Screenshots

All screenshots include **system date/time** and **name (Niranjana S)** as per submission requirements.

| Step | Description | File |
|------|--------------|------|
| 1 | Uploading Kaggle API Token | `images/01_kaggle_token.png` |
| 2 | Importing Dataset | `images/02_import_dataset.png` |
| 3 | Unzipped Dataset | `images/03_unzipped_dataset.png` |
| 4 | Dataframe Columns | `images/04_columns_output.png` |
| 5 | Cleaned Data Sample | `images/05_clean_text_sample.png` |
| 6 | Confusion Matrix (Linear SVM) | `images/06_confusion_matrix_svm.png` |
| 7 | Model Prediction Output (3 Complaints Example) | `images/07_prediction_output.png` |

---

## 🚀 How to Run (Google Colab)

1. Open the `.ipynb` file in **Google Colab**.  
2. Upload your **Kaggle API token** to access the dataset.  
3. Run all cells sequentially (Shift + Enter).  
4. The final section will display:
   - Cleaned data preview  
   - Model evaluation metrics  
   - Confusion matrix  
   - Predicted complaint categories  

---
## ✅ Conclusion

This project demonstrates the complete **data science workflow**  from text preprocessing to model evaluation and prediction  for categorizing consumer complaints into predefined classes using **Linear SVM**.  
The workflow is reproducible and can be easily extended to additional categories or larger datasets.

---
