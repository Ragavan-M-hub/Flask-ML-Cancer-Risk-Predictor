# Cancer Risk Prediction Web Application using Machine Learning

## 📌 Project Overview
This project is an end-to-end **Machine Learning–based Cancer Risk Prediction Web Application** built using **Python, Flask, and Random Forest Classifier**.  
The application predicts the **cancer risk level** based on selected health and environmental factors provided by the user through a web interface.

The model is trained on a structured dataset and deployed as a web application using **Flask**, allowing real-time predictions.

---

## 🎯 Objective
The primary objective of this project is to:
- Analyze critical health and environmental factors
- Train a machine learning model to classify cancer risk levels
- Deploy the trained model using a Flask web framework
- Provide prediction results along with class probabilities

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Random Forest Classifier
- **Reason for Selection:**
  - Handles non-linear relationships effectively
  - Robust against overfitting
  - Performs well on tabular medical data
- **Library:** scikit-learn

---

## 📊 Dataset Description
The dataset (`cancer.csv`) contains medical and lifestyle-related attributes influencing cancer risk.

### Selected Features:
- Air Pollution
- Genetic Risk
- Obesity
- Balanced Diet
- Occupational Hazards
- Coughing of Blood

### Target Variable:
- **Level** (Cancer risk category)

---

## ⚙️ Model Training Process
1. Data loaded using **Pandas**
2. Feature selection based on relevance
3. Dataset split into training and testing sets (70:30 ratio)
4. Model trained using Random Forest Classifier
5. Model evaluated using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

---

## 🌐 Web Application (Flask)
The Flask web application allows users to:
- Enter medical and lifestyle details via an HTML form
- Submit data for prediction
- View predicted cancer risk level
- View class-wise prediction probabilities

### Flask Routes:
- `/` → Home page
- `/predict` → Handles form submission and prediction logic

---

## 🖥️ Tech Stack Used
- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** scikit-learn
- **Data Handling:** Pandas
- **Frontend:** HTML (Jinja2 templates)
- **Model:** Random Forest Classifier

---

## 📁 Project Structure
- cancer-risk-prediction-flask-ml/
- │
- ├── app.py # Main Flask application
- ├── cancer.csv # Dataset
- ├── templates/
- │ └── index.html # HTML frontend
- └── README.md # Project documentation
