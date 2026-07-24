# Intelligent Multi-Layered Phishing Detection Framework using Machine Learning and Behavioral Analysis

A machine learning-based web application that detects phishing **URLs** and **Emails** using a stacking ensemble model. The application provides real-time predictions, confidence scores, risk levels, and downloadable PDF reports through an intuitive Flask web interface.

---

## Project Overview

Phishing attacks are among the most common cybersecurity threats. This project provides a multi-layered phishing detection framework capable of identifying phishing attempts from:

- URL Analysis
- Email Content Analysis

The system integrates trained Machine Learning models into a Flask web application for real-time detection.

---

## Features

- URL Phishing Detection
- Email Phishing Detection
- Stacking Ensemble Machine Learning Models
- Confidence Score Display
- Risk Level Classification
- PDF Report Generation
- Responsive User Interface
- Error Handling and Input Validation

---

## Technology Stack

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Data Processing
- Pandas
- NumPy
- SciPy

### Frontend
- HTML5
- CSS3
- JavaScript
- Font Awesome

### Report Generation
- ReportLab

---

## Project Structure

```text
Intelligent-Multi-Layered-Phishing-Detection/
│
├── app.py
├── predictor.py
├── feature_engineering.py
├── config.py
├── requirements.txt
├── README.md
├── Technical_Report.md
│
├── models/
├── saved_features/
├── static/
├── templates/
├── screenshots/
└── .gitignore
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## Screenshots

Create a folder named **screenshots** and add images such as:

- Home Page
- URL Detection
- Email Detection
- Phishing Detection Result
- Legitimate Detection Result
- PDF Report

Example:

```
screenshots/
│
├── home.png
├── url_detection.png
├── email_detection.png
├── phishing_result.png
├── legitimate_result.png
└── pdf_report.png
```

---

## Models Used

### URL Detection
- Stacking Ensemble
- Lexical Feature Engineering

### Email Detection
- Stacking Ensemble
- TF-IDF
- Handcrafted Features

---

## Technical Documentation

Detailed implementation, model development, feature engineering, deployment workflow, and evaluation are available in:

**Technical_Report.md**

---

## Team Members

### Rohit Khokhar
- Flask Web Application Development
- Machine Learning Model Integration
- Frontend Development
- URL & Email Detection Interface
- Confidence Score & Risk Level Display
- PDF Report Generation
- Deployment & Testing

### Shankha Suvro Dutta
- Dataset Preparation
- Data Preprocessing
- Feature Engineering
- Machine Learning Model Development
- Hyperparameter Tuning
- Model Evaluation

### Ranjeet Kumar Pandey
- Literature Survey
- Documentation
- Final Report
- Presentation
- GitHub Documentation

---

## Future Enhancements

- Real-time Threat Intelligence Integration
- Domain Reputation Checking
- WHOIS Lookup
- Continuous Model Retraining
- Browser Extension
- API Deployment
- Dashboard Analytics

---

## License

This project was developed as part of the **RV University Summer Internship 2026** for academic and educational purposes.

---

## Acknowledgements

- RV University
- Internship Mentors
- UCI Machine Learning Repository
- Kaggle Datasets
- Scikit-learn Community