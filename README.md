# 🎓 Student Performance Analysis Using MongoDB and Machine Learning

## 📌 Project Overview

The **Student Performance Analysis System** is a data analysis and machine learning project that uses MongoDB, PyMongo, and Python to analyze student academic performance and predict examination results.

The project demonstrates database management, data cleaning, aggregation pipelines, indexing, and machine learning classification.

## 🎯 Project Objectives

- Store student academic records in MongoDB.
- Perform CRUD operations using PyMongo.
- Clean and validate the dataset.
- Identify duplicate student IDs.
- Analyze student performance using aggregation pipelines.
- Build and compare machine learning classification models.
- Evaluate models using standard performance metrics.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Database:** MongoDB
- **Database Connectivity:** PyMongo
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib

## 📂 Project Structure

```text
student-performance-analysis-mongodb-ml/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── student_performance.json
│
├── src/
│   ├── data_cleaning.py
│   ├── data_analysis.py
│   └── model_training.py
│
├── mongodb/
│   └── queries.js
│
├── results/
│   └── model_results.json
│
├── screenshots/
│   ├── collections.png
│   ├── indexes.png
│   └── aggregation_results.png
│
└── report/
    └── CSE494_Project_Report.docx
```

*Note: The structure above is a suggested organization. Replace these filenames with your actual project files.*

## 📊 Dataset Description

The dataset contains student information and academic performance attributes.

| Attribute | Description |
|---|---|
| `student_id` | Unique student identifier |
| `name` | Student name |
| `gender` | Student gender |
| `age` | Student age |
| `department` | Academic department |
| `attendance_pct` | Attendance percentage |
| `previous_gpa` | Previous GPA |
| `assignment_score` | Assignment marks |
| `midterm_score` | Midterm examination marks |
| `final_score` | Final examination marks |
| `study_hours_per_week` | Weekly study hours |
| `sleep_hours` | Daily sleep duration |
| `extracurricular` | Extracurricular participation |
| `internet_access` | Internet availability |
| `result` | Pass or Fail |

## 🗄️ MongoDB Database

**Database Name:** `student_performance_db`

The project uses the following collections:

| Collection | Description | Documents |
|---|---|---:|
| `students_raw` | Raw student records | 615 |
| `students` | Cleaned student records | 600 |
| `model_results` | Machine learning evaluation results | 3 |

### Database Operations

- Database and collection management
- CRUD operations
- Duplicate ID detection
- Data cleaning and validation
- Indexing for efficient queries
- Aggregation pipelines
- Department-wise performance analysis
- Pass/Fail distribution analysis

## 🤖 Machine Learning Models

Three classification algorithms were evaluated to predict student results.

1. Logistic Regression
2. Decision Tree
3. Random Forest

### 📈 Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 76.67% | 81.25% | 76.47% | 0.7879 | 0.8371 |
| Decision Tree | 67.50% | 71.01% | 72.06% | 0.7153 | 0.6531 |
| Random Forest | 75.00% | 77.14% | 79.41% | 0.7826 | 0.8289 |

**Best-performing model:** Logistic Regression achieved the highest accuracy and ROC-AUC among the three evaluated models.

*The results are based on the current experiment and may vary when the models are retrained.*

## 🔍 Key Findings

- The raw collection contains 615 documents, while the cleaned collection contains 600.
- The raw-data audit detected 15 duplicate student IDs.
- The cleaned collection contains 340 Pass records and 260 Fail records.
- CSE has the highest average final score among the five departments in the reported results.
- Logistic Regression achieved 76.67% accuracy in the current experiment.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed
- MongoDB installed and running
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-analysis-mongodb-ml.git
```

### Step 2: Open the Project Folder

```bash
cd student-performance-analysis-mongodb-ml
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start MongoDB

Ensure that your MongoDB service is running and that the connection settings in your Python scripts match your local setup.

### Step 5: Run the Project

Run the Python scripts in the order required by your implementation:

```bash
python src/data_cleaning.py
python src/data_analysis.py
python src/model_training.py
```

*Adjust the script names and execution order to match the files in your repository.*

## 📷 Screenshots

Screenshots of the MongoDB collections, database indexes, aggregation results, and machine learning evaluation results can be added here.

## 📄 Project Report

The project report documents the introduction, objectives, dataset, MongoDB implementation, data preprocessing, aggregation queries, machine learning models, evaluation results, and conclusion.

## ⚠️ Limitations

- The model results have not been validated on real institutional student records.
- The dataset's quality and representativeness affect the reliability of the predictions.
- The system is intended for academic demonstration and should not be used for high-stakes student decisions without further validation.

## 🚀 Future Enhancements

- Develop an interactive performance dashboard.
- Add more machine learning models.
- Improve data validation and error handling.
- Integrate the analysis into a web application.
- Evaluate the models on additional datasets.

## 👨‍💻 Author

**Ritik Kumar**

B.Tech Computer Science and Engineering

**Course Project:** CSE494

---

⭐ If you find this project useful, consider giving the repository a star!
