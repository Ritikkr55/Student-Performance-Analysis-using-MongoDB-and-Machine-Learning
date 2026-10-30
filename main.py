"""
CSE494 - Intelligent NoSQL Databases | Academic Task 2
Project: Student Performance Analytics using MongoDB + PyMongo + Machine Learning

Run order:  python generate_dataset.py   ->   python main.py
Set MONGO_URI to use another server. Default: mongodb://localhost:27017
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")                       # save charts as files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymongo import ASCENDING, MongoClient
from pymongo.errors import WriteError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1. PYMONGO CONNECTION
# ----------------------------------------------------------------------
heading("1. CONNECTING TO MONGODB USING PYMONGO")
if os.getenv("USE_MOCK") == "1":            # only for testing without a server
    import mongomock
    client = mongomock.MongoClient()
else:
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                         serverSelectionTimeoutMS=5000)
    client.admin.command("ping")            # fails fast if server is down
db = client["student_performance_db"]
print("Connected. Database:", db.name)

# ----------------------------------------------------------------------
# 2. LOAD RAW JSON DATASET INTO MONGODB
# ----------------------------------------------------------------------
heading("2. LOADING RAW JSON DATASET")
db.drop_collection("students_raw")
db.drop_collection("students")
db.drop_collection("model_results")

with open("students.json") as f:
    raw_docs = json.load(f)
db.students_raw.insert_many(raw_docs)
print("Raw documents stored in 'students_raw':", db.students_raw.count_documents({}))

# ----------------------------------------------------------------------
# 3. DATA CLEANING & PREPROCESSING
# ----------------------------------------------------------------------
heading("3. DATA CLEANING & PREPROCESSING")
raw_df = pd.json_normalize(list(db.students_raw.find({}, {"_id": 0})))
print("Raw shape:", raw_df.shape)
print("\nMissing values before cleaning:")
print(raw_df.isnull().sum()[raw_df.isnull().sum() > 0])

df = raw_df.copy()

# (a) remove duplicate students
dups = df.duplicated(subset="student_id").sum()
df = df.drop_duplicates(subset="student_id").reset_index(drop=True)
print(f"\n(a) Duplicate rows removed: {dups}")

# (b) fix inconsistent text values (male, M, FEMALE, f ...)
df["gender"] = df["gender"].str.strip().str.lower().map(
    {"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
df["name"] = df["name"].str.strip().str.title()
print("(b) Gender standardised ->", df["gender"].value_counts().to_dict())

# (c) outliers: attendance must be 0-100, impossible values become missing
bad = (df["academics.attendance_pct"] > 100).sum()
df.loc[df["academics.attendance_pct"] > 100, "academics.attendance_pct"] = np.nan
print(f"(c) Impossible attendance values (>100) treated as missing: {bad}")

# (d) fill missing numeric values with the median
for col in ["academics.attendance_pct", "lifestyle.study_hours_per_week",
            "lifestyle.sleep_hours"]:
    med = df[col].median()
    n = df[col].isnull().sum()
    df[col] = df[col].fillna(med)
    print(f"(d) {col}: {n} missing values filled with median {med:.1f}")

# (e) convert types
df["age"] = df["age"].astype(int)
print("\nMissing values after cleaning:", int(df.isnull().sum().sum()))
print("Clean shape:", df.shape)


def to_document(row):
    """Rebuild the nested (embedded) MongoDB document from a flat row."""
    return {
        "student_id": row["student_id"],
        "name": row["name"],
        "gender": row["gender"],
        "age": int(row["age"]),
        "department": row["department"],
        "academics": {
            "attendance_pct": float(row["academics.attendance_pct"]),
            "previous_gpa": float(row["academics.previous_gpa"]),
            "assignment_score": float(row["academics.assignment_score"]),
            "midterm_score": float(row["academics.midterm_score"]),
            "final_score": float(row["academics.final_score"]),
        },
        "lifestyle": {
            "study_hours_per_week": float(row["lifestyle.study_hours_per_week"]),
            "sleep_hours": float(row["lifestyle.sleep_hours"]),
            "extracurricular": bool(row["lifestyle.extracurricular"]),
            "internet_access": bool(row["lifestyle.internet_access"]),
        },
        "result": row["result"],
    }


# ----------------------------------------------------------------------
# 4. SCHEMA VALIDATION + INDEXES
# ----------------------------------------------------------------------
heading("4. SCHEMA VALIDATION RULES & INDEXES")
num = ["int", "double"]
validator = {"$jsonSchema": {
    "bsonType": "object",
    "required": ["student_id", "name", "gender", "age", "department",
                 "academics", "lifestyle", "result"],
    "properties": {
        "student_id": {"bsonType": "string", "pattern": "^S[0-9]{4}$"},
        "name": {"bsonType": "string", "minLength": 2},
        "gender": {"enum": ["Male", "Female"]},
        "age": {"bsonType": "int", "minimum": 16, "maximum": 40},
        "department": {"enum": ["CSE", "ECE", "Mechanical", "Civil", "MBA"]},
        "academics": {
            "bsonType": "object",
            "required": ["attendance_pct", "previous_gpa", "final_score"],
            "properties": {
                "attendance_pct": {"bsonType": num, "minimum": 0, "maximum": 100},
                "previous_gpa": {"bsonType": num, "minimum": 0, "maximum": 10},
                "assignment_score": {"bsonType": num, "minimum": 0, "maximum": 100},
                "midterm_score": {"bsonType": num, "minimum": 0, "maximum": 100},
                "final_score": {"bsonType": num, "minimum": 0, "maximum": 100}}},
        "lifestyle": {
            "bsonType": "object",
            "properties": {
                "study_hours_per_week": {"bsonType": num, "minimum": 0, "maximum": 80},
                "sleep_hours": {"bsonType": num, "minimum": 0, "maximum": 24},
                "extracurricular": {"bsonType": "bool"},
                "internet_access": {"bsonType": "bool"}}},
        "result": {"enum": ["Pass", "Fail"]},
    }}}

if os.getenv("USE_MOCK") == "1":            # mongomock cannot enforce validators
    db.create_collection("students")
else:
    db.create_collection("students", validator=validator, validationLevel="strict",
                         validationAction="error")
print("Collection 'students' created with $jsonSchema validator.")

students = db.students
students.create_index([("student_id", ASCENDING)], unique=True)
students.create_index([("department", ASCENDING), ("result", ASCENDING)])
print("Indexes:", list(students.index_information().keys()))

# insert the cleaned documents
clean_docs = [to_document(r) for _, r in df.iterrows()]
students.insert_many(clean_docs)
print("Clean documents inserted:", students.count_documents({}))

# demonstrate that validation rejects bad data
bad_doc = to_document(df.iloc[0])
bad_doc["student_id"] = "S9999"
bad_doc["academics"]["attendance_pct"] = 150.0       # > 100
bad_doc["result"] = "Maybe"                          # not Pass/Fail
try:
    students.insert_one(bad_doc)
    print("WARNING: invalid document was accepted (validator not enforced)")
    students.delete_many({"student_id": "S9999"})       # clean up
except WriteError as e:
    print("Invalid document REJECTED by validation (error code", e.code, ")")

# ----------------------------------------------------------------------
# 5. CRUD OPERATIONS
# ----------------------------------------------------------------------
heading("5. CRUD OPERATIONS")
demo = lambda sid, name, att, fs, res: {
    "student_id": sid, "name": name, "gender": "Male", "age": 20,
    "department": "CSE",
    "academics": {"attendance_pct": att, "previous_gpa": 7.5,
                  "assignment_score": 75.0, "midterm_score": 70.0, "final_score": fs},
    "lifestyle": {"study_hours_per_week": 15.0, "sleep_hours": 7.0,
                  "extracurricular": True, "internet_access": True},
    "result": res}

print("--- CREATE ---")
r1 = students.insert_one(demo("S9001", "Test Student One", 80.0, 62.0, "Pass"))
print("insert_one  -> id:", r1.inserted_id)
r2 = students.insert_many([demo("S9002", "Test Student Two", 70.0, 45.0, "Fail"),
                           demo("S9003", "Test Student Three", 60.0, 40.0, "Fail")])
print("insert_many -> inserted:", len(r2.inserted_ids))

print("\n--- READ ---")
print("find_one:", students.find_one({"student_id": "S9001"},
                                      {"_id": 0, "student_id": 1, "name": 1, "result": 1}))
print("Count of CSE students:", students.count_documents({"department": "CSE"}))
print("\nTop 5 students with attendance >= 90 and score >= 70, sorted by score:")
cur = students.find({"academics.attendance_pct": {"$gte": 90},
                     "academics.final_score": {"$gte": 70}},
                    {"_id": 0, "student_id": 1, "department": 1,
                     "academics.attendance_pct": 1, "academics.final_score": 1}
                    ).sort("academics.final_score", -1).limit(5)
for d in cur:
    print(d)

print("\n--- UPDATE ---")
u1 = students.update_one({"student_id": "S9002"},
                         {"$set": {"academics.final_score": 52.0, "result": "Pass"}})
print("update_one  (S9002 re-evaluated) -> modified:", u1.modified_count)
u2 = students.update_many({"student_id": {"$in": ["S9001", "S9002", "S9003"]}},
                          {"$inc": {"academics.attendance_pct": 5}})
print("update_many (+5 attendance)      -> modified:", u2.modified_count)
u3 = students.update_one({"student_id": "S9004"},
                         {"$set": demo("S9004", "Upsert Student", 90.0, 85.0, "Pass")},
                         upsert=True)
print("upsert (new student S9004)       -> upserted id:", u3.upserted_id)
print("After update:", students.find_one({"student_id": "S9002"},
                                         {"_id": 0, "student_id": 1, "academics": 1, "result": 1}))

print("\n--- DELETE ---")
d1 = students.delete_one({"student_id": "S9003"})
print("delete_one  -> deleted:", d1.deleted_count)
d2 = students.delete_many({"student_id": {"$regex": "^S90"}})
print("delete_many (remaining demo students) -> deleted:", d2.deleted_count)
print("Documents in collection after CRUD demo:", students.count_documents({}))

# ----------------------------------------------------------------------
# 6. AGGREGATION PIPELINES
# ----------------------------------------------------------------------
heading("6. AGGREGATION PIPELINES")

print("--- Pipeline 1: Department-wise average score and pass percentage ---")
p1 = [{"$group": {"_id": "$department",
                  "students": {"$sum": 1},
                  "avg_score": {"$avg": "$academics.final_score"},
                  "avg_attendance": {"$avg": "$academics.attendance_pct"},
                  "passed": {"$sum": {"$cond": [{"$eq": ["$result", "Pass"]}, 1, 0]}}}},
      {"$project": {"students": 1, "passed": 1,
                    "avg_score": 1, "avg_attendance": 1,
                    "pass_pct": {"$multiply": [{"$divide": ["$passed", "$students"]}, 100]}}},
      {"$sort": {"avg_score": -1}}]
agg1 = pd.DataFrame(students.aggregate(p1)).rename(columns={"_id": "department"}).round(2)
print(agg1.to_string(index=False))

print("\n--- Pipeline 2: Top 5 students by final score ---")
p2 = [{"$sort": {"academics.final_score": -1}}, {"$limit": 5},
      {"$project": {"_id": 0, "student_id": 1, "name": 1, "department": 1,
                    "final_score": "$academics.final_score"}}]
print(pd.DataFrame(students.aggregate(p2)).to_string(index=False))

print("\n--- Pipeline 3: Attendance band vs average score ($bucket) ---")
p3 = [{"$bucket": {"groupBy": "$academics.attendance_pct",
                   "boundaries": [0, 60, 75, 90, 101],
                   "default": "Other",
                   "output": {"students": {"$sum": 1},
                              "avg_score": {"$avg": "$academics.final_score"}}}}]
agg3 = pd.DataFrame(students.aggregate(p3))
agg3["_id"] = agg3["_id"].map({0: "Below 60%", 60: "60-75%", 75: "75-90%", 90: "90%+"})
agg3["avg_score"] = agg3["avg_score"].round(2)
print(agg3.rename(columns={"_id": "attendance_band"}).to_string(index=False))

print("\n--- Pipeline 4: Effect of extracurricular activities ---")
p4 = [{"$group": {"_id": "$lifestyle.extracurricular",
                  "students": {"$sum": 1},
                  "avg_score": {"$avg": "$academics.final_score"}}},
      {"$sort": {"_id": 1}}]
agg4 = pd.DataFrame(students.aggregate(p4)).rename(columns={"_id": "extracurricular"})
agg4["avg_score"] = agg4["avg_score"].round(2)
print(agg4.to_string(index=False))

print("\n--- Pipeline 5: Fail students by department with low attendance ($match + $group) ---")
p5 = [{"$match": {"result": "Fail", "academics.attendance_pct": {"$lt": 75}}},
      {"$group": {"_id": "$department", "at_risk_students": {"$sum": 1},
                  "avg_study_hours": {"$avg": "$lifestyle.study_hours_per_week"}}},
      {"$sort": {"at_risk_students": -1}}]
agg5 = pd.DataFrame(students.aggregate(p5)).rename(columns={"_id": "department"})
agg5["avg_study_hours"] = agg5["avg_study_hours"].round(2)
print(agg5.to_string(index=False))

try:
    plan = students.find({"department": "CSE", "result": "Fail"}).explain()
    print("\nexplain() winning plan stage:",
          plan["queryPlanner"]["winningPlan"].get("inputStage", plan["queryPlanner"]["winningPlan"]).get("stage"))
except Exception:
    pass

# ----------------------------------------------------------------------
# 7. LOAD DATA FROM MONGODB FOR ML + VISUALISATION
# ----------------------------------------------------------------------
heading("7. EXPLORATORY DATA ANALYSIS & VISUALISATIONS")
data = pd.json_normalize(list(students.find({}, {"_id": 0})))
data.columns = [c.split(".")[-1] for c in data.columns]
data["extracurricular"] = data["extracurricular"].astype(int)
data["internet_access"] = data["internet_access"].astype(int)
print("Data read from MongoDB:", data.shape)
print(data.describe().round(2).T[["mean", "std", "min", "max"]])

sns.set_theme(style="whitegrid")

plt.figure(figsize=(5, 4))
sns.countplot(x="result", data=data, palette=["#2a9d8f", "#e76f51"], order=["Pass", "Fail"])
plt.title("Pass vs Fail Count"); plt.tight_layout()
plt.savefig("outputs/1_pass_fail_count.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
sns.barplot(x="department", y="final_score", data=data, palette="viridis", errorbar=None)
plt.title("Average Final Score by Department"); plt.tight_layout()
plt.savefig("outputs/2_dept_avg_score.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="attendance_pct", y="final_score", hue="result", data=data,
                palette={"Pass": "#2a9d8f", "Fail": "#e76f51"}, alpha=0.7)
plt.title("Attendance vs Final Score"); plt.tight_layout()
plt.savefig("outputs/3_attendance_vs_score.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.regplot(x="study_hours_per_week", y="final_score", data=data,
            scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
plt.title("Study Hours vs Final Score"); plt.tight_layout()
plt.savefig("outputs/4_study_vs_score.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(data.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("outputs/5_correlation_heatmap.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data["final_score"], bins=25, kde=True, color="#264653")
plt.axvline(50, color="red", linestyle="--", label="Pass mark = 50"); plt.legend()
plt.title("Distribution of Final Scores"); plt.tight_layout()
plt.savefig("outputs/6_score_distribution.png", dpi=150); plt.close()
print("Charts 1-6 saved in outputs/")

# ----------------------------------------------------------------------
# 8. MACHINE LEARNING - PASS / FAIL CLASSIFICATION
# ----------------------------------------------------------------------
heading("8. MACHINE LEARNING (CLASSIFICATION: PASS / FAIL)")
# final_score is NOT used as a feature (it would leak the answer)
X = data.drop(columns=["student_id", "name", "final_score", "result"])
X = pd.get_dummies(X, columns=["gender", "department"], drop_first=True)
y = (data["result"] == "Pass").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results, matrices = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV Accuracy (5-fold)": cv})
    matrices[name] = confusion_matrix(y_test, pred)

res_df = pd.DataFrame(results).round(4)
print("\nModel comparison:\n", res_df.to_string(index=False))
res_df.to_csv("outputs/model_comparison.csv", index=False)
best = res_df.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
print("\nBest model (by F1-Score):", best)

# store results back in MongoDB
db.model_results.insert_many(res_df.to_dict("records"))
print("Evaluation metrics saved in MongoDB collection 'model_results'.")

# ML charts
ax = res_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
    kind="bar", figsize=(9, 5), colormap="Set2")
ax.set_ylim(0.5, 1.05); plt.xticks(rotation=0); plt.title("Model Performance Comparison")
plt.tight_layout(); plt.savefig("outputs/7_model_comparison.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, cm) in zip(axes, matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("outputs/8_confusion_matrices.png", dpi=150); plt.close()

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5)); imp.plot(kind="barh", color="#2a9d8f")
plt.title("Random Forest - Feature Importance"); plt.tight_layout()
plt.savefig("outputs/9_feature_importance.png", dpi=150); plt.close()
print("Charts 7-9 saved in outputs/")

heading("PROJECT EXECUTED SUCCESSFULLY")
client.close()
