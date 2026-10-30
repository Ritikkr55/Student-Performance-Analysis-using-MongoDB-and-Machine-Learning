"""Step 0: Create the JSON dataset (students.json).
The dataset is synthetic but built to behave like real student data:
it deliberately contains missing values, duplicates, inconsistent text and
outliers so that the data-cleaning step has real work to do."""
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

departments = ["CSE", "ECE", "Mechanical", "Civil", "MBA"]
first = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Karan", "Arjun", "Rahul",
         "Ananya", "Diya", "Priya", "Sneha", "Neha", "Simran", "Kavya", "Riya",
         "Harpreet", "Gurpreet", "Manpreet", "Jaspreet", "Amit", "Sumit", "Pooja", "Anjali"]
last = ["Sharma", "Singh", "Kaur", "Verma", "Gupta", "Mehta", "Bansal", "Jain",
        "Kapoor", "Malhotra", "Gill", "Sidhu", "Chopra", "Arora", "Bhatia"]

records = []
for i in range(1, 601):
    gender = random.choice(["Male", "Female"])
    dept = random.choice(departments)
    attendance = float(np.clip(np.random.normal(78, 14), 30, 100))
    study = float(np.clip(np.random.normal(14, 6), 1, 40))
    prev_gpa = float(np.clip(np.random.normal(7.0, 1.2), 3, 10))
    assignment = float(np.clip(np.random.normal(70, 15), 10, 100))
    midterm = float(np.clip(0.5 * prev_gpa * 10 + 0.3 * attendance + np.random.normal(0, 8), 5, 100))
    sleep = float(np.clip(np.random.normal(6.8, 1.2), 3, 10))
    extra = random.random() < 0.45
    internet = random.random() < 0.85

    final = (0.25 * midterm + 0.20 * assignment + 0.20 * attendance
             + 0.80 * study + 3.0 * prev_gpa + 1.0 * sleep
             + (2 if internet else 0) + np.random.normal(0, 9)) - 33
    final = float(np.clip(final, 0, 100))

    rec = {
        "student_id": f"S{i:04d}",
        "name": f"{random.choice(first)} {random.choice(last)}",
        "gender": gender,
        "age": random.randint(18, 25),
        "department": dept,
        "academics": {
            "attendance_pct": round(attendance, 1),
            "previous_gpa": round(prev_gpa, 2),
            "assignment_score": round(assignment, 1),
            "midterm_score": round(midterm, 1),
            "final_score": round(final, 1),
        },
        "lifestyle": {
            "study_hours_per_week": round(study, 1),
            "sleep_hours": round(sleep, 1),
            "extracurricular": extra,
            "internet_access": internet,
        },
        "result": "Pass" if final >= 50 else "Fail",
    }
    records.append(rec)

# ---- Make the data "dirty" on purpose ----
for r in random.sample(records, 30):
    r["academics"]["attendance_pct"] = None            # missing values
for r in random.sample(records, 30):
    r["lifestyle"]["study_hours_per_week"] = None
for r in random.sample(records, 25):
    r["lifestyle"]["sleep_hours"] = None
for r in random.sample(records, 40):                   # inconsistent text
    r["gender"] = random.choice(["male", "M", "FEMALE", "f"])
for r in random.sample(records, 5):                    # impossible values
    r["academics"]["attendance_pct"] = round(random.uniform(105, 130), 1)
records += random.sample(records, 15)                  # duplicate rows

with open("students.json", "w") as f:
    json.dump(records, f, indent=2)
print(f"students.json created with {len(records)} records")
