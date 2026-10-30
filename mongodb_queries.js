// CSE494 - Student Performance Analytics | MongoDB Shell (mongosh) queries
// Step 0 (normal terminal, not mongosh):
//   mongoimport --db student_performance_db --collection students_raw --file students.json --jsonArray
// Then open mongosh and run: load("mongodb_queries.js")   (or paste section by section)

use("student_performance_db");

// ---------- 1. RAW DATA ----------
print("Raw documents:", db.students_raw.countDocuments());   // 615

// ---------- 2. CLEANING ----------
db.students_raw.aggregate([                                   // (a) remove duplicates
  { $group: { _id: "$student_id", doc: { $first: "$$ROOT" } } },
  { $replaceRoot: { newRoot: "$doc" } },
  { $out: "students_clean" }
]);
db.students_clean.updateMany({ gender: { $in: ["male", "M"] } },   { $set: { gender: "Male" } });    // (b)
db.students_clean.updateMany({ gender: { $in: ["FEMALE", "f"] } }, { $set: { gender: "Female" } });
db.students_clean.updateMany({ "academics.attendance_pct": { $gt: 100 } },                           // (c)
                             { $set: { "academics.attendance_pct": null } });
db.students_clean.updateMany({ "academics.attendance_pct": null },                                   // (d) medians
                             { $set: { "academics.attendance_pct": 79.1 } });
db.students_clean.updateMany({ "lifestyle.study_hours_per_week": null },
                             { $set: { "lifestyle.study_hours_per_week": 13.9 } });
db.students_clean.updateMany({ "lifestyle.sleep_hours": null },
                             { $set: { "lifestyle.sleep_hours": 6.8 } });
print("Clean documents:", db.students_clean.countDocuments());  // 600

// ---------- 3. VALIDATION + INDEXES ----------
db.students.drop();
db.createCollection("students", {
  validator: { $jsonSchema: {
    bsonType: "object",
    required: ["student_id","name","gender","age","department","academics","lifestyle","result"],
    properties: {
      student_id: { bsonType: "string", pattern: "^S[0-9]{4}$" },
      gender:     { enum: ["Male","Female"] },
      age:        { bsonType: "int", minimum: 16, maximum: 40 },
      department: { enum: ["CSE","ECE","Mechanical","Civil","MBA"] },
      academics: {
        bsonType: "object",
        required: ["attendance_pct","previous_gpa","final_score"],
        properties: {
          attendance_pct: { bsonType: ["int","double"], minimum: 0, maximum: 100 },
          previous_gpa:   { bsonType: ["int","double"], minimum: 0, maximum: 10 },
          final_score:    { bsonType: ["int","double"], minimum: 0, maximum: 100 }
        }
      },
      result: { enum: ["Pass","Fail"] }
    }
  }},
  validationLevel: "strict",
  validationAction: "error"
});
db.students.insertMany(db.students_clean.find({}, { _id: 0 }).toArray());
db.students.createIndex({ student_id: 1 }, { unique: true });
db.students.createIndex({ department: 1, result: 1 });
print("Validated documents:", db.students.countDocuments());    // 600

// This insert MUST fail ("Document failed validation")
try {
  db.students.insertOne({ student_id: "S9999", name: "Bad Data", gender: "Male", age: 20,
    department: "CSE", academics: { attendance_pct: 150, previous_gpa: 7, final_score: 60 },
    lifestyle: {}, result: "Maybe" });
} catch (e) { print("REJECTED by validation:", e.message); }

// ---------- 4. CRUD ----------
// CREATE
db.students.insertOne({ student_id: "S9001", name: "Test Student One", gender: "Male", age: 20,
  department: "CSE",
  academics: { attendance_pct: 80.0, previous_gpa: 7.5, assignment_score: 75.0, midterm_score: 70.0, final_score: 62.0 },
  lifestyle: { study_hours_per_week: 15.0, sleep_hours: 7.0, extracurricular: true, internet_access: true },
  result: "Pass" });
// READ
printjson(db.students.findOne({ student_id: "S9001" }));
printjson(db.students.find({ "academics.attendance_pct": { $gte: 90 }, "academics.final_score": { $gte: 70 } },
                           { _id: 0, student_id: 1, department: 1, "academics.final_score": 1 })
                     .sort({ "academics.final_score": -1 }).limit(5).toArray());
// UPDATE
db.students.updateOne({ student_id: "S9001" }, { $set: { "academics.final_score": 66.0 } });
db.students.updateMany({ student_id: "S9001" }, { $inc: { "academics.attendance_pct": 5 } });
printjson(db.students.findOne({ student_id: "S9001" }, { _id: 0, academics: 1 }));
// DELETE
db.students.deleteOne({ student_id: "S9001" });
print("After CRUD demo:", db.students.countDocuments());        // 600

// ---------- 5. AGGREGATION ----------
print("Pipeline 1: department-wise performance");
printjson(db.students.aggregate([
  { $group: { _id: "$department", students: { $sum: 1 },
              avg_score: { $avg: "$academics.final_score" },
              avg_attendance: { $avg: "$academics.attendance_pct" },
              passed: { $sum: { $cond: [{ $eq: ["$result", "Pass"] }, 1, 0] } } } },
  { $project: { students: 1, passed: 1,
                avg_score: { $round: ["$avg_score", 2] },
                avg_attendance: { $round: ["$avg_attendance", 2] },
                pass_pct: { $round: [{ $multiply: [{ $divide: ["$passed", "$students"] }, 100] }, 2] } } },
  { $sort: { avg_score: -1 } }
]).toArray());

print("Pipeline 2: top 5 students");
printjson(db.students.aggregate([
  { $sort: { "academics.final_score": -1 } }, { $limit: 5 },
  { $project: { _id: 0, student_id: 1, name: 1, department: 1, final_score: "$academics.final_score" } }
]).toArray());

print("Pipeline 3: attendance band vs average score");
printjson(db.students.aggregate([
  { $bucket: { groupBy: "$academics.attendance_pct", boundaries: [0, 60, 75, 90, 101], default: "Other",
               output: { students: { $sum: 1 }, avg_score: { $avg: "$academics.final_score" } } } }
]).toArray());

print("Pipeline 4: extracurricular effect");
printjson(db.students.aggregate([
  { $group: { _id: "$lifestyle.extracurricular", students: { $sum: 1 },
              avg_score: { $avg: "$academics.final_score" } } }
]).toArray());

print("Pipeline 5: at-risk students");
printjson(db.students.aggregate([
  { $match: { result: "Fail", "academics.attendance_pct": { $lt: 75 } } },
  { $group: { _id: "$department", at_risk_students: { $sum: 1 },
              avg_study_hours: { $avg: "$lifestyle.study_hours_per_week" } } },
  { $sort: { at_risk_students: -1 } }
]).toArray());

// ---------- 6. INDEX PERFORMANCE ----------
printjson(db.students.find({ department: "CSE", result: "Fail" }).explain("executionStats").queryPlanner.winningPlan);
