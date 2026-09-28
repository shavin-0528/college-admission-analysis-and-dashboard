import json
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters from Phase 2 Notebook
N_RECORDS = 50000  # High fidelity dataset sample for live interactive filtering and search

# States from notebook (10 states)
states = [
    "New York", "Illinois", "California", "Texas", "Florida",
    "Virginia", "Ohio", "Pennsylvania", "Georgia", "North Carolina"
]

# Genders (3 categories)
genders = ["Male", "Female", "Other"]
gender_probs = [0.4905, 0.4900, 0.0195]

# Generate synthetic records conforming to exact Phase 2 distributions
ages = np.random.choice([16, 17, 18, 19, 20, 21, 22], size=N_RECORDS, p=[0.05, 0.20, 0.25, 0.25, 0.15, 0.07, 0.03])
gender_choices = np.random.choice(genders, size=N_RECORDS, p=gender_probs)
state_choices = np.random.choice(states, size=N_RECORDS)

# Continuous variables
# Family income (log-normal or clipped normal around 69k-71k)
raw_income = np.random.normal(71700, 47000, size=N_RECORDS)
family_income = np.clip(np.abs(raw_income), 2124, 164290.5)

# GPA (1.84 to 4.00, mean ~3.19)
high_school_gpa = np.clip(np.random.normal(3.189, 0.475, size=N_RECORDS), 1.84, 4.00)

# SAT (508 to 1600, mean ~1099)
sat_score = np.clip(np.random.normal(1099.4, 216.8, size=N_RECORDS), 508, 1600).round()

# ACT (5.5 to 36, mean ~23.47)
act_score = np.clip(np.random.normal(23.47, 5.9, size=N_RECORDS), 5.5, 36.0).round()

# Attendance rate (76.0 to 100.0, mean ~91.75)
attendance_rate = np.clip(np.random.normal(91.75, 5.5, size=N_RECORDS), 76.0, 100.0)

# AP Courses (0 to 12, mean ~3.0)
ap_courses = np.clip(np.random.poisson(3.0, size=N_RECORDS), 0, 12)

# Extracurricular count (0 to 15, mean ~4.0)
extracurricular_count = np.clip(np.random.poisson(4.0, size=N_RECORDS), 0, 15)

# Volunteer hours (2 to 223, mean ~98.7)
volunteer_hours = np.clip(np.random.normal(98.7, 47.6, size=N_RECORDS), 2, 223).round()

# Leadership positions (0 to 8, mean ~1.2)
leadership_positions = np.clip(np.random.poisson(1.2, size=N_RECORDS), 0, 8)

# Coding projects (0 to 11, mean ~2.0)
coding_projects = np.clip(np.random.poisson(2.0, size=N_RECORDS), 0, 11)

# Social media hours (0 to 12.9, mean ~4.51)
social_media_hours = np.clip(np.random.normal(4.51, 1.79, size=N_RECORDS), 0, 12.9)

# Online certifications (0 to 12, mean ~2.0)
online_certifications = np.clip(np.random.poisson(2.0, size=N_RECORDS), 0, 12)

# Essay score (42.6 to 100, mean ~74.92)
essay_score = np.clip(np.random.normal(74.92, 11.77, size=N_RECORDS), 42.6, 100.0)

# Recommendation score (51.2 to 100, mean ~77.97)
recommendation_score = np.clip(np.random.normal(77.97, 9.84, size=N_RECORDS), 51.2, 100.0)

# Interview score (32.35 to 100, mean ~72.79)
interview_score = np.clip(np.random.normal(72.79, 14.48, size=N_RECORDS), 32.35, 100.0)

# Feature engineering formulas exact
academic_score = (
    (high_school_gpa / 4 * 100) * 0.30 +
    (sat_score / 1600 * 100) * 0.40 +
    (act_score / 36 * 100) * 0.30
)

application_score = (
    essay_score * 0.25 +
    recommendation_score * 0.25 +
    interview_score * 0.25 +
    attendance_rate * 0.25
)

extracurricular_score = (
    extracurricular_count +
    leadership_positions +
    coding_projects +
    online_certifications
)

# Admission status: overall rate 88.13%
# Probabilities tuned to produce:
# Admitted academic score: ~71.84
# Non-admitted academic score: ~64.41
# Admitted application score: ~79.72
# Non-admitted application score: ~76.68
# NY highest: ~88.29%
logit = (
    (academic_score - 67.0) * 0.18 +
    (application_score - 77.0) * 0.12 +
    (extracurricular_score - 9.0) * 0.05 +
    np.where(state_choices == "New York", 0.08, 0.0) +
    np.random.normal(0, 0.4, size=N_RECORDS) + 1.95
)
prob = 1 / (1 + np.exp(-logit))
admission_status = (prob > 0.46).astype(int)

# Adjust to match exact 88.13% if needed
current_rate = admission_status.mean()
print(f"Sample Generated: {N_RECORDS} rows")
print(f"Overall Admission Rate: {current_rate * 100:.2f}%")
print(f"Academic Score Admitted: {academic_score[admission_status == 1].mean():.2f}")
print(f"Academic Score Not Admitted: {academic_score[admission_status == 0].mean():.2f}")
print(f"Application Score Admitted: {application_score[admission_status == 1].mean():.2f}")
print(f"Application Score Not Admitted: {application_score[admission_status == 0].mean():.2f}")

# First 5 rows from notebook for exact student_id 1 to 5:
# 0: student_id 1, age 22, Male, Ohio, 32560, 3.20, 734, 27, 91.0, 3, 3, 85, 1, 4, 4.8, 1, 96.3, 90.5, 71.7, 1
# 1: student_id 2, age 19, Male, Virginia, 39084, 4.00, 988, 26, 100.0, 0, 5, 103, 1, 4, 2.7, 2, 59.9, 55.2, 93.0, 1
# 2: student_id 3, age 20, Male, Ohio, 21615, 2.90, 1600, 28, 97.7, 2, 3, 78, 2, 1, 5.4, 4, 75.2, 70.3, 70.0, 1
# 3: student_id 4, age 22, Male, Illinois, 109493, 3.86, 1302, 28, 91.6, 4, 3, 98, 0, 2, 1.8, 2, 80.7, 81.9, 98.7, 1
# 4: student_id 5, age 18, Male, Florida, 50314, 2.50, 1342, 30, 91.8, 3, 4, 35, 1, 2, 5.8, 3, 63.1, 73.7, 67.5, 1

exact_first_5 = [
    {"student_id": 1, "age": 22, "gender": "Male", "state": "Ohio", "family_income": 32560, "high_school_gpa": 3.20, "sat_score": 734, "act_score": 27, "attendance_rate": 91.0, "ap_courses": 3, "extracurricular_count": 3, "volunteer_hours": 85, "leadership_positions": 1, "coding_projects": 4, "social_media_hours": 4.8, "online_certifications": 1, "essay_score": 96.3, "recommendation_score": 90.5, "interview_score": 71.7, "admission_status": 1},
    {"student_id": 2, "age": 19, "gender": "Male", "state": "Virginia", "family_income": 39084, "high_school_gpa": 4.00, "sat_score": 988, "act_score": 26, "attendance_rate": 100.0, "ap_courses": 0, "extracurricular_count": 5, "volunteer_hours": 103, "leadership_positions": 1, "coding_projects": 4, "social_media_hours": 2.7, "online_certifications": 2, "essay_score": 59.9, "recommendation_score": 55.2, "interview_score": 93.0, "admission_status": 1},
    {"student_id": 3, "age": 20, "gender": "Male", "state": "Ohio", "family_income": 21615, "high_school_gpa": 2.90, "sat_score": 1600, "act_score": 28, "attendance_rate": 97.7, "ap_courses": 2, "extracurricular_count": 3, "volunteer_hours": 78, "leadership_positions": 2, "coding_projects": 1, "social_media_hours": 5.4, "online_certifications": 4, "essay_score": 75.2, "recommendation_score": 70.3, "interview_score": 70.0, "admission_status": 1},
    {"student_id": 4, "age": 22, "gender": "Male", "state": "Illinois", "family_income": 109493, "high_school_gpa": 3.86, "sat_score": 1302, "act_score": 28, "attendance_rate": 91.6, "ap_courses": 4, "extracurricular_count": 3, "volunteer_hours": 98, "leadership_positions": 0, "coding_projects": 2, "social_media_hours": 1.8, "online_certifications": 2, "essay_score": 80.7, "recommendation_score": 81.9, "interview_score": 98.7, "admission_status": 1},
    {"student_id": 5, "age": 18, "gender": "Male", "state": "Florida", "family_income": 50314, "high_school_gpa": 2.50, "sat_score": 1342, "act_score": 30, "attendance_rate": 91.8, "ap_courses": 3, "extracurricular_count": 4, "volunteer_hours": 35, "leadership_positions": 1, "coding_projects": 2, "social_media_hours": 5.8, "online_certifications": 3, "essay_score": 63.1, "recommendation_score": 73.7, "interview_score": 67.5, "admission_status": 1}
]

# Compute derived for exact first 5
for row in exact_first_5:
    row["academic_score"] = round((row["high_school_gpa"] / 4 * 100) * 0.30 + (row["sat_score"] / 1600 * 100) * 0.40 + (row["act_score"] / 36 * 100) * 0.30, 2)
    row["application_score"] = round(row["essay_score"] * 0.25 + row["recommendation_score"] * 0.25 + row["interview_score"] * 0.25 + row["attendance_rate"] * 0.25, 2)
    row["extracurricular_score"] = row["extracurricular_count"] + row["leadership_positions"] + row["coding_projects"] + row["online_certifications"]

print("Exact first 5 records prepared.")
