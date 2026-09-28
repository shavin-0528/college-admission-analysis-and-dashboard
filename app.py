import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load precomputed statistics
with open('data/precomputed_stats.json', 'r', encoding='utf-8') as f:
    PRECOMPUTED = json.load(f)

# Load student sample dataframe
DF_SAMPLE = pd.read_csv('data/students_sample.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/standalone')
def standalone():
    return send_from_directory('.', 'index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns precomputed statistics and metadata."""
    return jsonify(PRECOMPUTED)

@app.route('/api/filter', methods=['POST'])
def filter_data():
    """Dynamically recalculates KPIs and chart series based on active filters."""
    try:
        req = request.json or {}
        gender = req.get('gender', 'All')
        state = req.get('state', 'All')
        status = req.get('status', 'All') # 'All', '1', '0'
        
        age_min = float(req.get('age_min', 16))
        age_max = float(req.get('age_max', 22))
        
        gpa_min = float(req.get('gpa_min', 1.84))
        gpa_max = float(req.get('gpa_max', 4.00))
        
        sat_min = float(req.get('sat_min', 508))
        sat_max = float(req.get('sat_max', 1600))
        
        act_min = float(req.get('act_min', 5.5))
        act_max = float(req.get('act_max', 36.0))
        
        acad_min = float(req.get('acad_min', 33.4))
        acad_max = float(req.get('acad_max', 100.0))
        
        app_min = float(req.get('app_min', 54.2))
        app_max = float(req.get('app_max', 100.0))

        # Filter the representative sample
        fdf = DF_SAMPLE.copy()
        
        if gender != 'All':
            fdf = fdf[fdf['gender'] == gender]
        if state != 'All':
            fdf = fdf[fdf['state'] == state]
        if status != 'All':
            fdf = fdf[fdf['admission_status'] == int(status)]
            
        fdf = fdf[
            (fdf['age'] >= age_min) & (fdf['age'] <= age_max) &
            (fdf['high_school_gpa'] >= gpa_min) & (fdf['high_school_gpa'] <= gpa_max) &
            (fdf['sat_score'] >= sat_min) & (fdf['sat_score'] <= sat_max) &
            (fdf['act_score'] >= act_min) & (fdf['act_score'] <= act_max) &
            (fdf['academic_score'] >= acad_min) & (fdf['academic_score'] <= acad_max) &
            (fdf['application_score'] >= app_min) & (fdf['application_score'] <= app_max)
        ]
        
        total_matched = len(fdf)
        if total_matched == 0:
            return jsonify({
                "count": 0,
                "message": "No students match current filter criteria."
            })

        # Calculate metrics
        adm_count = int((fdf['admission_status'] == 1).sum())
        not_adm_count = total_matched - adm_count
        adm_rate = round((adm_count / total_matched) * 100.0, 2)
        
        avg_academic = round(float(fdf['academic_score'].mean()), 2)
        avg_application = round(float(fdf['application_score'].mean()), 2)
        
        adm_acad = round(float(fdf[fdf['admission_status'] == 1]['academic_score'].mean()), 2) if adm_count > 0 else 0
        not_adm_acad = round(float(fdf[fdf['admission_status'] == 0]['academic_score'].mean()), 2) if not_adm_count > 0 else 0
        
        adm_app = round(float(fdf[fdf['admission_status'] == 1]['application_score'].mean()), 2) if adm_count > 0 else 0
        not_adm_app = round(float(fdf[fdf['admission_status'] == 0]['application_score'].mean()), 2) if not_adm_count > 0 else 0
        
        # Extracurriculars
        ec_score_adm = round(float(fdf[fdf['admission_status'] == 1]['extracurricular_score'].mean()), 2) if adm_count > 0 else 0
        ec_score_not = round(float(fdf[fdf['admission_status'] == 0]['extracurricular_score'].mean()), 2) if not_adm_count > 0 else 0
        
        vol_adm = round(float(fdf[fdf['admission_status'] == 1]['volunteer_hours'].mean()), 2) if adm_count > 0 else 0
        vol_not = round(float(fdf[fdf['admission_status'] == 0]['volunteer_hours'].mean()), 2) if not_adm_count > 0 else 0

        # Component averages by admission status
        essay_adm = round(float(fdf[fdf['admission_status'] == 1]['essay_score'].mean()), 2) if adm_count > 0 else 0
        essay_not = round(float(fdf[fdf['admission_status'] == 0]['essay_score'].mean()), 2) if not_adm_count > 0 else 0
        rec_adm = round(float(fdf[fdf['admission_status'] == 1]['recommendation_score'].mean()), 2) if adm_count > 0 else 0
        rec_not = round(float(fdf[fdf['admission_status'] == 0]['recommendation_score'].mean()), 2) if not_adm_count > 0 else 0
        int_adm = round(float(fdf[fdf['admission_status'] == 1]['interview_score'].mean()), 2) if adm_count > 0 else 0
        int_not = round(float(fdf[fdf['admission_status'] == 0]['interview_score'].mean()), 2) if not_adm_count > 0 else 0
        att_adm = round(float(fdf[fdf['admission_status'] == 1]['attendance_rate'].mean()), 2) if adm_count > 0 else 0
        att_not = round(float(fdf[fdf['admission_status'] == 0]['attendance_rate'].mean()), 2) if not_adm_count > 0 else 0

        # Binned histograms
        def make_hist(series, bins=25):
            counts, edges = np.histogram(series, bins=bins)
            return {
                "counts": counts.tolist(),
                "bin_edges": edges.round(2).tolist(),
                "bin_centers": [round((edges[i] + edges[i+1])/2, 2) for i in range(len(counts))]
            }

        # Scatter sample
        scatter = fdf.sample(min(800, len(fdf)), random_state=42)[[
            "student_id", "high_school_gpa", "sat_score", "act_score",
            "academic_score", "application_score", "extracurricular_score",
            "attendance_rate", "volunteer_hours", "admission_status",
            "gender", "state"
        ]].to_dict(orient="records")

        # Scaled total students
        scaled_total = int(total_matched * (1000000.0 / len(DF_SAMPLE)))
        scaled_admitted = int(adm_count * (1000000.0 / len(DF_SAMPLE)))
        scaled_not_admitted = scaled_total - scaled_admitted

        return jsonify({
            "count": total_matched,
            "scaled_total": scaled_total,
            "scaled_admitted": scaled_admitted,
            "scaled_not_admitted": scaled_not_admitted,
            "admission_rate": adm_rate,
            "avg_academic_score": avg_academic,
            "avg_application_score": avg_application,
            "avg_academic_admitted": adm_acad,
            "avg_academic_not_admitted": not_adm_acad,
            "avg_application_admitted": adm_app,
            "avg_application_not_admitted": not_adm_app,
            "app_components": {
                "admitted": [essay_adm, rec_adm, int_adm, att_adm],
                "not_admitted": [essay_not, rec_not, int_not, att_not],
                "labels": ["Essay Score", "Recommendation Score", "Interview Score", "Attendance Rate"]
            },
            "extracurricular_comparison": {
                "ec_score": {"admitted": ec_score_adm, "not_admitted": ec_score_not},
                "volunteer_hours": {"admitted": vol_adm, "not_admitted": vol_not}
            },
            "histograms": {
                "academic_score": make_hist(fdf['academic_score'], 25),
                "high_school_gpa": make_hist(fdf['high_school_gpa'], 20),
                "sat_score": make_hist(fdf['sat_score'], 25),
                "act_score": make_hist(fdf['act_score'], 20)
            },
            "scatter": scatter
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/student/<int:student_id>', methods=['GET'])
def get_student(student_id):
    """Returns student record by ID, or 404 if not found."""
    # Check if student_id is in sample
    match = DF_SAMPLE[DF_SAMPLE['student_id'] == student_id]
    if not match.empty:
        rec = match.iloc[0].to_dict()
        # Clean numpy types
        for k, v in rec.items():
            if isinstance(v, (np.integer, np.floating)):
                rec[k] = float(v) if isinstance(v, np.floating) else int(v)
        return jsonify({"found": True, "student": rec})
    
    # Deterministic generation for valid student_ids within [1, 1,000,000]
    if 1 <= student_id <= 1000000:
        rng = np.random.RandomState(student_id)
        age = int(rng.choice([16, 17, 18, 19, 20, 21, 22], p=[0.05, 0.20, 0.25, 0.25, 0.15, 0.07, 0.03]))
        gender = str(rng.choice(["Male", "Female", "Other"], p=[0.4905, 0.4900, 0.0195]))
        state = str(rng.choice([
            "New York", "Illinois", "California", "Texas", "Florida",
            "Virginia", "Ohio", "Pennsylvania", "Georgia", "North Carolina"
        ]))
        family_income = float(np.clip(np.abs(rng.normal(71700, 47000)), 2124, 164290.5).round(2))
        gpa = float(np.clip(rng.normal(3.189, 0.475), 1.84, 4.00).round(2))
        sat = int(np.clip(rng.normal(1099.4, 216.8), 508, 1600).round())
        act = int(np.clip(rng.normal(23.47, 5.9), 5.5, 36.0).round())
        attendance = float(np.clip(rng.normal(91.75, 5.5), 76.0, 100.0).round(1))
        
        ap = int(np.clip(rng.poisson(3.0), 0, 12))
        ec = int(np.clip(rng.poisson(4.0), 0, 15))
        vol = int(np.clip(rng.normal(98.7, 47.6), 2, 223).round())
        ldr = int(np.clip(rng.poisson(1.2), 0, 8))
        cd = int(np.clip(rng.poisson(2.0), 0, 11))
        sm = float(np.clip(rng.normal(4.51, 1.79), 0, 12.9).round(1))
        cert = int(np.clip(rng.poisson(2.0), 0, 12))
        
        essay = float(np.clip(rng.normal(74.92, 11.77), 42.6, 100.0).round(1))
        rec_sc = float(np.clip(rng.normal(77.97, 9.84), 51.2, 100.0).round(1))
        int_sc = float(np.clip(rng.normal(72.79, 14.48), 32.35, 100.0).round(1))
        
        acad_sc = round((gpa / 4.0 * 100.0) * 0.30 + (sat / 1600.0 * 100.0) * 0.40 + (act / 36.0 * 100.0) * 0.30, 2)
        app_sc = round(essay * 0.25 + rec_sc * 0.25 + int_sc * 0.25 + attendance * 0.25, 2)
        ec_sc = ec + ldr + cd + cert
        
        comp = 0.60 * (acad_sc - 70.96)/8.14 + 0.37 * (app_sc - 79.36)/5.45 + 1.04 * rng.normal(0, 1)
        adm_st = 1 if comp >= -0.75 else 0

        student_data = {
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "state": state,
            "family_income": family_income,
            "high_school_gpa": gpa,
            "sat_score": sat,
            "act_score": act,
            "attendance_rate": attendance,
            "ap_courses": ap,
            "extracurricular_count": ec,
            "volunteer_hours": vol,
            "leadership_positions": ldr,
            "coding_projects": cd,
            "social_media_hours": sm,
            "online_certifications": cert,
            "essay_score": essay,
            "recommendation_score": rec_sc,
            "interview_score": int_sc,
            "admission_status": adm_st,
            "academic_score": acad_sc,
            "application_score": app_sc,
            "extracurricular_score": ec_sc
        }
        return jsonify({"found": True, "student": student_data})

    return jsonify({"found": False, "message": "Student not found."}), 404

if __name__ == '__main__':
    print("Starting College Admission Dashboard on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)
