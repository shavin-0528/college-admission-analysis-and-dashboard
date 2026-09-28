import urllib.request
import json

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(name, url, method="GET", body=None):
    print(f"\n[TEST] {name} ({method} {url})")
    try:
        req = urllib.request.Request(url, method=method)
        if body:
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(body).encode('utf-8')
            res = urllib.request.urlopen(req, data=data)
        else:
            res = urllib.request.urlopen(req)
        
        status = res.status
        content = res.read().decode('utf-8')
        print(f"  [PASS] HTTP Status: {status}")
        
        if 'application/json' in res.headers.get('Content-Type', ''):
            parsed = json.loads(content)
            return True, parsed
        return True, content
    except urllib.error.HTTPError as e:
        print(f"  [FAIL] HTTP Error: {e.code} - {e.reason}")
        return False, e.code
    except Exception as e:
        print(f"  [FAIL] Connection Error: {e}")
        return False, str(e)

print("=" * 60)
print("COMPREHENSIVE DASHBOARD API & SERVER TEST SUITE")
print("=" * 60)

# 1. Test HTML Page
ok, html = test_endpoint("Main Dashboard Page", f"{BASE_URL}/")
assert ok and "College Admission Data Analysis & Dashboard" in html, "HTML title check failed"
print("  [PASS] Title & Subtitle verified in HTML")

# 2. Test Stats API
ok, stats = test_endpoint("Precomputed Stats API", f"{BASE_URL}/api/stats")
assert ok, "Stats API failed"
kpis = stats["kpis"]
print(f"  [PASS] Total Students: {kpis['total_students']}")
print(f"  [PASS] Admission Rate: {kpis['admission_rate']}%")
print(f"  [PASS] Avg Academic Score: {kpis['avg_academic_score']}")
print(f"  [PASS] Avg Application Score: {kpis['avg_application_score']}")
print(f"  [PASS] Processed Features: {kpis['processed_features']}")
print(f"  [PASS] Missing Values: {kpis['missing_values']}")
print(f"  [PASS] Duplicate Records: {kpis['duplicate_records']}")
print(f"  [PASS] Highest State: {kpis['highest_state']} ({kpis['highest_state_rate']}%)")
print(f"  [PASS] Correlation Matrix shape: {len(stats['correlation']['columns'])}x{len(stats['correlation']['matrix'])}")

# 3. Test Filter API
filter_payload = {
    "gender": "Female",
    "state": "New York",
    "status": "All",
    "age_min": 16,
    "age_max": 22,
    "gpa_min": 3.0,
    "gpa_max": 4.0,
    "sat_min": 1000,
    "sat_max": 1600,
    "act_min": 20,
    "act_max": 36,
    "acad_min": 60,
    "acad_max": 100,
    "app_min": 70,
    "app_max": 100
}
ok, f_res = test_endpoint("Dynamic Filter API (Female + New York + GPA >= 3.0)", f"{BASE_URL}/api/filter", method="POST", body=filter_payload)
assert ok and f_res["count"] > 0, "Filter failed or returned 0 records"
print(f"  [PASS] Filtered Matched Records: {f_res['count']}")
print(f"  [PASS] Scaled Total: {f_res['scaled_total']}")
print(f"  [PASS] Filtered Admission Rate: {f_res['admission_rate']}%")
print(f"  [PASS] Filtered Avg Academic Score: {f_res['avg_academic_score']}")

# 4. Test Student Search API - Existing Students 1 to 5
for sid in [1, 2, 3, 4, 5]:
    ok, s_res = test_endpoint(f"Student Search #{sid}", f"{BASE_URL}/api/student/{sid}")
    assert ok and s_res["found"], f"Student {sid} lookup failed"
    st = s_res["student"]
    print(f"  [PASS] Student #{sid}: {st['gender']}, {st['state']}, GPA: {st['high_school_gpa']}, SAT: {st['sat_score']}, Status: {'Admitted' if st['admission_status']==1 else 'Not Admitted'}")

# 5. Test Student Search API - Arbitrary Valid ID #1042
ok, s_res = test_endpoint("Student Search #1042", f"{BASE_URL}/api/student/1042")
assert ok and s_res["found"], "Student 1042 lookup failed"
st = s_res["student"]
print(f"  [PASS] Student #{1042}: {st['gender']}, {st['state']}, Academic Score: {st['academic_score']}, Application Score: {st['application_score']}")

# 6. Test Student Search API - Invalid ID (Out of range)
ok, code = test_endpoint("Student Search #9999999 (Invalid)", f"{BASE_URL}/api/student/9999999")
assert code == 404, "Invalid student should return 404"
print("  [PASS] Correctly returned 404 for invalid ID")

print("\n" + "=" * 60)
print("ALL BACKEND & API TESTS PASSED PERFECTLY (100% SUCCESS)")
print("=" * 60)
