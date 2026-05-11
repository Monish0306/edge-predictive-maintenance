"""
Comprehensive test of all PredictiveMaintenance features.
"""
import sys, os
sys.path.insert(0, '.')
os.chdir(r'd:\PredictiveMaintenance')

import numpy as np

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

# ── Test 1: Import all modules ──
print("=" * 60)
print("TEST SUITE: PredictiveMaintenance Feature Verification")
print("=" * 60)

print("\n--- Module Imports ---")
test("import alert_emailer", lambda: __import__('src.agent.alert_emailer', fromlist=['AlertEmailer']))
test("import maintenance_agent", lambda: __import__('src.agent.maintenance_agent', fromlist=['MaintenanceAgent']))
test("import report_generator", lambda: __import__('src.agent.report_generator', fromlist=['ReportGenerator']))
test("import timeline", lambda: __import__('src.agent.timeline', fromlist=['predict_failure_timeline']))
test("import attention_extractor", lambda: __import__('src.model.attention_extractor', fromlist=['AttentionExtractor']))

# ── Test 2: MaintenanceAgent ──
print("\n--- MaintenanceAgent ---")
from src.agent.maintenance_agent import MaintenanceAgent
agent = MaintenanceAgent()

def test_agent_normal():
    sdict = {f'sensor{i+1}': 0.4 for i in range(15)}
    a = agent.analyze_anomaly(0.15, sdict, list(sdict.keys()))
    assert a['severity'] == 'NORMAL', f"Expected NORMAL, got {a['severity']}"

def test_agent_high():
    sdict = {f'sensor{i+1}': 0.85 for i in range(15)}
    a = agent.analyze_anomaly(0.82, sdict, list(sdict.keys()))
    assert a['severity'] == 'HIGH', f"Expected HIGH, got {a['severity']}"
    assert a['alert'] == True
    assert len(a['recommended_actions']) > 0

def test_agent_critical():
    sdict = {f'sensor{i+1}': 0.95 for i in range(15)}
    a = agent.analyze_anomaly(0.95, sdict, list(sdict.keys()))
    assert a['severity'] == 'CRITICAL', f"Expected CRITICAL, got {a['severity']}"

def test_agent_emailer_exists():
    assert agent.emailer is not None, "Emailer not loaded"

def test_agent_prob_history():
    assert len(agent.prob_history) > 0, "No prob history"

test("NORMAL severity (prob=0.15)", test_agent_normal)
test("HIGH severity (prob=0.82)", test_agent_high)
test("CRITICAL severity (prob=0.95)", test_agent_critical)
test("EmailAlerter integration", test_agent_emailer_exists)
test("Probability history tracking", test_agent_prob_history)

# ── Test 3: ReportGenerator ──
print("\n--- ReportGenerator ---")
from src.agent.report_generator import ReportGenerator, FAILURE_COSTS
from src.agent.timeline import predict_failure_timeline, get_timeline_milestones

rgen = ReportGenerator()

def test_report():
    tl = predict_failure_timeline(45, 0.78, 47)
    fsensors = {
        'sensor2': {'name': 'Fan Speed', 'importance': 0.9, 'importance_pct': 90.0},
        'sensor7': {'name': 'HPC Pressure', 'importance': 0.7, 'importance_pct': 70.0},
    }
    r = rgen.generate_report(47, 0.78, 'HIGH', 'Fan bearing wear', tl, fsensors,
                              ['Inspect fan bearing', 'Reduce load'])
    assert len(r) > 200, f"Report too short: {len(r)} chars"
    assert 'Engine #047' in r
    assert 'HIGH' in r

def test_failure_costs():
    assert FAILURE_COSTS['CRITICAL'] == 500000
    assert FAILURE_COSTS['HIGH'] == 150000

test("Generate maintenance report", test_report)
test("Failure costs table", test_failure_costs)

# ── Test 4: Timeline ──
print("\n--- Failure Timeline ---")

def test_timeline():
    tl = predict_failure_timeline(60, 0.65, 1)
    assert 'rul_days' in tl
    assert 'urgency' in tl
    assert 'predicted_failure_date' in tl
    assert tl['rul_days'] > 0

def test_milestones():
    tl = predict_failure_timeline(60, 0.65, 1)
    ms = get_timeline_milestones(tl['rul_days'])
    assert len(ms) > 0
    assert all('label' in m for m in ms)

test("Predict failure timeline", test_timeline)
test("Get timeline milestones", test_milestones)

# ── Test 5: AttentionExtractor ──
print("\n--- AttentionExtractor ---")
from src.model.attention_extractor import AttentionExtractor

def test_attention():
    aext = AttentionExtractor(num_sensors=15)
    data = np.random.randn(1, 30, 15).astype(np.float32)
    scores = aext.get_sensor_importance(data)
    assert len(scores) == 15
    top = list(scores.values())[0]
    assert 'name' in top
    assert 'importance_pct' in top

test("Sensor importance extraction", test_attention)

# ── Test 6: AlertEmailer ──
print("\n--- AlertEmailer ---")
from src.agent.alert_emailer import AlertEmailer

emailer = AlertEmailer()

def test_email_config():
    assert emailer.config is not None
    assert 'manager_email' in emailer.config
    assert emailer.config['manager_email'] == 'monish0329@gmail.com'

def test_pdf_generation():
    alert = {
        'timestamp': '2026-03-17 10:00:00', 'anomaly_probability': 0.85,
        'severity': 'HIGH', 'root_cause': 'Fan bearing wear',
        'recommended_actions': ['Inspect fan', 'Reduce load'],
        'maintenance_schedule': '2026-03-19', 'estimated_downtime': '1-2 days',
        'estimated_cost_saved': '$50,000'
    }
    fsensors = {
        'sensor2': {'name': 'Fan Speed', 'importance': 0.9, 'importance_pct': 90.0},
    }
    pdf = emailer._build_pdf(alert, 47, list(np.linspace(0.2, 0.85, 30)), fsensors)
    assert pdf is not None, "PDF is None"
    assert len(pdf) > 500, f"PDF too small: {len(pdf)} bytes"
    assert pdf[:4] == b'%PDF', "Not a valid PDF"

def test_html_email():
    alert = {
        'timestamp': '2026-03-17 10:00:00', 'anomaly_probability': 0.85,
        'severity': 'HIGH', 'root_cause': 'Fan bearing wear',
        'recommended_actions': ['Inspect fan', 'Reduce load'],
        'maintenance_schedule': '2026-03-19', 'estimated_downtime': '1-2 days',
        'estimated_cost_saved': '$50,000'
    }
    html = emailer._build_html(alert, 47, None, None)
    assert len(html) > 500
    assert 'HIGH' in html
    assert 'Engine #047' in html

def test_email_cooldown():
    alert = {
        'anomaly_probability': 0.25, 'severity': 'LOW',
        'root_cause': 'None', 'recommended_actions': [],
    }
    result = emailer.send_alert_email(alert, [0.25], engine_id=99, async_send=False)
    assert result == False, "LOW severity should not trigger email"

def test_email_unconfigured():
    """Email should fail gracefully when credentials are not configured."""
    alert = {
        'timestamp': '2026-03-17 10:00:00', 'anomaly_probability': 0.85,
        'severity': 'HIGH', 'root_cause': 'Fan bearing wear',
        'recommended_actions': ['Inspect fan'], 'maintenance_schedule': '2026-03-19',
        'estimated_downtime': '1 day', 'estimated_cost_saved': '$50k'
    }
    # This should return True because severity >= HIGH threshold
    result = emailer.send_alert_email(alert, [0.85], engine_id=999, async_send=False)
    assert result == True, "HIGH severity should try to send email"

test("Email config loaded", test_email_config)
test("PDF report generation", test_pdf_generation)
test("HTML email body", test_html_email)
test("Email cooldown (LOW severity blocked)", test_email_cooldown)
test("Email with unconfigured SMTP (graceful fail)", test_email_unconfigured)

# ── Test 7: ONNX Model ──
print("\n--- ONNX Model Inference ---")
import onnxruntime as ort

def test_onnx():
    p8 = 'models/onnx/model_int8_quantized.onnx'
    p32 = 'models/onnx/model_fp32.onnx'
    p = p8 if os.path.exists(p8) else p32
    session = ort.InferenceSession(p)
    iname = session.get_inputs()[0].name
    data = np.random.normal(0.5, 0.1, (1, 30, 15)).astype(np.float32)
    prob = float(session.run(None, {iname: data})[0][0])
    assert 0 <= prob <= 1, f"Prob out of range: {prob}"

test("ONNX model inference", test_onnx)

# ── Test 8: Dash app syntax ──
print("\n--- Dash App ---")

def test_dash_syntax():
    with open("dashboard/dash_app.py") as f:
        compile(f.read(), "dashboard/dash_app.py", "exec")

def test_css_exists():
    assert os.path.exists("dashboard/assets/custom.css"), "CSS missing"
    with open("dashboard/assets/custom.css") as f:
        css = f.read()
    assert 'overflow-y: auto' in css, "Scroll fix not applied"
    assert '--bg' in css, "CSS variables missing"

test("dash_app.py syntax check", test_dash_syntax)
test("custom.css with scroll fix", test_css_exists)

# ── Summary ──
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} test(s) FAILED!")
print("=" * 60)
