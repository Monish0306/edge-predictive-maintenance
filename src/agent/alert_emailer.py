"""
alert_emailer.py
================
Sends proactive HTML email alerts to the manager BEFORE machine failure occurs.
Attaches a PDF report with embedded sensor charts.

Usage:
    from src.agent.alert_emailer import AlertEmailer
    emailer = AlertEmailer()
    emailer.send_alert_email(alert_data, sensor_history, engine_id=47)
"""

import os
import json
import base64
import smtplib
import threading
import io
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/alert_config.json')

SEVERITY_COLORS = {
    'NORMAL':   '#10b981',
    'LOW':      '#84cc16',
    'MEDIUM':   '#f59e0b',
    'HIGH':     '#ef4444',
    'CRITICAL': '#7c3aed',
}

SEVERITY_EMOJI = {
    'NORMAL':   '✅',
    'LOW':      '🟡',
    'MEDIUM':   '🟠',
    'HIGH':     '🔴',
    'CRITICAL': '💀',
}


# ── ALERT EMAILER ──────────────────────────────────────────────────────────────
class AlertEmailer:
    """
    Sends rich HTML emails with embedded charts and PDF attachment
    to the manager when a HIGH or CRITICAL anomaly is detected.
    """

    def __init__(self):
        self.config = self._load_config()
        self._last_sent: dict[str, datetime] = {}  # engine_id → last sent time

    # ── PUBLIC API ─────────────────────────────────────────────────────────────
    def send_alert_email(self, alert_data: dict, prob_history: list,
                         sensor_importance: dict | None = None,
                         engine_id: int = 1, async_send: bool = True):
        """
        Send alert email. Respects cooldown. Can run async (non-blocking).

        Parameters
        ----------
        alert_data       : dict from MaintenanceAgent.analyze_anomaly()
        prob_history     : list of recent anomaly probabilities (for trend chart)
        sensor_importance: dict of sensor importance scores (optional)
        engine_id        : engine identifier
        async_send       : if True, send in background thread (default)
        """
        severity = alert_data.get('severity', 'NORMAL')
        min_sev  = self.config.get('min_severity_to_email', 'HIGH')
        sev_rank = {'NORMAL': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}

        if sev_rank.get(severity, 0) < sev_rank.get(min_sev, 3):
            return False  # below threshold

        engine_key = str(engine_id)
        cooldown_min = self.config.get('alert_cooldown_minutes', 15)
        last = self._last_sent.get(engine_key)
        if last and (datetime.now() - last) < timedelta(minutes=cooldown_min):
            return False  # in cooldown window

        if async_send:
            t = threading.Thread(
                target=self._do_send,
                args=(alert_data, prob_history, sensor_importance, engine_id),
                daemon=True
            )
            t.start()
        else:
            self._do_send(alert_data, prob_history, sensor_importance, engine_id)

        self._last_sent[engine_key] = datetime.now()
        return True

    # ── INTERNAL ───────────────────────────────────────────────────────────────
    def _load_config(self) -> dict:
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                'manager_email': 'monish0329@gmail.com',
                'sender_email': '',
                'sender_app_password': '',
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'alert_cooldown_minutes': 15,
                'min_severity_to_email': 'HIGH',
                'company_name': 'Edge AI Predictive Maintenance',
                'facility_name': 'NASA Turbofan Facility',
            }

    def _do_send(self, alert_data: dict, prob_history: list,
                 sensor_importance: dict | None, engine_id: int):
        """Builds and sends the email (runs in thread)."""
        try:
            # Build chart images
            trend_png    = self._make_trend_chart(prob_history, alert_data)
            sensor_png   = self._make_sensor_chart(sensor_importance) if sensor_importance else None

            # Build email
            msg = MIMEMultipart('related')
            msg['Subject'] = self._subject(alert_data, engine_id)
            msg['From']    = self.config.get('sender_email', '')
            msg['To']      = self.config.get('manager_email', '')

            # HTML body
            html_body = self._build_html(alert_data, engine_id, trend_png, sensor_png)
            msg.attach(MIMEText(html_body, 'html'))

            # PDF attachment
            pdf_bytes = self._build_pdf(alert_data, engine_id, prob_history, sensor_importance)
            if pdf_bytes:
                attachment = MIMEApplication(pdf_bytes, _subtype='pdf')
                attachment.add_header(
                    'Content-Disposition', 'attachment',
                    filename=f"alert_report_engine_{engine_id:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
                msg.attach(attachment)

            # Send
            self._smtp_send(msg)
            print(f"[AlertEmailer] ✅ Alert email sent to {self.config.get('manager_email')} for Engine #{engine_id:03d}")

        except Exception as exc:
            print(f"[AlertEmailer] ❌ Failed to send email: {exc}")

    def _subject(self, alert_data: dict, engine_id: int) -> str:
        sev   = alert_data.get('severity', 'UNKNOWN')
        emoji = SEVERITY_EMOJI.get(sev, '⚠️')
        prob  = alert_data.get('anomaly_probability', 0)
        return (f"{emoji} [{sev}] Engine #{engine_id:03d} Predictive Alert — "
                f"{prob:.0%} Failure Probability | Action Required")

    # ── CHARTS ─────────────────────────────────────────────────────────────────
    def _fig_to_png_b64(self, fig) -> str | None:
        """Export a Plotly figure to base64-encoded PNG."""
        try:
            png = pio.to_image(fig, format='png', width=700, height=320, scale=1.5)
            return base64.b64encode(png).decode('utf-8')
        except Exception as e:
            print(f"[AlertEmailer] Chart export failed: {e}")
            return None

    def _make_trend_chart(self, prob_history: list, alert_data: dict) -> str | None:
        if not prob_history:
            return None
        steps = list(range(len(prob_history)))
        prob  = alert_data.get('anomaly_probability', 0)

        color_map = []
        for v in prob_history:
            if v >= 0.9:   color_map.append('#7c3aed')
            elif v >= 0.7: color_map.append('#ef4444')
            elif v >= 0.5: color_map.append('#f59e0b')
            else:          color_map.append('#10b981')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=prob_history,
            mode='lines+markers',
            line=dict(color='#6366f1', width=2.5),
            marker=dict(size=5, color=color_map),
            name='Anomaly Probability'
        ))
        fig.add_hline(y=0.7, line_dash='dash', line_color='#ef4444',
                      annotation_text='HIGH threshold (0.70)', annotation_font_color='#ef4444')
        fig.add_hline(y=0.9, line_dash='dash', line_color='#7c3aed',
                      annotation_text='CRITICAL threshold (0.90)', annotation_font_color='#7c3aed')
        fig.add_annotation(
            x=steps[-1], y=prob, text=f"Now: {prob:.3f}",
            showarrow=True, arrowhead=2, arrowcolor='white',
            font=dict(color='white', size=11), bgcolor='#ef4444'
        )
        fig.update_layout(
            title='📈 Anomaly Probability Trend',
            paper_bgcolor='#1e1e2e', plot_bgcolor='#1e1e2e',
            font=dict(color='white'),
            xaxis=dict(title='Time Steps', gridcolor='#333'),
            yaxis=dict(title='Probability', range=[0, 1], gridcolor='#333'),
            height=320, margin=dict(l=50, r=30, t=50, b=40)
        )
        return self._fig_to_png_b64(fig)

    def _make_sensor_chart(self, sensor_importance: dict) -> str | None:
        if not sensor_importance:
            return None
        names  = [v['name'] for v in sensor_importance.values()]
        values = [v['importance_pct'] for v in sensor_importance.values()]
        colors = ['#ef4444' if v > 70 else '#f59e0b' if v > 40 else '#10b981' for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=names, orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in values], textposition='outside',
            textfont=dict(color='white')
        ))
        fig.update_layout(
            title='🗺️ Sensor Importance (Which sensors triggered alert)',
            paper_bgcolor='#1e1e2e', plot_bgcolor='#1e1e2e',
            font=dict(color='white'),
            xaxis=dict(title='Importance %', range=[0, 120], gridcolor='#333'),
            yaxis=dict(autorange='reversed'),
            height=320, margin=dict(l=150, r=30, t=50, b=40)
        )
        return self._fig_to_png_b64(fig)

    # ── HTML EMAIL BODY ─────────────────────────────────────────────────────────
    def _build_html(self, alert_data: dict, engine_id: int,
                    trend_png: str | None, sensor_png: str | None) -> str:
        sev        = alert_data.get('severity', 'UNKNOWN')
        prob       = alert_data.get('anomaly_probability', 0)
        root_cause = alert_data.get('root_cause', 'N/A')
        actions    = alert_data.get('recommended_actions', [])
        maint_date = alert_data.get('maintenance_schedule', 'N/A')
        downtime   = alert_data.get('estimated_downtime', 'N/A')
        cost_saved = alert_data.get('estimated_cost_saved', 'N/A')
        timestamp  = alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        badge_color  = SEVERITY_COLORS.get(sev, '#6366f1')
        company      = self.config.get('company_name', 'Edge AI PdM')
        facility     = self.config.get('facility_name', 'Turbofan Facility')
        actions_html = ''.join(f"<li style='margin:6px 0;'>{a}</li>" for a in actions[:6])

        trend_img  = f'<img src="data:image/png;base64,{trend_png}" style="width:100%;border-radius:8px;margin:10px 0;" />' if trend_png else ""
        sensor_img = f'<img src="data:image/png;base64,{sensor_png}" style="width:100%;border-radius:8px;margin:10px 0;" />' if sensor_png else ""

        # Determine urgency text
        if sev == 'CRITICAL':
            urgency = '⚡ IMMEDIATE ACTION REQUIRED — Failure is imminent!'
            border_color = '#7c3aed'
        else:
            urgency = '⚠️ Action required before maintenance deadline'
            border_color = '#ef4444'

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e2e8f0; margin: 0; padding: 20px; }}
  .container {{ max-width: 750px; margin: 0 auto; background: #1a1a2e; border-radius: 12px; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.5); }}
  .header {{ background: linear-gradient(135deg, #1e1e3f 0%, #12122b 100%); padding: 30px; border-bottom: 3px solid {border_color}; }}
  .header-title {{ font-size: 22px; font-weight: 700; margin: 0; color: #fff; }}
  .header-sub {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
  .badge {{ display: inline-block; background: {badge_color}; color: white; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 15px; letter-spacing: 1px; margin: 14px 0 0; }}
  .urgency {{ background: rgba(239,68,68,0.15); border-left: 4px solid {border_color}; padding: 14px 18px; margin: 20px; border-radius: 0 8px 8px 0; color: #fca5a5; font-weight: 600; }}
  .section {{ padding: 10px 24px; }}
  .section-title {{ font-size: 14px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; border-bottom: 1px solid #2d2d4e; padding-bottom: 8px; }}
  .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: #0f0f1a; border-radius: 8px; padding: 14px; border: 1px solid #2d2d4e; }}
  .kpi-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  th {{ background: #0f0f1a; padding: 10px; text-align: left; font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px; border-bottom: 1px solid #2d2d4e; font-size: 13px; }}
  .actions {{ background: #0f0f1a; border-radius: 8px; padding: 16px 18px 16px 36px; border: 1px solid #2d2d4e; }}
  .footer {{ background: #0f0f1a; padding: 18px 24px; text-align: center; font-size: 11px; color: #475569; border-top: 1px solid #1e293b; }}
  .prob-bar {{ background: #1e293b; border-radius: 10px; height: 8px; overflow: hidden; margin-top: 6px; }}
  .prob-fill {{ height: 100%; border-radius: 10px; background: {badge_color}; width: {min(prob*100, 100):.0f}%; }}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div class="header">
    <div class="header-title">⚙️ {company}</div>
    <div class="header-sub">{facility} · Alert Generated: {timestamp}</div>
    <div class="badge">{SEVERITY_EMOJI.get(sev, '⚠️')} {sev} ALERT</div>
  </div>

  <!-- URGENCY BANNER -->
  <div class="urgency">{urgency}</div>

  <!-- KPI GRID -->
  <div class="section">
    <div class="section-title">📊 Alert Summary</div>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Engine ID</div>
        <div class="kpi-value">#{engine_id:03d}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Anomaly Probability</div>
        <div class="kpi-value" style="color:{badge_color}">{prob:.1%}</div>
        <div class="prob-bar"><div class="prob-fill"></div></div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Maintenance Deadline</div>
        <div class="kpi-value" style="font-size:15px;">{maint_date}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Estimated Cost Saved</div>
        <div class="kpi-value" style="color:#10b981;font-size:15px;">{cost_saved}</div>
        <div class="kpi-sub">if prevented now</div>
      </div>
    </div>
  </div>

  <!-- FAULT DETAILS -->
  <div class="section">
    <div class="section-title">🔍 Fault Analysis</div>
    <table>
      <tr><th>Parameter</th><th>Details</th></tr>
      <tr><td><b>Alert Level</b></td><td><span style="color:{badge_color};font-weight:700;">{sev}</span></td></tr>
      <tr><td><b>Root Cause</b></td><td>{root_cause}</td></tr>
      <tr><td><b>Detection Timestamp</b></td><td>{timestamp}</td></tr>
      <tr><td><b>Estimated Downtime</b></td><td>{downtime}</td></tr>
      <tr><td><b>Act Before</b></td><td><b style="color:#f59e0b;">{maint_date}</b></td></tr>
    </table>
  </div>

  <!-- WHY THIS IS HAPPENING -->
  <div class="section">
    <div class="section-title">💡 Why Is This Happening?</div>
    <p style="font-size:14px;line-height:1.7;color:#cbd5e1;">
      The Edge AI model has detected a statistically significant anomaly pattern across multiple 
      sensor readings on Engine #{engine_id:03d}. The model is trained on NASA Turbofan engine 
      degradation data and uses a Dual-Head Transformer architecture to simultaneously detect 
      anomalies and estimate Remaining Useful Life (RUL).
      <br><br>
      <b style="color:#f59e0b;">Root Cause:</b> {root_cause}
      <br>
      The probability of failure ({prob:.1%}) has exceeded the alert threshold ({self.config.get('min_severity_to_email','HIGH')} = 70%). 
      Based on historical patterns, machines with this profile fail within the predicted maintenance window 
      without intervention. <b>Acting now prevents catastrophic unplanned downtime.</b>
    </p>
  </div>

  <!-- TREND CHART -->
  {f'<div class="section"><div class="section-title">📈 Anomaly Probability Trend</div>{trend_img}</div>' if trend_img else ""}

  <!-- SENSOR CHART -->
  {f'<div class="section"><div class="section-title">🗺️ Sensor Importance (What is Failing)</div>{sensor_img}</div>' if sensor_img else ""}

  <!-- RECOMMENDED ACTIONS -->
  <div class="section">
    <div class="section-title">🔧 Recommended Actions (in priority order)</div>
    <ol class="actions">{actions_html}</ol>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    📎 Full PDF report attached · Generated by Edge AI Predictive Maintenance System v2.0<br>
    Model: Dual-Head Transformer · Dataset: NASA Turbofan (FD001–FD004)<br>
    <i>This alert was auto-generated. Verify with qualified engineer before shutdown.</i>
  </div>
</div>
</body>
</html>"""

    # ── PDF REPORT ─────────────────────────────────────────────────────────────
    def _build_pdf(self, alert_data: dict, engine_id: int,
                   prob_history: list, sensor_importance: dict | None) -> bytes | None:
        """Build a professional PDF report using fpdf2."""
        try:
            from fpdf import FPDF
        except ImportError:
            return None

        try:
            sev       = alert_data.get('severity', 'UNKNOWN')
            prob      = alert_data.get('anomaly_probability', 0)
            root_cause= alert_data.get('root_cause', 'N/A')
            actions   = alert_data.get('recommended_actions', [])
            maint     = alert_data.get('maintenance_schedule', 'N/A')
            downtime  = alert_data.get('estimated_downtime', 'N/A')
            cost      = alert_data.get('estimated_cost_saved', 'N/A')
            ts        = alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            company   = self.config.get('company_name', 'Edge AI PdM')
            report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M')}-E{engine_id:03d}"

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # ── HEADER ──
            pdf.set_fill_color(15, 15, 26)
            pdf.rect(0, 0, 210, 35, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', 'B', 18)
            pdf.set_xy(10, 8)
            pdf.cell(0, 10, f'PREDICTIVE ALERT REPORT', ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_xy(10, 20)
            pdf.cell(0, 6, f'{company}  |  Report ID: {report_id}  |  {ts}', ln=True)

            pdf.set_y(40)
            pdf.set_text_color(30, 30, 30)

            # ── SEVERITY BANNER ──
            sev_rgb = {
                'HIGH': (239, 68, 68), 'CRITICAL': (124, 58, 237),
                'MEDIUM': (245, 158, 11), 'LOW': (132, 204, 22), 'NORMAL': (16, 185, 129)
            }.get(sev, (99, 102, 241))
            pdf.set_fill_color(*sev_rgb)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 12, f'  {SEVERITY_EMOJI.get(sev, "!")} ALERT LEVEL: {sev}  |  Engine #{engine_id:03d}  |  Probability: {prob:.1%}', ln=True, fill=True)
            pdf.ln(4)

            # ── SECTION: SUMMARY ──
            def section_header(title):
                pdf.set_fill_color(30, 30, 46)
                pdf.set_text_color(180, 180, 220)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 8, f'  {title}', ln=True, fill=True)
                pdf.set_text_color(30, 30, 30)
                pdf.ln(2)

            def row(label, value, bold_value=False):
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_fill_color(245, 245, 250)
                pdf.cell(65, 7, f'  {label}', border='B', fill=True)
                pdf.set_font('Helvetica', 'B' if bold_value else '', 10)
                pdf.cell(0, 7, f'  {value}', border='B', ln=True)

            section_header('ALERT SUMMARY')
            row('Engine ID', f'Engine #{engine_id:03d}')
            row('Alert Severity', sev, bold_value=True)
            row('Anomaly Probability', f'{prob:.4f} ({prob:.1%})', bold_value=True)
            row('Root Cause', root_cause)
            row('Detection Time', ts)
            row('Maintenance Deadline', maint, bold_value=True)
            row('Estimated Downtime', downtime)
            row('Cost Saved (if fixed)', cost, bold_value=True)
            pdf.ln(6)

            # ── SECTION: WHY ──
            section_header('WHY THIS IS HAPPENING')
            pdf.set_font('Helvetica', '', 10)
            why_texts = {
                'CRITICAL': f'The AI model is {prob:.0%} confident that Engine #{engine_id:03d} will fail imminently. '
                            f'Sensor patterns match historical pre-failure signatures in the NASA Turbofan dataset. '
                            f'STOP ENGINE if safe to do so. Catastrophic, unplanned failure is predicted without immediate action.',
                'HIGH':     f'Significant degradation detected with {prob:.0%} confidence. The engine shows an '
                            f'accelerating wear pattern consistent with "{root_cause}". '
                            f'Reduce operational load by 20% immediately and schedule emergency maintenance within 48 hours.',
                'MEDIUM':   f'Moderate anomaly with {prob:.0%} confidence. Early-stage degradation is developing. '
                            f'Schedule maintenance before {maint} to avoid escalation to critical stage.',
            }
            why = why_texts.get(sev, f'Anomaly probability {prob:.1%} has exceeded the alert threshold.')
            pdf.multi_cell(0, 7, why)
            pdf.ln(4)

            # ── SECTION: RECOMMENDED ACTIONS ──
            section_header('RECOMMENDED ACTIONS (Priority Order)')
            for i, action in enumerate(actions[:8], 1):
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_fill_color(250, 250, 255)
                pdf.cell(10, 7, f'{i}.', border='B', fill=True)
                pdf.set_font('Helvetica', '', 10)
                pdf.cell(0, 7, f' {action}', border='B', ln=True, fill=True)
            pdf.ln(6)

            # ── SECTION: SENSOR TABLE ──
            if sensor_importance:
                section_header('SENSOR IMPORTANCE ANALYSIS')
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_fill_color(30, 30, 46)
                pdf.set_text_color(200, 200, 255)
                pdf.cell(60, 8, '  Sensor', fill=True)
                pdf.cell(40, 8, '  Importance %', fill=True)
                pdf.cell(0, 8, '  Status', fill=True, ln=True)
                pdf.set_text_color(30, 30, 30)
                for sk, sv in list(sensor_importance.items())[:8]:
                    imp = sv.get('importance_pct', 0)
                    status = 'CRITICAL' if imp > 70 else 'HIGH' if imp > 50 else 'MODERATE' if imp > 30 else 'LOW'
                    pdf.set_font('Helvetica', '', 10)
                    pdf.set_fill_color(250, 250, 255) if list(sensor_importance.keys()).index(sk) % 2 == 0 else pdf.set_fill_color(240, 240, 250)
                    pdf.cell(60, 7, f'  {sv.get("name", sk)}', border='B', fill=True)
                    pdf.cell(40, 7, f'  {imp:.1f}%', border='B', fill=True)
                    pdf.cell(0, 7, f'  {status}', border='B', ln=True, fill=True)
                pdf.ln(6)

            # ── EMBED CHARTS ──
            # Save chart images to temp files for fpdf2
            import tempfile

            charts_added = False
            if prob_history and len(prob_history) > 1:
                trend_b64 = self._make_trend_chart(prob_history, alert_data)
                if trend_b64:
                    try:
                        png_bytes = base64.b64decode(trend_b64)
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            tmp.write(png_bytes)
                            tmp_path = tmp.name
                        pdf.add_page()
                        section_header('ANOMALY PROBABILITY TREND CHART')
                        pdf.image(tmp_path, x=10, w=185)
                        os.unlink(tmp_path)
                        charts_added = True
                    except Exception:
                        pass

            if sensor_importance:
                sensor_b64 = self._make_sensor_chart(sensor_importance)
                if sensor_b64:
                    try:
                        png_bytes = base64.b64decode(sensor_b64)
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            tmp.write(png_bytes)
                            tmp_path = tmp.name
                        if not charts_added:
                            pdf.add_page()
                        pdf.ln(10)
                        section_header('SENSOR IMPORTANCE CHART')
                        pdf.image(tmp_path, x=10, w=185)
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            # ── FOOTER on last page ──
            pdf.ln(8)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(120, 120, 140)
            pdf.multi_cell(0, 5, (
                'This report was auto-generated by Edge AI Predictive Maintenance System v2.0. '
                'Model: Dual-Head Transformer | Dataset: NASA Turbofan FD001-FD004. '
                'Verify findings with a qualified engineer before taking maintenance action.'
            ))

            return bytes(pdf.output())

        except Exception as exc:
            print(f"[AlertEmailer] PDF generation failed: {exc}")
            return None

    # ── SMTP ───────────────────────────────────────────────────────────────────
    def _smtp_send(self, msg: MIMEMultipart):
        host     = self.config.get('smtp_host', 'smtp.gmail.com')
        port     = self.config.get('smtp_port', 587)
        user     = self.config.get('sender_email', '')
        password = self.config.get('sender_app_password', '')

        if not user or not password or password == 'YOUR_APP_PASSWORD_HERE':
            raise ValueError(
                "Email not configured. Edit config/alert_config.json with your "
                "Gmail sender address and App Password."
            )

        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            server.login(user, password)
            server.send_message(msg)


# ── DEMO ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing AlertEmailer (will fail unless config/alert_config.json is set up)...")
    emailer = AlertEmailer()

    fake_alert = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_probability': 0.87,
        'severity': 'HIGH',
        'root_cause': 'Fan Speed: Fan bearing wear | Static Pressure: Seal leak',
        'recommended_actions': [
            '⚠️ IMMEDIATE INSPECTION REQUIRED within 48 hours!',
            'Reduce operational load by 20%.',
            '→ PRIORITY FIX: Lubricate or replace fan bearing',
            '→ PRIORITY FIX: Inspect and replace seals',
            'Order replacement: SKF 6205-2RS Bearing',
            'Notify maintenance team lead'
        ],
        'maintenance_schedule': '2026-03-18',
        'estimated_downtime': '1-2 days',
        'estimated_cost_saved': '$50,000-100,000',
    }
    fake_history = list(np.random.uniform(0.2, 0.5, 20)) + list(np.linspace(0.5, 0.87, 10))
    fake_sensors = {
        'sensor2':  {'name': 'Fan Speed',       'importance': 0.90, 'importance_pct': 90.0},
        'sensor7':  {'name': 'HPC Pressure',    'importance': 0.70, 'importance_pct': 70.0},
        'sensor11': {'name': 'Static Pressure', 'importance': 0.55, 'importance_pct': 55.0},
        'sensor3':  {'name': 'Core Speed',      'importance': 0.30, 'importance_pct': 30.0},
    }

    success = emailer.send_alert_email(fake_alert, fake_history, fake_sensors, engine_id=47, async_send=False)
    print(f"Email sent: {success}")
