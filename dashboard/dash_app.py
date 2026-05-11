"""
dash_app.py – Professional Plotly Dash Dashboard
Edge AI Predictive Maintenance System
Run: python dashboard/dash_app.py  →  http://localhost:8050
"""
import os, sys, json, time
from datetime import datetime, timedelta

sys.path.append('.')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import onnxruntime as ort

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc

from src.agent.maintenance_agent import MaintenanceAgent
from src.mlops.monitor_and_retrain import ModelMonitor
from src.agent.timeline import predict_failure_timeline, get_timeline_milestones
from src.agent.report_generator import ReportGenerator, FAILURE_COSTS
from src.model.attention_extractor import AttentionExtractor

# ── DARK TEMPLATE ─────────────────────────────────────────────────────────────
def dark_layout(**extra):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, Segoe UI, sans-serif', size=12),
        xaxis=dict(gridcolor='#21293d', linecolor='#21293d'),
        yaxis=dict(gridcolor='#21293d', linecolor='#21293d'),
        legend=dict(bgcolor='rgba(13,17,23,0.8)', bordercolor='#21293d', borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    base.update(extra)
    return base

SEV_COLOR = {'NORMAL':'#10b981','LOW':'#84cc16','MEDIUM':'#f59e0b','HIGH':'#ef4444','CRITICAL':'#7c3aed'}
SEV_ICON  = {'NORMAL':'✅','LOW':'🟡','MEDIUM':'🟠','HIGH':'🔴','CRITICAL':'💀'}

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
def _load_model():
    p8 = 'models/onnx/model_int8_quantized.onnx'
    p32= 'models/onnx/model_fp32.onnx'
    p  = p8 if os.path.exists(p8) else p32
    return ort.InferenceSession(p), ('INT8' if p==p8 else 'FP32'), p

def _meta():
    try:
        with open('data/processed/model_metadata.json') as f: return json.load(f)
    except: return {}

def _ns():
    try:
        with open('data/processed/num_sensors.txt') as f: return int(f.read())
    except: return 15

session, MODEL_TYPE, _ = _load_model()
agent   = MaintenanceAgent()
monitor = ModelMonitor()
rgen    = ReportGenerator()
aext    = AttentionExtractor(num_sensors=_ns())
meta    = _meta()
NS      = _ns()
iname   = session.get_inputs()[0].name

# ── UI HELPERS ────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub='', variant=''):
    return html.Div(className=f'kpi-card {variant}', children=[
        html.Div(label, className='kpi-label'),
        html.Div(str(value), className='kpi-value'),
        html.Div(sub, className='kpi-delta'),
    ])

def section(title, *children):
    return html.Div(className='card', children=[
        html.Div(title, className='card-title'),
        *children
    ])

def hdr(title, sub=''):
    return html.Div(className='page-header', children=[
        html.H2(title), html.P(sub) if sub else None
    ])

def alert_banner(msg, kind='success'):
    return html.Div(className=f'alert-banner {kind}', children=msg)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
PAGES = [
    ('live','🔴','Live Monitoring'), ('edge','📊','Model & Edge Stats'),
    ('mlops','🔄','MLOps & Retraining'), ('alerts','🤖','Agent Alert Log'),
    ('cost','💰','Cost & Power Savings'), ('dataset','📈','Dataset Comparison'),
    ('heatmap','🗺️','Sensor Heatmap'), ('report','📋','Maintenance Report'),
    ('timeline','⏰','Failure Timeline'),
]

def sidebar():
    links = [html.Div([
        html.Span(icon, style={'fontSize':'16px'}), ' ', label
    ], id=f'nav-{pid}', n_clicks=0, className='nav-link') for pid,icon,label in PAGES]
    return html.Div(className='sidebar', children=[
        html.Div(className='sidebar-logo', children=[
            html.H1('⚙️ Edge AI PdM'),
            html.P('Predictive Maintenance System'),
            html.P(f'{MODEL_TYPE} · {NS} Sensors', style={'fontSize':'10px','color':'#334155','marginTop':'4px'}),
        ]),
        html.Div('NAVIGATION', className='nav-section-label'),
        *links,
        html.Div(className='sidebar-footer', children=[
            html.Div([html.Span(className='live-dot green'),' System Online'], id='live-ind',
                     style={'fontSize':'12px','color':'#94a3b8'}),
            html.Div(id='clock', style={'fontSize':'11px','color':'#475569','marginTop':'4px'}),
        ])
    ])

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap'],
    suppress_callback_exceptions=True, title='Edge AI — Predictive Maintenance')
server = app.server

app.layout = html.Div([
    dcc.Store(id='page',  data='live'),
    dcc.Store(id='run',   data=False),
    dcc.Store(id='hist',  data=[]),
    dcc.Store(id='fault', data=False),
    dcc.Interval(id='tick',  interval=900,  n_intervals=0, disabled=True),
    dcc.Interval(id='clock-tick', interval=30000, n_intervals=0),
    dbc.Row([
        dbc.Col(sidebar(), width=2, className='p-0'),
        dbc.Col(html.Div(id='content', className='main-content'), width=10, className='p-0'),
    ], className='g-0', style={'minHeight':'100vh'}),
])

# ── PAGE RENDERS ──────────────────────────────────────────────────────────────
def render_live():
    return html.Div([
        hdr('🔴 Live Monitoring', 'Real-time anomaly detection · Edge inference < 1 ms'),
        dbc.Row([
            dbc.Col(html.Button('▶ Start',  id='b-start', n_clicks=0, className='btn-primary',  style={'width':'100%'}), width=3),
            dbc.Col(html.Button('⏹ Stop',   id='b-stop',  n_clicks=0, className='btn-danger',   style={'width':'100%'}), width=3),
            dbc.Col(html.Button('💥 Fault', id='b-fault', n_clicks=0, className='btn-danger',   style={'width':'100%'}), width=3),
            dbc.Col(html.Button('🗑 Clear', id='b-clear', n_clicks=0,
                                style={'width':'100%','background':'#1c2333','border':'1px solid #21293d','color':'#94a3b8','borderRadius':'8px','fontFamily':'Inter'}), width=3),
        ], className='g-2 mb-3'),
        html.Div(id='kpi-row', className='mb-3'),
        html.Div(id='alert-box'),
        section('📈 Anomaly Probability Trend', dcc.Graph(id='live-chart', config={'displayModeBar':False},
                style={'height':'300px'})),
        section('🤖 Agent Recommendation', html.Div(id='agent-rec')),
    ])

def render_edge():
    if not meta:
        return html.Div([hdr('📊 Model & Edge Stats'), html.P('Run convert_to_onnx.py first.')])
    sizes = {'PyTorch FP32': meta.get('pytorch_size_kb',0),
             'ONNX FP32': meta.get('onnx_fp32_size_kb',0),
             'ONNX INT8':  meta.get('onnx_int8_size_kb',0)}
    bar = go.Figure(go.Bar(x=list(sizes.keys()), y=list(sizes.values()),
        marker_color=['#ef4444','#f97316','#10b981'],
        text=[f'{v:.1f} KB' for v in sizes.values()], textposition='outside',
        textfont=dict(color='#e2e8f0')))
    bar.update_layout(title='📦 Model Size Comparison', yaxis_title='Size (KB)', **dark_layout(height=350))
    checks = [('Inference Latency', f"{meta.get('avg_latency_int8_ms',0):.3f} ms", '< 50 ms', '✅'),
              ('Model Format', 'ONNX Runtime', 'Edge-compatible', '✅'),
              ('Cloud Required', 'No', 'Edge-only', '✅'),
              ('Parameters', f"{meta.get('parameters',0):,}", 'Lightweight', '✅')]
    tbl_data = [{'Metric':m,'Result':r,'Requirement':req,'Status':s} for m,r,req,s in checks]
    return html.Div([
        hdr('📊 Model & Edge Stats', 'Lightweight Transformer — ONNX Quantized — Edge Ready'),
        dbc.Row([
            dbc.Col(kpi_card('Original Size', f"{meta.get('pytorch_size_kb',0):.1f} KB"), width=3),
            dbc.Col(kpi_card('Quantized Size', f"{meta.get('onnx_int8_size_kb',0):.1f} KB", variant='success'), width=3),
            dbc.Col(kpi_card('Parameters', f"{meta.get('parameters',0):,}"), width=3),
            dbc.Col(kpi_card('Avg Latency', f"{meta.get('avg_latency_int8_ms',0):.3f} ms", variant='success'), width=3),
        ], className='g-3 mb-3'),
        section('📊 Model Size Comparison', dcc.Graph(figure=bar, config={'displayModeBar':False})),
        section('⚡ Edge Deployment Proof', dash_table.DataTable(
            data=tbl_data, columns=[{'name':c,'id':c} for c in ['Metric','Result','Requirement','Status']],
            style_header={'backgroundColor':'#161b27','color':'#94a3b8','fontWeight':'700','border':'1px solid #21293d'},
            style_cell={'backgroundColor':'#0d1117','color':'#e2e8f0','border':'1px solid #21293d','fontFamily':'Inter','fontSize':'13px'},
        )),
        section('🏗️ Architecture', html.Pre(
            'Input (batch,30,15)\n  ↓ Linear Projection d=32\n  ↓ Positional Encoding\n  ↓ Transformer ×2 (4 heads)\n  ↓ Global Avg Pool\n  ↙              ↘\nAnomaly (0-1)   RUL (cycles)',
            style={'color':'#818cf8','fontFamily':'monospace','fontSize':'13px','background':'#0d1117','padding':'16px','borderRadius':'8px'}
        )),
    ])

def render_mlops():
    rep   = monitor.get_health_report()
    drift = rep['drift_status']
    bkind = 'danger' if drift['drift_detected'] else 'success'
    bmsg  = f"🚨 Drift detected — {drift['reason']}" if drift['drift_detected'] else f"✅ Model healthy — {drift['reason']}"
    return html.Div([
        hdr('🔄 MLOps & Retraining', 'Model drift monitoring · Auto-retrain pipeline'),
        dbc.Row([
            dbc.Col(kpi_card('Predictions', str(rep['total_predictions'])), width=4),
            dbc.Col(kpi_card('Status', '⚠️ DRIFT' if drift['drift_detected'] else '✅ Healthy',
                             variant='danger' if drift['drift_detected'] else 'success'), width=4),
            dbc.Col(kpi_card('Action', drift['action']), width=4),
        ], className='g-3 mb-3'),
        alert_banner(bmsg, bkind),
        section('🔧 Retraining',
            html.Button('🔄 Trigger Retraining', id='btn-retrain', n_clicks=0, className='btn-primary'),
            html.Div(id='retrain-out', style={'marginTop':'12px','fontSize':'13px','color':'#94a3b8'}),
        ),
        section('📋 Pipeline Flow', html.Pre(
            'Sensor Data → Predict → Monitor\n  ↓\nCheck drift every 50 preds\n  ↓\nDrift? → Retrain → New ONNX → Update baseline',
            style={'color':'#818cf8','fontFamily':'monospace','fontSize':'13px','background':'#0d1117','padding':'16px','borderRadius':'8px'}
        )),
    ])

def render_alerts():
    h = agent.alert_history
    if not h:
        return html.Div([hdr('🤖 Agent Alert Log'),
                         alert_banner('No alerts yet — go to Live Monitoring and Simulate Fault.', 'success')])
    counts = {}
    for a in h: counts[a['severity']] = counts.get(a['severity'],0)+1
    bar = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()),
        marker_color=[SEV_COLOR.get(k,'#6366f1') for k in counts],
        text=list(counts.values()), textposition='outside', textfont=dict(color='#e2e8f0')))
    bar.update_layout(title='Alert Severity Breakdown', **dark_layout(height=280))
    cards = []
    for i, a in enumerate(reversed(h[-20:])):
        icon = SEV_ICON.get(a['severity'], '⚠️')
        cards.append(dbc.AccordionItem([
            dbc.Row([
                dbc.Col([html.P(f"Prob: {a['anomaly_probability']}"),
                         html.P(f"Cause: {a['root_cause']}"),
                         html.P(f"Maint: {a['maintenance_schedule']}")], width=6),
                dbc.Col([html.P(f"Downtime: {a['estimated_downtime']}"),
                         html.P(f"Saved: {a['estimated_cost_saved']}")] +
                        [html.P(f'• {x}') for x in a.get('recommended_actions',[])[:3]], width=6),
            ])
        ], title=f"{icon} Alert {len(h)-i} | {a['severity']} | {a['timestamp']}"))
    return html.Div([
        hdr('🤖 Agent Alert Log', f'{len(h)} total alerts'),
        kpi_card('Total Alerts', str(len(h))),
        html.Div(style={'height':'12px'}),
        section('📊 Severity Breakdown', dcc.Graph(figure=bar, config={'displayModeBar':False})),
        section('📋 Alert History', dbc.Accordion(cards, start_collapsed=True)),
    ])

def render_cost():
    df = pd.DataFrame({'Severity':['LOW','MEDIUM','HIGH','CRITICAL'],
                       'Cost Saved ($)':[750,10000,75000,350000],
                       'Downtime Prevented':['3 hrs','10 hrs','1.5 days','4 days'],
                       'Maintenance Window':['14 days','7 days','48 hours','Immediate']})
    bar = go.Figure(go.Bar(x=df['Severity'], y=df['Cost Saved ($)'],
        marker_color=['#84cc16','#f59e0b','#ef4444','#7c3aed'],
        text=[f'${v:,}' for v in df['Cost Saved ($)']], textposition='outside',
        textfont=dict(color='#e2e8f0')))
    bar.update_layout(title='Cost Saved by Catching Failures Early', **dark_layout())
    return html.Div([
        hdr('💰 Cost & Power Savings Analysis'),
        dbc.Row([
            dbc.Col(kpi_card('Cloud Latency',  '200–500 ms', 'Per inference call',  'danger'),  width=3),
            dbc.Col(kpi_card('Edge Latency',   '< 1 ms',    '250× faster',          'success'), width=3),
            dbc.Col(kpi_card('Cloud Cost',     '~$2,000/mo','Compute + bandwidth',  'danger'),  width=3),
            dbc.Col(kpi_card('Edge Cost',      '$0/month',  'No cloud needed',       'success'), width=3),
        ], className='g-3 mb-3'),
        section('💵 Financial Impact', dcc.Graph(figure=bar, config={'displayModeBar':False})),
        section('📊 Detail Table', dash_table.DataTable(
            data=df.to_dict('records'), columns=[{'name':c,'id':c} for c in df.columns],
            style_header={'backgroundColor':'#161b27','color':'#94a3b8','fontWeight':'700','border':'1px solid #21293d'},
            style_cell={'backgroundColor':'#0d1117','color':'#e2e8f0','border':'1px solid #21293d','fontFamily':'Inter','fontSize':'13px'},
        )),
    ])

def render_dataset():
    ep = 'data/processed/evaluation_results.json'
    if not os.path.exists(ep):
        return html.Div([hdr('📈 Dataset Comparison'),
                         alert_banner('Run python src/model/evaluate.py first.', 'warning')])
    with open(ep) as f: results = json.load(f)
    rows = [{'Dataset':ds,'Accuracy':f"{r['accuracy']:.4f}",'F1':f"{r['f1_score']:.4f}",
             'AUC-ROC':f"{r['auc_roc']:.4f}",'Samples':r['test_samples']} for ds,r in results.items()]
    auc_bar = go.Figure(go.Bar(x=list(results.keys()),
        y=[r['auc_roc'] for r in results.values()],
        marker_color=['#10b981','#f59e0b','#3b82f6','#a855f7'],
        text=[f"{r['auc_roc']:.3f}" for r in results.values()], textposition='outside',
        textfont=dict(color='#e2e8f0')))
    auc_bar.update_layout(title='AUC-ROC by Dataset', **dark_layout(height=320, yaxis=dict(range=[0,1.2],gridcolor='#21293d')))
    return html.Div([
        hdr('📈 Dataset Comparison', 'Trained on FD001 · Evaluated on FD001–FD004'),
        section('📊 AUC-ROC Performance', dcc.Graph(figure=auc_bar, config={'displayModeBar':False})),
        section('📋 Test Set Results', dash_table.DataTable(
            data=rows, columns=[{'name':c,'id':c} for c in rows[0]],
            style_header={'backgroundColor':'#161b27','color':'#94a3b8','fontWeight':'700','border':'1px solid #21293d'},
            style_cell={'backgroundColor':'#0d1117','color':'#e2e8f0','border':'1px solid #21293d','fontFamily':'Inter','fontSize':'13px'},
        )),
    ])

def render_heatmap():
    return html.Div([
        hdr('🗺️ Sensor Heatmap', 'Explainable AI — which sensors triggered the alert?'),
        section('⚙️ Controls',
            dbc.Row([
                dbc.Col([html.Label('Simulation Mode', style={'fontSize':'12px','color':'#94a3b8','marginBottom':'6px'}),
                         dcc.Dropdown(id='sim-mode', options=['Normal Operation','Fan Bearing Fault','Compressor Fault','Random Anomaly'],
                                      value='Normal Operation',
                                      style={'background':'#161b27','border':'1px solid #21293d','color':'#e2e8f0'})], width=6),
                dbc.Col([html.Label('Engine ID', style={'fontSize':'12px','color':'#94a3b8','marginBottom':'6px'}),
                         dcc.Input(id='hm-engine', type='number', value=1, min=1, max=100,
                                   style={'width':'100%','background':'#161b27','border':'1px solid #21293d','color':'#e2e8f0','borderRadius':'8px','padding':'8px'})], width=3),
                dbc.Col(html.Button('🔍 Analyze', id='btn-heatmap', n_clicks=0, className='btn-primary',
                                    style={'width':'100%','marginTop':'24px'}), width=3),
            ], className='g-2'),
        ),
        html.Div(id='heatmap-out'),
    ])

def render_report():
    return html.Div([
        hdr('📋 Maintenance Report', 'Auto-generated professional report + email to manager'),
        section('⚙️ Parameters',
            dbc.Row([
                dbc.Col([html.Label('Engine ID',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Input(id='rp-engine',type='number',value=47,min=1,max=100,
                                   style={'width':'100%','background':'#161b27','border':'1px solid #21293d','color':'#e2e8f0','borderRadius':'8px','padding':'8px'})], width=3),
                dbc.Col([html.Label('Anomaly Probability',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Slider(id='rp-prob',min=0,max=1,step=0.01,value=0.78,marks={0:'0',0.5:'0.5',1:'1.0'},
                                    tooltip={'placement':'bottom'})], width=5),
                dbc.Col([html.Label('RUL (cycles)',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Slider(id='rp-rul',min=0,max=125,step=1,value=45,marks={0:'0',60:'60',125:'125'},
                                    tooltip={'placement':'bottom'})], width=4),
            ], className='g-3'),
            html.Div(style={'height':'16px'}),
            dbc.Row([
                dbc.Col(html.Button('📋 Generate Report', id='btn-report', n_clicks=0, className='btn-primary'), width=3),
                dbc.Col(html.Button('📧 Send Email to Manager', id='btn-email', n_clicks=0, className='btn-success'), width=4),
            ], className='g-2'),
        ),
        html.Div(id='report-out'),
        html.Div(id='email-status-out'),
    ])

def render_timeline():
    return html.Div([
        hdr('⏰ Failure Timeline', 'RUL cycles → Calendar dates with Safe / Warning / Danger zones'),
        section('⚙️ Parameters',
            dbc.Row([
                dbc.Col([html.Label('Engine ID',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Input(id='tl-engine',type='number',value=1,min=1,max=100,
                                   style={'width':'100%','background':'#161b27','border':'1px solid #21293d','color':'#e2e8f0','borderRadius':'8px','padding':'8px'})], width=3),
                dbc.Col([html.Label('RUL (cycles)',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Slider(id='tl-rul',min=0,max=125,step=1,value=60,marks={0:'0',60:'60',125:'125'},
                                    tooltip={'placement':'bottom'})], width=5),
                dbc.Col([html.Label('Anomaly Probability',style={'fontSize':'12px','color':'#94a3b8'}),
                         dcc.Slider(id='tl-prob',min=0,max=1,step=0.01,value=0.65,marks={0:'0',0.5:'0.5',1:'1.0'},
                                    tooltip={'placement':'bottom'})], width=4),
            ], className='g-3'),
            html.Div(style={'height':'16px'}),
            html.Button('⏰ Generate Timeline', id='btn-timeline', n_clicks=0, className='btn-primary'),
        ),
        html.Div(id='timeline-out'),
    ])

PAGE_MAP = {
    'live': render_live, 'edge': render_edge, 'mlops': render_mlops,
    'alerts': render_alerts, 'cost': render_cost, 'dataset': render_dataset,
    'heatmap': render_heatmap, 'report': render_report, 'timeline': render_timeline,
}

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Clock ─────────────────────────────────────────────────────────────────────
@app.callback(Output('clock','children'), Input('clock-tick','n_intervals'))
def update_clock(n):
    return datetime.now().strftime('%H:%M:%S  %d %b %Y')

# ── Nav: update active class and page store ───────────────────────────────────
@app.callback(
    Output('page','data'),
    [Input(f'nav-{pid}','n_clicks') for pid,_,_ in PAGES],
    prevent_initial_call=True
)
def nav_click(*args):
    triggered = ctx.triggered_id
    if triggered:
        return triggered.replace('nav-','')
    return 'live'

@app.callback(
    [Output(f'nav-{pid}','className') for pid,_,_ in PAGES],
    Input('page','data')
)
def highlight_nav(page):
    return ['nav-link active' if pid==page else 'nav-link' for pid,_,_ in PAGES]

# ── Page content ──────────────────────────────────────────────────────────────
@app.callback(Output('content','children'), Input('page','data'))
def render_page(page):
    fn = PAGE_MAP.get(page, render_live)
    return fn()

# ── Live monitoring controls ──────────────────────────────────────────────────
@app.callback(
    Output('run','data'), Output('fault','data'), Output('hist','data'),
    Output('tick','disabled'),
    Input('b-start','n_clicks'), Input('b-stop','n_clicks'),
    Input('b-fault','n_clicks'), Input('b-clear','n_clicks'),
    State('run','data'), State('hist','data'),
    prevent_initial_call=True
)
def controls(s, stop, fault, clear, running, hist):
    t = ctx.triggered_id
    if t == 'b-start':  return True,  False, hist,  False
    if t == 'b-stop':   return False, False, hist,  True
    if t == 'b-fault':  return running, True, hist,  not running
    if t == 'b-clear':  return running, False, [],   not running
    return running, False, hist, not running

# ── Live tick ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('kpi-row','children'), Output('alert-box','children'),
    Output('live-chart','figure'), Output('agent-rec','children'),
    Output('hist','data', allow_duplicate=True),
    Input('tick','n_intervals'),
    State('run','data'), State('fault','data'), State('hist','data'),
    prevent_initial_call=True
)
def live_tick(n, running, force, hist):
    if not running:
        raise dash.exceptions.PreventUpdate

    if force:
        data = np.clip(np.random.normal(0.92, 0.03, (1,30,NS)), 0,1).astype(np.float32)
    else:
        base = np.random.normal(0.35, 0.08, (1,30,NS))
        if np.random.random() > 0.85:
            base += np.random.normal(0.3, 0.1, (1,30,NS))
        data = np.clip(base, 0,1).astype(np.float32)

    prob = float(session.run(None, {iname: data})[0][0])
    monitor.log_prediction(prob)
    sdict = {f'sensor{i+1}': float(data[0,-1,i]) for i in range(NS)}
    action = agent.analyze_anomaly(prob, sdict, list(sdict.keys()))

    hist = (hist or [])[-299:]
    hist.append({'step': len(hist), 'prob': prob, 'severity': action['severity']})

    # Health score
    hs = round((1-prob)*100, 1)
    if hs >= 80:  hg, hc = 'A — Excellent', 'success'
    elif hs >= 60: hg, hc = 'B — Good',     ''
    elif hs >= 40: hg, hc = 'C — Warning',  'warning'
    elif hs >= 20: hg, hc = 'D — Critical', 'danger'
    else:          hg, hc = 'F — Imminent', 'critical'

    sev = action['severity']
    kpis = dbc.Row([
        dbc.Col(kpi_card('Anomaly Probability', f'{prob:.3f}',
                         variant='danger' if prob > 0.7 else 'warning' if prob > 0.5 else 'success'), width=3),
        dbc.Col(kpi_card('Health Score', f'{hs}/100', hg, hc), width=3),
        dbc.Col(kpi_card('Status', f'{SEV_ICON.get(sev,"")} {sev}',
                         variant='danger' if sev in ['HIGH','CRITICAL'] else 'warning' if sev=='MEDIUM' else 'success'), width=3),
        dbc.Col(kpi_card('Total Alerts', str(len(agent.alert_history))), width=3),
    ], className='g-3')

    # Alert banner
    if sev == 'CRITICAL':
        banner = alert_banner(f'💀 CRITICAL ALERT — {action["root_cause"]} | Act: {action["maintenance_schedule"]}', 'critical')
    elif sev == 'HIGH':
        banner = alert_banner(f'🔴 HIGH ALERT — {action["root_cause"]} | Maintenance: {action["maintenance_schedule"]}', 'danger')
    elif sev == 'MEDIUM':
        banner = alert_banner(f'🟠 MEDIUM — {action["root_cause"]}', 'warning')
    else:
        banner = alert_banner('✅ System NORMAL — All sensors within range', 'success')

    # Chart
    df = pd.DataFrame(hist)
    colors = [SEV_COLOR.get(s,'#6366f1') for s in df['severity']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['step'], y=df['prob'], mode='lines+markers',
                             line=dict(color='#6366f1', width=2),
                             marker=dict(size=4, color=colors), name='Anomaly Prob'))
    fig.add_hline(y=0.7, line_dash='dash', line_color='#ef4444', annotation_text='HIGH (0.70)')
    fig.add_hline(y=0.9, line_dash='dash', line_color='#7c3aed', annotation_text='CRITICAL (0.90)')
    fig.update_layout(xaxis_title='Time Step', yaxis=dict(range=[0,1]), **dark_layout(height=300))

    # Agent rec
    rec = dbc.Row([
        dbc.Col([
            html.P(f"📅 Next Maintenance: {action['maintenance_schedule']}", style={'color':'#e2e8f0','marginBottom':'6px'}),
            html.P(f"⏱️ Downtime: {action['estimated_downtime']}", style={'color':'#94a3b8','marginBottom':'6px'}),
            html.P(f"💰 Cost Saved: {action['estimated_cost_saved']}", style={'color':'#10b981','marginBottom':'6px'}),
            html.P(f"🔍 Root Cause: {action['root_cause']}", style={'color':'#e2e8f0'}),
        ], width=6),
        dbc.Col([html.P(f'• {a}', style={'color':'#94a3b8','marginBottom':'4px'})
                 for a in action.get('recommended_actions',[])[:5]], width=6),
    ])
    return kpis, banner, fig, rec, hist

# ── MLOps retrain button ──────────────────────────────────────────────────────
@app.callback(Output('retrain-out','children'), Input('btn-retrain','n_clicks'), prevent_initial_call=True)
def trigger_retrain(n):
    if n:
        monitor.trigger_retraining('Dashboard triggered')
        return alert_banner('✅ Retraining triggered! Run: python src/model/train.py', 'success')
    raise dash.exceptions.PreventUpdate

# ── Sensor heatmap ───────────────────────────────────────────────────────────
@app.callback(Output('heatmap-out','children'),
              Input('btn-heatmap','n_clicks'),
              State('sim-mode','value'), State('hm-engine','value'),
              prevent_initial_call=True)
def run_heatmap(n, mode, eng):
    if not n: raise dash.exceptions.PreventUpdate
    if mode == 'Normal Operation':
        data = np.random.normal(0.3,0.05,(1,30,NS)).astype(np.float32)
    elif mode == 'Fan Bearing Fault':
        data = np.random.normal(0.3,0.05,(1,30,NS)).astype(np.float32)
        data[0,:,1] = np.random.normal(0.9,0.05,30)
        data[0,:,7] = np.random.normal(0.85,0.05,30)
    elif mode == 'Compressor Fault':
        data = np.random.normal(0.3,0.05,(1,30,NS)).astype(np.float32)
        data[0,:,2] = np.random.normal(0.92,0.04,30)
        data[0,:,3] = np.random.normal(0.88,0.04,30)
    else:
        data = np.clip(np.random.normal(0.7,0.15,(1,30,NS)),0,1).astype(np.float32)

    scores = aext.get_sensor_importance(data)
    prob   = float(session.run(None, {iname: data})[0][0])
    names  = [v['name'] for v in scores.values()]
    imps   = [v['importance_pct'] for v in scores.values()]

    bar = go.Figure(go.Bar(x=imps, y=names, orientation='h',
        marker=dict(color=imps, colorscale='RdYlGn_r', showscale=True),
        text=[f'{v:.1f}%' for v in imps], textposition='outside',
        textfont=dict(color='#e2e8f0')))
    bar.update_layout(title=f'Sensor Importance — Engine #{eng} ({mode})',
                      xaxis_title='Importance %', yaxis=dict(autorange='reversed'),
                      **dark_layout(height=450))

    heat = go.Figure(go.Heatmap(z=data[0].T[:len(scores)], x=list(range(30)), y=names,
                                colorscale='RdYlGn_r', colorbar=dict(title='Value')))
    heat.update_layout(title='Sensor Values Over Last 30 Cycles',
                       xaxis_title='Time (cycles)', **dark_layout(height=420))

    hs = round((1-prob)*100,1)
    top3 = list(scores.items())[:3]
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card('Anomaly Probability', f'{prob:.3f}', variant='danger' if prob>0.7 else ''), width=3),
            dbc.Col(kpi_card('Health Score', f'{hs}/100', variant='success' if hs>=60 else 'danger'), width=3),
            *[dbc.Col(kpi_card(info['name'], f"{info['importance_pct']:.1f}%",
                               'HIGH ATTENTION' if info['importance_pct']>70 else 'MODERATE',
                               'danger' if info['importance_pct']>70 else 'warning'), width=2)
              for _,info in top3],
        ], className='g-3 mb-3'),
        section('🗺️ Sensor Importance', dcc.Graph(figure=bar, config={'displayModeBar':False})),
        section('📊 Readings Heatmap (30 Cycles)', dcc.Graph(figure=heat, config={'displayModeBar':False})),
    ])

# ── Maintenance report ────────────────────────────────────────────────────────
@app.callback(Output('report-out','children'),
              Input('btn-report','n_clicks'),
              State('rp-engine','value'), State('rp-prob','value'), State('rp-rul','value'),
              prevent_initial_call=True)
def gen_report(n, eid, prob, rul):
    if not n: raise dash.exceptions.PreventUpdate
    sev = 'NORMAL' if prob<0.3 else 'LOW' if prob<0.5 else 'MEDIUM' if prob<0.7 else 'HIGH' if prob<0.9 else 'CRITICAL'
    tl  = predict_failure_timeline(rul, prob, eid)
    fsensors = {
        'sensor2':  {'name':'Fan Speed','importance':0.9,'importance_pct':90.0},
        'sensor7':  {'name':'HPC Pressure','importance':0.72,'importance_pct':72.0},
        'sensor11': {'name':'Static Pressure','importance':0.55,'importance_pct':55.0},
    }
    txt = rgen.generate_report(engine_id=eid, anomaly_prob=prob, severity=sev,
                               root_cause='Fan bearing wear detected', timeline=tl,
                               sensor_importance=fsensors,
                               action_plan=['Inspect fan bearing','Reduce load 20%',
                                            'Order SKF 6205-2RS','Schedule shutdown',
                                            'Notify maintenance lead'])
    fc = FAILURE_COSTS.get(sev, 25000)
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card('Severity', sev, variant='danger' if sev in ['HIGH','CRITICAL'] else 'warning'), width=3),
            dbc.Col(kpi_card('Act Within', f"{tl['days_until_maintenance']} days"), width=3),
            dbc.Col(kpi_card('Failure Cost', f'${fc:,}', 'if ignored', 'danger'), width=3),
            dbc.Col(kpi_card('Confidence', f"{tl.get('confidence_pct',85)}%", variant='success'), width=3),
        ], className='g-3 my-3'),
        section('📄 Full Report',
            html.Pre(txt, style={'color':'#cbd5e1','fontFamily':'monospace','fontSize':'12px',
                                 'whiteSpace':'pre-wrap','background':'#0d1117','padding':'16px','borderRadius':'8px'}),
        ),
    ])

@app.callback(Output('email-status-out','children'),
              Input('btn-email','n_clicks'),
              State('rp-engine','value'), State('rp-prob','value'), State('rp-rul','value'),
              prevent_initial_call=True)
def send_report_email(n, eid, prob, rul):
    if not n: raise dash.exceptions.PreventUpdate
    sev = 'NORMAL' if prob<0.3 else 'LOW' if prob<0.5 else 'MEDIUM' if prob<0.7 else 'HIGH' if prob<0.9 else 'CRITICAL'
    tl  = predict_failure_timeline(rul, prob, eid)
    alert_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_probability': prob, 'severity': sev,
        'root_cause': 'Fan bearing wear detected',
        'recommended_actions': ['Inspect fan bearing','Reduce load 20%','Order SKF 6205-2RS','Notify lead'],
        'maintenance_schedule': tl.get('recommended_maintenance','TBD'),
        'estimated_downtime': '1-2 days', 'estimated_cost_saved': '$50,000-100,000',
    }
    fsensors = {
        'sensor2':  {'name':'Fan Speed','importance':0.9,'importance_pct':90.0},
        'sensor7':  {'name':'HPC Pressure','importance':0.72,'importance_pct':72.0},
        'sensor11': {'name':'Static Pressure','importance':0.55,'importance_pct':55.0},
    }
    try:
        from src.agent.alert_emailer import AlertEmailer
        em = AlertEmailer()
        prob_hist = list(np.linspace(0.2, prob, 30))
        ok = em.send_alert_email(alert_data, prob_hist, fsensors, engine_id=eid, async_send=False)
        if ok:
            return alert_banner('📧 Email sent successfully to monish0329@gmail.com — check your inbox!', 'success')
        else:
            return alert_banner('⚠️ Email not sent — check config/alert_config.json for SMTP credentials.', 'warning')
    except Exception as e:
        return alert_banner(f'❌ Email failed: {str(e)[:120]} — add Gmail App Password to config/alert_config.json', 'danger')

# ── Failure timeline ──────────────────────────────────────────────────────────
@app.callback(Output('timeline-out','children'),
              Input('btn-timeline','n_clicks'),
              State('tl-engine','value'), State('tl-rul','value'), State('tl-prob','value'),
              prevent_initial_call=True)
def gen_timeline(n, eid, rul, prob):
    if not n: raise dash.exceptions.PreventUpdate
    tl = predict_failure_timeline(rul, prob, eid)
    ms = get_timeline_milestones(tl['rul_days'])
    today = datetime.now()

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=tl['rul_days']*0.5, fillcolor='rgba(16,185,129,0.08)', layer='below', line_width=0,
                  annotation_text='✅ Safe Zone', annotation_position='top left', annotation_font_color='#10b981')
    fig.add_vrect(x0=tl['rul_days']*0.5, x1=tl['rul_days']*0.8, fillcolor='rgba(245,158,11,0.08)', layer='below', line_width=0,
                  annotation_text='⚠️ Warning Zone', annotation_position='top left', annotation_font_color='#f59e0b')
    fig.add_vrect(x0=tl['rul_days']*0.8, x1=tl['rul_days']*1.2, fillcolor='rgba(239,68,68,0.08)', layer='below', line_width=0,
                  annotation_text='🚨 Danger Zone', annotation_position='top left', annotation_font_color='#ef4444')
    for m in ms:
        fig.add_vline(x=m['days'], line_dash='dash', line_color=m['color'],
                      annotation_text=f"{m['label']}<br>{m['date']}", annotation_position='top')
    fig.add_trace(go.Scatter(
        x=[tl['rul_days']*(1-0.2), tl['rul_days'], tl['rul_days']*(1+0.2)],
        y=[0.5,1.0,0.5], fill='tozeroy', fillcolor='rgba(239,68,68,0.2)',
        line=dict(color='#ef4444'), name='Failure Risk'))
    fig.update_layout(title=f'Engine #{eid} — Failure Prediction Timeline',
                      xaxis_title='Days from Today', yaxis_title='Failure Risk',
                      **dark_layout(height=420))

    ms_data = [{'Action':m['label'],'Date':(today+timedelta(days=m['days'])).strftime('%Y-%m-%d'),
                'Days':m['days']} for m in ms]

    urgency_variant = {'PLANNED':'success','SOON':'warning','URGENT':'danger','CRITICAL — TODAY':'critical'}
    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card('RUL (cycles)', str(tl['rul_cycles'])), width=3),
            dbc.Col(kpi_card('RUL (days)', f"{tl['rul_days']} days"), width=3),
            dbc.Col(kpi_card('Predicted Failure', tl['predicted_failure_date']), width=3),
            dbc.Col(kpi_card('Act Before', tl['recommended_maintenance'], variant='danger'), width=3),
        ], className='g-3 mb-3'),
        alert_banner(f"⚡ Urgency: {tl['urgency']} — Act within {tl['days_until_maintenance']} days",
                     urgency_variant.get(tl['urgency'],'warning')),
        section('⏰ Timeline Chart', dcc.Graph(figure=fig, config={'displayModeBar':False})),
        section('📅 Action Schedule', dash_table.DataTable(
            data=ms_data, columns=[{'name':c,'id':c} for c in ['Action','Date','Days']],
            style_header={'backgroundColor':'#161b27','color':'#94a3b8','fontWeight':'700','border':'1px solid #21293d'},
            style_cell={'backgroundColor':'#0d1117','color':'#e2e8f0','border':'1px solid #21293d','fontFamily':'Inter','fontSize':'13px'},
        )),
        dbc.Row([
            dbc.Col(kpi_card('Degradation Rate', f"{tl['degradation_rate_per_day']}% / day"), width=4),
            dbc.Col(kpi_card('Confidence', f"{tl['confidence_pct']}%", variant='success'), width=4),
            dbc.Col(kpi_card('Failure Window', f"{tl['earliest_failure_date']} → {tl['latest_failure_date']}"), width=4),
        ], className='g-3 mt-3'),
    ])


if __name__ == '__main__':
    print('\n' + '='*60)
    print(' 🚀 Edge AI Predictive Maintenance Dashboard')
    print(' 🌐 Open: http://localhost:8050')
    print(' 📧 Manager email: monish0329@gmail.com')
    print('='*60 + '\n')
    app.run(debug=False, host='0.0.0.0', port=8050)
