"""
====================================================================
SmartPark AI – Automated Parking Slot Occupancy Detection System
Using Overhead CCTV for Smart City Parking Management
====================================================================
"""

import os
import sys
import time
import tempfile
import textwrap
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import (
    SLOT_DEFINITIONS, THEME_COLORS, DEFAULT_TEMPORAL_FRAMES,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_FRAME_SKIP,
    EMERGENCY_CATEGORIES, APP_METADATA
)
from utils import (
    generate_synthetic_cctv_video, generate_analytics_dataframe,
    export_analytics_csv, get_video_metadata, get_occupancy_timeline_df
)
from detector import VehicleDetector
from slot_manager import SlotManager
from recommender import SmartRecommender

# Page Configuration
st.set_page_config(
    page_title="SmartPark AI – Overhead CCTV Detection System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(content: str):
    """
    Renders clean HTML without any line having leading whitespace that triggers
    Markdown indented code block parsing (<pre><code>). Guarantees unsafe_allow_html=True.
    """
    clean_lines = [line.strip() for line in content.splitlines() if line.strip()]
    clean_content = "\n".join(clean_lines)
    st.markdown(clean_content, unsafe_allow_html=True)

# Load Custom Glassmorphism Theme CSS
css_path = os.path.join(BASE_DIR, "assets", "css", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        render_html(f"<style>{f.read()}</style>")

# --------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False

if "slot_manager" not in st.session_state:
    st.session_state.slot_manager = SlotManager(temporal_frames=DEFAULT_TEMPORAL_FRAMES)

if "detector" not in st.session_state:
    st.session_state.detector = VehicleDetector(
        model_path=os.path.join(BASE_DIR, "yolov8n.pt"),
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

if "last_video_file" not in st.session_state:
    st.session_state.last_video_file = None

if "last_annotated_frame" not in st.session_state:
    st.session_state.last_annotated_frame = None

if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None

if "temporal_frames" not in st.session_state:
    st.session_state.temporal_frames = DEFAULT_TEMPORAL_FRAMES

if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = DEFAULT_CONFIDENCE_THRESHOLD

if "frame_skip" not in st.session_state:
    st.session_state.frame_skip = DEFAULT_FRAME_SKIP

if "show_boxes" not in st.session_state:
    st.session_state.show_boxes = True

if "enable_temporal" not in st.session_state:
    st.session_state.enable_temporal = True

if "selected_slot_modal" not in st.session_state:
    st.session_state.selected_slot_modal = "P01"

if "selected_emergency_cat" not in st.session_state:
    st.session_state.selected_emergency_cat = "Medical Emergency"

if "emergency_submitted" not in st.session_state:
    st.session_state.emergency_submitted = False

if "smart_rec_mode" not in st.session_state:
    st.session_state.smart_rec_mode = "normal"

recommender = SmartRecommender(st.session_state.slot_manager)

def switch_page(page_name):
    st.session_state.active_page = page_name
    st.rerun()

# --------------------------------------------------------------------
# Sidebar Navigation (Exact 7 Menus + Strict Locking Mechanism)
# --------------------------------------------------------------------
LOCKED_MENUS = ["Parking Map", "Emergency Mode", "Smart Recommendation", "Analytics", "Settings"]

with st.sidebar:
    # Sidebar Brand Header
    render_html("""
    <div class="sidebar-brand-box">
        <div class="brand-icon-p">P</div>
        <div>
            <div class="brand-text-title">SmartPark <span class="brand-text-ai">AI</span></div>
            <div class="brand-subtitle">Overhead CCTV Parking Management</div>
        </div>
    </div>
    """)

    if st.session_state.is_analyzed:
        render_html("""
        <div class="brand-status-pill">
            <span class="status-dot-green"></span>
            <span>ACTIVE MONITORING — ONLINE</span>
        </div>
        """)
    else:
        render_html("""
        <div class="brand-status-pill" style="color: #F59E0B;">
            <span style="width:6px; height:6px; background:#F59E0B; border-radius:50%; display:inline-block; box-shadow:0 0 8px #F59E0B;"></span>
            <span>STANDBY — AWAITING CCTV</span>
        </div>
        """)

    nav_items = [
        {"id": "Dashboard", "icon": "⬡", "label": "Dashboard"},
        {"id": "Video Analysis", "icon": "▶", "label": "Video Analysis"},
        {"id": "Parking Map", "icon": "⊞", "label": "Parking Map"},
        {"id": "Emergency Mode", "icon": "🚨", "label": "Emergency Mode", "has_badge": True},
        {"id": "Smart Recommendation", "icon": "🎯", "label": "Smart Recommendation"},
        {"id": "Analytics", "icon": "📊", "label": "Analytics"},
        {"id": "Settings", "icon": "⚙", "label": "Settings"},
    ]

    for item in nav_items:
        is_active = (st.session_state.active_page == item["id"])
        btn_kind = "primary" if is_active else "secondary"
        
        # Check lock status
        is_locked = (not st.session_state.is_analyzed) and (item["id"] in LOCKED_MENUS)
        
        if is_locked:
            btn_text = f"{item['icon']}   {item['label']}   🔒"
        else:
            btn_text = f"{item['icon']}   {item['label']}"
            if item.get("has_badge") and st.session_state.is_analyzed:
                btn_text += "   ●" # Active dot

        if st.button(btn_text, key=f"side_{item['id']}", use_container_width=True, type=btn_kind):
            st.session_state.active_page = item["id"]
            st.rerun()

    render_html("""
    <div style="margin-top: 45px; padding-top: 15px; border-top: 1px solid #142042; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B; letter-spacing: 0.05em;">
        <div>CAM: CCTV-NORTH-04</div>
        <div style="color: #38BDF8; margin-top: 2px;">YOLOv8n + OPENCV</div>
        <div style="margin-top: 2px;">V2.4 — SMART CITY EDITION</div>
    </div>
    """)

# --------------------------------------------------------------------
# Top Header Renderer
# --------------------------------------------------------------------
def render_header(title, subtitle):
    now_date = time.strftime("%A, %d %b %Y")
    now_time = time.strftime("%I:%M:%S %p").lower()

    col_l, col_r = st.columns([1.8, 1])
    with col_l:
        render_html(f"""
        <div style="margin-bottom: 20px;">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """)
    with col_r:
        render_html(f"""
        <div class="page-datetime">
            <div>{now_date}</div>
            <div style="color:#38BDF8; font-family:'JetBrains Mono'; font-weight:600;">{now_time}</div>
        </div>
        """)

# --------------------------------------------------------------------
# Locked Feature Guard Renderer
# --------------------------------------------------------------------
def render_locked_feature_screen(feature_name):
    render_header(feature_name, "Restricted Feature — Video Analysis Required")
    render_html(f"""
    <div class="locked-screen-card">
        <div class="lock-big-icon">🔒</div>
        <div class="lock-card-title">{feature_name} is Locked</div>
        <div class="lock-card-message">Upload and analyse a parking video to unlock this feature.</div>
        <div class="lock-card-desc">
            To ensure data integrity, live parking slot mapping, emergency priority reservations, 
            AI smart recommendations, and historical analytics are strictly activated only after 
            an overhead CCTV video is ingested and verified through YOLO vehicle detection and 
            3-frame temporal verification.
        </div>
    </div>
    """)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🚀   Proceed to Video Analysis", key=f"btn_unlock_go_{feature_name}", type="primary", use_container_width=True):
            switch_page("Video Analysis")

# --------------------------------------------------------------------
# Check If Current Page is Locked
# --------------------------------------------------------------------
if (not st.session_state.is_analyzed) and (st.session_state.active_page in LOCKED_MENUS):
    render_locked_feature_screen(st.session_state.active_page)

# --------------------------------------------------------------------
# PAGE 1: SYSTEM DASHBOARD
# --------------------------------------------------------------------
elif st.session_state.active_page == "Dashboard":
    render_header("System Dashboard", "AI-powered automated parking occupancy overview & surveillance grid")

    is_anal = st.session_state.is_analyzed
    kpi = st.session_state.slot_manager.get_kpi_summary() if is_anal else None

    # Row of 5 KPI Cards (Exact Figma Structure)
    occ_str = str(kpi["occupied"]) if is_anal else "--"
    avail_str = str(kpi["available"]) if is_anal else "--"
    verif_str = str(kpi["verifying"]) if is_anal else "--"
    occ_pct_str = f"{kpi['occupancy_pct']}%" if is_anal else "--%"

    dash_red_cls = "dash-red" if is_anal else ""
    dash_green_cls = "dash-green" if is_anal else ""
    dash_amber_cls = "dash-amber" if is_anal else ""
    dash_purple_cls = "dash-purple" if is_anal else ""

    render_html(f"""
    <div class="kpi-row-5">
        <div class="kpi-card-exact">
            <div class="kpi-exact-label">TOTAL SLOTS</div>
            <div class="kpi-exact-val">14</div>
        </div>
        <div class="kpi-card-exact">
            <div class="kpi-exact-label">OCCUPIED</div>
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:{'#EF4444' if is_anal else '#64748B'};">{occ_str}</span>
                <div class="kpi-dash-line {dash_red_cls}"></div>
            </div>
        </div>
        <div class="kpi-card-exact">
            <div class="kpi-exact-label">AVAILABLE</div>
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:{'#10B981' if is_anal else '#64748B'};">{avail_str}</span>
                <div class="kpi-dash-line {dash_green_cls}"></div>
            </div>
        </div>
        <div class="kpi-card-exact">
            <div class="kpi-exact-label">VERIFYING</div>
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:{'#F59E0B' if is_anal else '#64748B'};">{verif_str}</span>
                <div class="kpi-dash-line {dash_amber_cls}"></div>
            </div>
        </div>
        <div class="kpi-card-exact">
            <div class="kpi-exact-label">OCCUPANCY</div>
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:{'#8B5CF6' if is_anal else '#64748B'}; font-size:1.8rem;">{occ_pct_str}</span>
                <div class="kpi-dash-line {dash_purple_cls}"></div>
            </div>
        </div>
    </div>
    """)

    # Middle Row: 3 Columns
    col_rel, col_alert, col_slot = st.columns([1, 1.25, 1.1])

    # Left: AI Reliability Section
    with col_rel:
        yolo_val = f"{kpi['avg_confidence']}%" if is_anal else "--%"
        temp_val = "Active (3 Frames)" if is_anal else "Standby (3-Frame Buffer)"
        stab_val = f"{kpi['stability_score']}%" if is_anal else "--%"
        cons_val = f"{kpi['frame_consistency']}%" if is_anal else "--%"

        render_html(f"""
        <div class="figma-card">
            <div class="figma-card-header">
                <span style="color:#F59E0B;">⭐</span>
                <span>AI Reliability</span>
            </div>
            <div class="rel-item-row">
                <span class="rel-item-label">YOLO Confidence</span>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#10B981' if is_anal else '#64748B'}; font-size:0.85rem;">{yolo_val}</span>
                    <div class="rel-dash-green" style="background:{'#10B981' if is_anal else '#475569'};"></div>
                </div>
            </div>
            <div class="rel-item-row">
                <span class="rel-item-label">Temporal Verification</span>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-family:'JetBrains Mono'; font-weight:700; color:#38BDF8; font-size:0.82rem;">{temp_val}</span>
                    <div class="rel-dash-blue"></div>
                </div>
            </div>
            <div class="rel-item-row">
                <span class="rel-item-label">Detection Stability</span>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#10B981' if is_anal else '#64748B'}; font-size:0.85rem;">{stab_val}</span>
                    <div class="rel-dash-green" style="background:{'#10B981' if is_anal else '#475569'};"></div>
                </div>
            </div>
            <div class="rel-item-row">
                <span class="rel-item-label">Frame Consistency</span>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#8B5CF6' if is_anal else '#64748B'}; font-size:0.85rem;">{cons_val}</span>
                    <div class="rel-dash-purple" style="background:{'#8B5CF6' if is_anal else '#475569'};"></div>
                </div>
            </div>
        </div>
        """)

    # Center: Live Alerts / Smart Alert Center
    with col_alert:
        if not is_anal:
            alert_body = """
            <div class="alert-blue-box">
                <span style="font-size:1.25rem; color:#3B82F6; line-height:1;">ℹ</span>
                <div>
                    <strong style="color:#FFFFFF;">System Ready & Camera Standby:</strong><br>
                    Upload an overhead parking CCTV video in <strong>Video Analysis</strong> to run YOLO detection, map slots, and activate live occupancy tracking.
                </div>
            </div>
            """
        else:
            alerts = st.session_state.slot_manager.alerts[:3]
            if not alerts:
                alert_body = "<p style='color:#94A3B8; font-size:0.85rem;'>No recent alerts recorded.</p>"
            else:
                alert_cards = []
                for a in alerts:
                    c_border = "#10B981" if a["type"] == "success" else ("#F59E0B" if a["type"] in ["warning", "amber"] else "#3B82F6")
                    alert_cards.append(f"""
                    <div style="background:rgba(14,24,56,0.7); border-left:3px solid {c_border}; padding:8px 12px; margin-bottom:8px; border-radius:4px; font-size:0.82rem;">
                        <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#94A3B8;">{a['timestamp']}</div>
                        <div style="color:#F1F5F9; margin-top:2px;">{a['message']}</div>
                    </div>
                    """)
                alert_body = "".join(alert_cards)

        render_html(f"""
        <div class="figma-card">
            <div class="figma-card-header">
                <span style="color:#EF4444;">❖</span>
                <span>Live Alert Center</span>
            </div>
            {alert_body}
        </div>
        """)

    # Right: Slot Overview & Quick Action Buttons
    with col_slot:
        slots_dict = st.session_state.slot_manager.slots
        dots_html_list = []
        for i in range(1, 15):
            sid = f"P{i:02d}"
            s_obj = slots_dict.get(sid, {})
            dot_c = "dot-gray"
            if is_anal:
                stt = s_obj.get("status", "AVAILABLE")
                if stt == "OCCUPIED": dot_c = "dot-red"
                elif stt == "VERIFYING": dot_c = "dot-amber"
                elif stt == "AVAILABLE": dot_c = "dot-green"
                elif stt == "MAINTENANCE": dot_c = "dot-gray"
            dots_html_list.append(f'<div class="slot-dot-box"><span class="slot-dot-indicator {dot_c}"></span></div>')

        render_html(f"""
        <div class="figma-card">
            <div class="figma-card-header">
                <span style="color:#38BDF8;">⊞</span>
                <span>Slot Overview Grid</span>
            </div>
            <div class="slot-matrix-grid">
                {''.join(dots_html_list)}
            </div>
            <div class="slot-matrix-legend">
                <span><span class="slot-dot-indicator dot-green" style="display:inline-block; margin-right:4px;"></span> Avail</span>
                <span><span class="slot-dot-indicator dot-red" style="display:inline-block; margin-right:4px;"></span> Occ</span>
                <span><span class="slot-dot-indicator dot-amber" style="display:inline-block; margin-right:4px;"></span> Verif</span>
                <span><span class="slot-dot-indicator dot-gray" style="display:inline-block; margin-right:4px;"></span> Maint</span>
            </div>
        </div>
        """)

        # Navigation Action Buttons directly below card
        if not is_anal:
            if st.button("▶   Upload & Analyze Video", key="dash_btn_up", type="primary", use_container_width=True):
                switch_page("Video Analysis")
        else:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("⊞   Parking Map", key="dash_btn_map", type="primary", use_container_width=True):
                    switch_page("Parking Map")
            with c_b2:
                if st.button("🚨   Emergency", key="dash_btn_emg", use_container_width=True):
                    switch_page("Emergency Mode")

    render_html("<div style='height:18px;'></div>")

    # Bottom Row: Zone A & Zone B Cards + Temporal Verification Diagram
    z_col1, z_col2, tv_col = st.columns([1, 1, 1.2])

    with z_col1:
        z_a_pct = kpi["zone_a_pct"] if is_anal else 0.0
        z_a_occ = kpi["zone_a_occ"] if is_anal else 0
        z_a_avail = kpi["zone_a_avail"] if is_anal else 0
        render_html(f"""
        <div class="zone-card">
            <div class="zone-header">
                <span class="zone-name">ZONE A (P01 – P07)</span>
                <span class="zone-pct-val">{z_a_pct if is_anal else '--'}% OCCUPIED</span>
            </div>
            <div class="zone-progress-bg">
                <div class="zone-progress-fill" style="width: {z_a_pct}%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94A3B8; margin-top:6px;">
                <span>Occupied: <strong style="color:#EF4444;">{z_a_occ if is_anal else '--'}</strong></span>
                <span>Available: <strong style="color:#10B981;">{z_a_avail if is_anal else '--'}</strong></span>
                <span>Maintenance: <strong style="color:#64748B;">1</strong></span>
            </div>
        </div>
        """)

    with z_col2:
        z_b_pct = kpi["zone_b_pct"] if is_anal else 0.0
        z_b_occ = kpi["zone_b_occ"] if is_anal else 0
        z_b_avail = kpi["zone_b_avail"] if is_anal else 0
        render_html(f"""
        <div class="zone-card">
            <div class="zone-header">
                <span class="zone-name">ZONE B (P08 – P14)</span>
                <span class="zone-pct-val">{z_b_pct if is_anal else '--'}% OCCUPIED</span>
            </div>
            <div class="zone-progress-bg">
                <div class="zone-progress-fill" style="width: {z_b_pct}%; background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94A3B8; margin-top:6px;">
                <span>Occupied: <strong style="color:#EF4444;">{z_b_occ if is_anal else '--'}</strong></span>
                <span>Available: <strong style="color:#10B981;">{z_b_avail if is_anal else '--'}</strong></span>
                <span>Reserved: <strong style="color:#A855F7;">1 (🚨)</strong></span>
            </div>
        </div>
        """)

    with tv_col:
        render_html("""
        <div class="tv-flow-box">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.85rem; font-weight:700; color:#38BDF8;">⏱ TEMPORAL VERIFICATION ENGINE</span>
                <span style="font-size:0.72rem; color:#10B981; font-weight:700;">ACTIVE (3 FRAMES)</span>
            </div>
            <div class="tv-flow-steps">
                <div class="tv-step-node">
                    <div class="tv-step-label">FRAME 1</div>
                    <div class="tv-step-val">Scan</div>
                </div>
                <span class="tv-arrow">➔</span>
                <div class="tv-step-node">
                    <div class="tv-step-label">FRAME 2</div>
                    <div class="tv-step-val">Verify</div>
                </div>
                <span class="tv-arrow">➔</span>
                <div class="tv-step-node">
                    <div class="tv-step-label">FRAME 3</div>
                    <div class="tv-step-val">Confirm</div>
                </div>
                <span class="tv-arrow">➔</span>
                <div class="tv-step-node confirmed">
                    <div class="tv-step-label" style="color:#10B981;">STATE</div>
                    <div class="tv-step-val" style="color:#10B981;">LOCKED</div>
                </div>
            </div>
            <div style="font-size:0.72rem; color:#64748B; margin-top:8px;">
                Eliminates false triggers caused by camera noise, headlight flicker, and brief vehicle transit.
            </div>
        </div>
        """)

# --------------------------------------------------------------------
# PAGE 2: VIDEO ANALYSIS (Main Ingestion Engine)
# --------------------------------------------------------------------
elif st.session_state.active_page == "Video Analysis":
    render_header("Video Analysis", "Upload overhead parking CCTV surveillance video for OpenCV & YOLOv8 analysis")

    demo_video_path = os.path.join(BASE_DIR, "assets", "demo_parking_video.mp4")
    if not os.path.exists(demo_video_path):
        with st.spinner("Generating synthetic 1080p overhead CCTV footage..."):
            generate_synthetic_cctv_video(demo_video_path, duration_seconds=8, fps=25)

    # Big Dashed Upload Box
    render_html("""
    <div class="upload-dashed-container">
        <div class="upload-cam-icon">📹</div>
        <div style="font-size:1.35rem; font-weight:800; color:#FFFFFF; margin-bottom:6px;">
            Upload Parking CCTV Video
        </div>
        <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:18px;">
            Upload an overhead CCTV recording (MP4, AVI, MOV) for automated vehicle detection and slot occupancy mapping.
        </div>
        <div style="margin-bottom:16px;">
            <span class="upload-pill">MP4</span>
            <span class="upload-pill">AVI</span>
            <span class="upload-pill">MOV</span>
        </div>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    col_btn_demo, _ = st.columns([1.2, 1.8])
    with col_btn_demo:
        use_demo = st.button("🎥   Load Demo CCTV Surveillance Feed", key="up_demo_btn", use_container_width=True)

    target_video = None
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        tfile.write(uploaded_file.read())
        target_video = tfile.name
        st.session_state.last_video_file = target_video
    elif use_demo:
        target_video = demo_video_path
        st.session_state.last_video_file = target_video
    elif st.session_state.last_video_file:
        target_video = st.session_state.last_video_file

    render_html("<div style='height:16px;'></div>")

    if not target_video:
        render_html("""
        <div class="figma-card" style="text-align:center; padding:44px 20px;">
            <div style="font-size:2.2rem; color:#64748B; margin-bottom:12px;">📹</div>
            <div style="font-size:1.2rem; font-weight:700; color:#FFFFFF; margin-bottom:6px;">
                No Parking Video Selected
            </div>
            <div style="font-size:0.88rem; color:#94A3B8;">
                Please upload a video above or click <strong>Load Demo CCTV Surveillance Feed</strong> to test the live AI pipeline.
            </div>
        </div>
        """)
    else:
        # Extract and display video metadata
        vmeta = get_video_metadata(target_video)

        # Video Information Bar
        render_html(f"""
        <div class="figma-card" style="margin-bottom:16px; padding:14px 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">CAMERA STREAM</span>
                    <div style="font-family:'JetBrains Mono'; font-weight:700; color:#FFFFFF; font-size:0.95rem;">CCTV-NORTH-LOT-04</div>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">RESOLUTION</span>
                    <div style="font-family:'JetBrains Mono'; font-weight:700; color:#38BDF8; font-size:0.95rem;">{vmeta['width']} x {vmeta['height']}</div>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">FRAME RATE</span>
                    <div style="font-family:'JetBrains Mono'; font-weight:700; color:#10B981; font-size:0.95rem;">{vmeta['fps']} FPS</div>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">DURATION / FRAMES</span>
                    <div style="font-family:'JetBrains Mono'; font-weight:700; color:#F59E0B; font-size:0.95rem;">{vmeta['duration_seconds']}s ({vmeta['total_frames']} frames)</div>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#64748B; text-transform:uppercase;">CODEC / SIZE</span>
                    <div style="font-family:'JetBrains Mono'; font-weight:700; color:#8B5CF6; font-size:0.95rem;">{vmeta['codec']} ({vmeta['file_size_mb']} MB)</div>
                </div>
            </div>
        </div>
        """)

        render_html("""
        <div class="figma-card-header" style="margin-bottom:12px;">
            <span>📹 Video Ingestion & YOLOv8 Inference Stream</span>
        </div>
        """)

        vc1, vc2 = st.columns(2)
        with vc1:
            render_html("<p style='font-size:0.85rem; color:#94A3B8; margin-bottom:6px;'>Original CCTV Surveillance Feed</p>")
            st.video(target_video)
        with vc2:
            render_html("<p style='font-size:0.85rem; color:#10B981; margin-bottom:6px;'>OpenCV + YOLOv8 Vehicle & Slot Mapping Viewport</p>")
            frame_view = st.empty()
            if st.session_state.last_annotated_frame is not None:
                frame_view.image(st.session_state.last_annotated_frame, channels="RGB", use_container_width=True)
            else:
                frame_view.info("Ready to process. Click 'Start AI Detection & Slot Analysis' below.")

        render_html("<div style='height:14px;'></div>")

        if st.button("🚀   Start AI Detection & Slot Analysis", key="btn_run_yolo", type="primary", use_container_width=True):
            p_bar = st.progress(0)
            status_txt = st.empty()

            cap = cv2.VideoCapture(target_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

            f_idx = 0
            latest_annotated = None
            detected_vehicle_count = 0

            while cap.isOpened() and f_idx < 120:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                if f_idx % st.session_state.frame_skip != 0: continue

                h, w = frame.shape[:2]
                if w != 1280 or h != 720: frame = cv2.resize(frame, (1280, 720))

                # Step 1: Detect vehicles using YOLOv8
                detections = st.session_state.detector.detect_vehicles(frame, conf_threshold=st.session_state.conf_threshold)
                detected_vehicle_count = max(detected_vehicle_count, len(detections))

                # Step 2: Map detections to parking slots
                raw_mappings = st.session_state.detector.map_detections_to_slots(detections)

                # Step 3: Run temporal verification engine
                st.session_state.slot_manager.update_frame(raw_mappings, enable_temporal=st.session_state.enable_temporal)

                # Step 4: Annotate frame with OpenCV
                annotated = st.session_state.detector.annotate_frame(
                    frame, detections, st.session_state.slot_manager.slots, show_boxes=st.session_state.show_boxes
                )
                latest_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Step 5: Render live viewport & telemetry
                frame_view.image(latest_annotated, channels="RGB", use_container_width=True)
                p_bar.progress(min(1.0, f_idx / min(120, total_frames)))
                status_txt.text(f"Processing Frame {f_idx}/{min(120, total_frames)} | YOLO Detections: {len(detections)} vehicles | Temporal Filter: Active")
                time.sleep(0.015)

            cap.release()
            st.session_state.is_analyzed = True
            st.session_state.last_annotated_frame = latest_annotated
            
            # Save analysis summary
            kpi_post = st.session_state.slot_manager.get_kpi_summary()
            st.session_state.analysis_summary = {
                "frames_processed": f_idx,
                "vehicles_detected": detected_vehicle_count,
                "slots_scanned": 14,
                "occupied": kpi_post["occupied"],
                "available": kpi_post["available"],
                "verifying": kpi_post["verifying"],
                "maintenance": kpi_post["maintenance"],
                "avg_confidence": kpi_post["avg_confidence"],
                "stability_score": kpi_post["stability_score"]
            }

            st.balloons()
            st.success("🎉 Video Analysis Complete! All system pages and tactical features are now UNLOCKED.")
            st.rerun()

        # Final Analysis Summary Card (Requirement 5.10)
        if st.session_state.analysis_summary is not None:
            summ = st.session_state.analysis_summary
            render_html(f"""
            <div style="background:rgba(16,185,129,0.08); border:1px solid #10B981; border-radius:10px; padding:18px; margin-top:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF;">
                        ✓ AI Analysis Summary Report
                    </div>
                    <span style="background:#10B981; color:#060B1E; font-size:0.75rem; font-weight:800; padding:4px 10px; border-radius:6px;">
                        SYSTEM UNLOCKED
                    </span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; font-family:'JetBrains Mono'; font-size:0.85rem;">
                    <div>Frames Processed:<br><strong style="color:#FFFFFF; font-size:1.1rem;">{summ['frames_processed']}</strong></div>
                    <div>Vehicles Detected:<br><strong style="color:#38BDF8; font-size:1.1rem;">{summ['vehicles_detected']}</strong></div>
                    <div>Slots Available:<br><strong style="color:#10B981; font-size:1.1rem;">{summ['available']}</strong></div>
                    <div>Slots Occupied:<br><strong style="color:#EF4444; font-size:1.1rem;">{summ['occupied']}</strong></div>
                    <div>YOLO Confidence:<br><strong style="color:#F59E0B; font-size:1.1rem;">{summ['avg_confidence']}%</strong></div>
                </div>
            </div>
            """)

            render_html("<div style='height:12px;'></div>")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if st.button("⊞   View Parking Map", key="btn_go_map_after", type="primary", use_container_width=True):
                    switch_page("Parking Map")
            with sc2:
                if st.button("🎯   Smart Recommendation", key="btn_go_rec_after", use_container_width=True):
                    switch_page("Smart Recommendation")
            with sc3:
                if st.button("🚨   Emergency Mode", key="btn_go_emg_after", use_container_width=True):
                    switch_page("Emergency Mode")

# --------------------------------------------------------------------
# PAGE 3: PARKING MAP (Overhead Theatre-Style Layout)
# --------------------------------------------------------------------
elif st.session_state.active_page == "Parking Map":
    render_header("Parking Map", "Theatre-style overhead parking layout with real-time slot occupancy & restrictions")

    kpi = st.session_state.slot_manager.get_kpi_summary()
    slots_dict = st.session_state.slot_manager.slots

    # Top 4 KPI Cards
    map_k_cols = st.columns(4)
    with map_k_cols[0]:
        render_html("""
        <div class="kpi-card-exact" style="min-height:85px;">
            <div class="kpi-exact-val">14</div>
            <div class="kpi-exact-label" style="margin-top:4px;">TOTAL BAYS</div>
        </div>
        """)

    with map_k_cols[1]:
        render_html(f"""
        <div class="kpi-card-exact" style="min-height:85px;">
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:#EF4444;">{kpi['occupied']}</span>
                <div class="kpi-dash-line dash-red"></div>
            </div>
            <div class="kpi-exact-label" style="margin-top:4px;">OCCUPIED</div>
        </div>
        """)

    with map_k_cols[2]:
        render_html(f"""
        <div class="kpi-card-exact" style="min-height:85px;">
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:#10B981;">{kpi['available']}</span>
                <div class="kpi-dash-line dash-green"></div>
            </div>
            <div class="kpi-exact-label" style="margin-top:4px;">AVAILABLE</div>
        </div>
        """)

    with map_k_cols[3]:
        render_html(f"""
        <div class="kpi-card-exact" style="min-height:85px;">
            <div style="display:flex; align-items:baseline; gap:8px;">
                <span class="kpi-exact-val" style="color:#8B5CF6; font-size:1.8rem;">{kpi['occupancy_pct']}%</span>
                <div class="kpi-dash-line dash-purple"></div>
            </div>
            <div class="kpi-exact-label" style="margin-top:4px;">OCCUPANCY</div>
        </div>
        """)

    render_html("<div style='height:18px;'></div>")

    # Main Grid: Left Map Layout (70%) vs Right Slot Inspector (30%)
    col_map, col_inspector = st.columns([2.2, 1.1])

    with col_map:
        # Centered Entrance Badge
        render_html("""
        <div style="text-align:center; margin-bottom:12px;">
            <div class="entrance-center-badge">
                <span style="color:#F59E0B; font-size:0.9rem;">▼</span>
                <span>>>> ENTRANCE GATE (NORTH INGRESS) >>></span>
            </div>
        </div>
        """)

        # ZONE A Box (P01 to P07)
        zone_a_boxes = []
        for i in range(1, 8):
            sid = f"P{i:02d}"
            s_obj = slots_dict.get(sid, {})
            stt = s_obj.get("status", "AVAILABLE")
            
            border_cls = "slot-border-available"
            dot_c = "#10B981"
            if stt == "OCCUPIED":
                border_cls = "slot-border-occupied"
                dot_c = "#EF4444"
            elif stt == "VERIFYING":
                border_cls = "slot-border-verifying"
                dot_c = "#F59E0B"
            elif stt == "MAINTENANCE":
                border_cls = "slot-border-maintenance"
                dot_c = "#64748B"

            # Check special restrictions
            extra_pill = ""
            if s_obj.get("is_accessible"):
                border_cls = "slot-border-accessible"
                extra_pill = '<div class="accessible-tag-pill">♿ Accessible</div>'
            elif s_obj.get("is_emergency"):
                border_cls = "slot-border-emergency"
                extra_pill = '<div class="emergency-tag-pill">🚨 Emergency</div>'
            elif s_obj.get("is_maintenance"):
                extra_pill = '<div style="font-size:0.65rem; color:#F59E0B; margin-top:3px; font-weight:700;">🔧 Maint</div>'

            zone_a_boxes.append(f"""
            <div class="slot-box-figma {border_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'JetBrains Mono'; font-weight:800; font-size:0.92rem; color:#FFFFFF;">{sid}</span>
                    <span style="width:7px; height:7px; border-radius:50%; background:{dot_c}; box-shadow:0 0 6px {dot_c};"></span>
                </div>
                <div>
                    <div style="font-size:0.66rem; color:#64748B;">Zone A</div>
                    <div style="font-size:0.68rem; font-weight:700; color:{dot_c};">{stt}</div>
                    {extra_pill}
                </div>
            </div>
            """)

        render_html(f"""
        <div class="figma-card" style="padding:16px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:0.8rem; font-weight:800; color:#38BDF8; letter-spacing:0.05em;">ZONE A (SLOTS P01 – P07)</span>
                <span style="font-size:0.72rem; color:#94A3B8;">NORTH CORRIDOR</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(7, 1fr); gap:8px;">
                {''.join(zone_a_boxes)}
            </div>
        </div>
        """)

        # Central Driveway
        render_html("""
        <div style="text-align:center; color:#64748B; font-size:0.75rem; letter-spacing:0.25em; font-weight:800; margin:16px 0; font-family:'JetBrains Mono';">
            ═════ CENTRAL DRIVEWAY & MANEUVERING AISLE ═════
        </div>
        """)

        # ZONE B Box (P08 to P14)
        zone_b_boxes = []
        for i in range(8, 15):
            sid = f"P{i:02d}"
            s_obj = slots_dict.get(sid, {})
            stt = s_obj.get("status", "AVAILABLE")
            
            border_cls = "slot-border-available"
            dot_c = "#10B981"
            if stt == "OCCUPIED":
                border_cls = "slot-border-occupied"
                dot_c = "#EF4444"
            elif stt == "VERIFYING":
                border_cls = "slot-border-verifying"
                dot_c = "#F59E0B"
            elif stt == "MAINTENANCE":
                border_cls = "slot-border-maintenance"
                dot_c = "#64748B"

            extra_pill = ""
            if s_obj.get("is_emergency"):
                border_cls = "slot-border-emergency"
                extra_pill = '<div class="emergency-tag-pill">🚨 Emergency</div>'
            elif s_obj.get("is_accessible"):
                border_cls = "slot-border-accessible"
                extra_pill = '<div class="accessible-tag-pill">♿ Accessible</div>'
            elif s_obj.get("is_maintenance"):
                extra_pill = '<div style="font-size:0.65rem; color:#F59E0B; margin-top:3px; font-weight:700;">🔧 Maint</div>'
            elif sid == "P14":
                extra_pill = '<div style="font-size:0.65rem; color:#10B981; margin-top:3px; font-weight:700;">⚡ Quick Exit</div>'

            zone_b_boxes.append(f"""
            <div class="slot-box-figma {border_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'JetBrains Mono'; font-weight:800; font-size:0.92rem; color:#FFFFFF;">{sid}</span>
                    <span style="width:7px; height:7px; border-radius:50%; background:{dot_c}; box-shadow:0 0 6px {dot_c};"></span>
                </div>
                <div>
                    <div style="font-size:0.66rem; color:#64748B;">Zone B</div>
                    <div style="font-size:0.68rem; font-weight:700; color:{dot_c};">{stt}</div>
                    {extra_pill}
                </div>
            </div>
            """)

        render_html(f"""
        <div class="figma-card" style="padding:16px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:0.8rem; font-weight:800; color:#38BDF8; letter-spacing:0.05em;">ZONE B (SLOTS P08 – P14)</span>
                <span style="font-size:0.72rem; color:#94A3B8;">SOUTH CORRIDOR</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(7, 1fr); gap:8px;">
                {''.join(zone_b_boxes)}
            </div>
        </div>
        """)

        # Centered Exit Badge
        render_html("""
        <div style="text-align:center; margin-top:12px;">
            <div class="exit-center-badge">
                <span>▲</span>
                <span>>>> EXIT BARRIER GATE (SOUTH EGRESS) >>></span>
            </div>
        </div>
        """)

    # Right: Slot Details & Administrator Maintenance Management
    with col_inspector:
        render_html("""
        <div class="figma-card-header" style="margin-bottom: 12px;">
            <span style="color:#38BDF8;">🔍</span>
            <span>Slot Telemetry Inspector</span>
        </div>
        """)

        selected_slot = st.selectbox(
            "Select Parking Bay to Inspect",
            [f"P{i:02d}" for i in range(1, 15)],
            index=0,
            key="slot_inspector_sel"
        )

        s_info = slots_dict.get(selected_slot, {})
        s_defn = SLOT_DEFINITIONS.get(selected_slot, {})
        stt_show = s_info.get("status", "AVAILABLE")

        c_status = "#10B981"
        if stt_show == "OCCUPIED": c_status = "#EF4444"
        elif stt_show == "VERIFYING": c_status = "#F59E0B"
        elif stt_show == "MAINTENANCE": c_status = "#64748B"

        restr_label = "None (Regular Bay)"
        if s_info.get("is_accessible"): restr_label = "♿ Accessible Only"
        elif s_info.get("is_emergency"): restr_label = "🚨 Emergency Reserved"
        elif s_info.get("is_maintenance"): restr_label = "🔧 Under Maintenance"

        render_html(f"""
        <div style="background:#0E1838; border-radius:10px; padding:16px; border:1px solid #162450; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-family:'JetBrains Mono'; font-size:1.4rem; font-weight:800; color:#FFFFFF;">{selected_slot}</span>
                <span style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:800; color:{c_status}; background:rgba(0,0,0,0.3); padding:4px 10px; border-radius:6px;">
                    {stt_show}
                </span>
            </div>
            <div style="font-size:0.88rem; font-weight:700; color:#E2E8F0; margin-bottom:10px;">{s_info.get('name')}</div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">Zone: <strong style="color:#FFFFFF;">Zone {s_defn.get('zone')}</strong></div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">Vehicle Detected: <strong style="color:#38BDF8;">{s_info.get('vehicle_type')}</strong></div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">YOLO Confidence: <strong style="color:#10B981;">{int(s_info.get('confidence', 0.9)*100)}%</strong></div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">Entrance Distance: <strong style="color:#FFFFFF;">{s_defn.get('dist_entrance')}m</strong></div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">Exit Distance: <strong style="color:#FFFFFF;">{s_defn.get('dist_exit')}m</strong></div>
            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:6px;">Restriction: <strong style="color:#F59E0B;">{restr_label}</strong></div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:8px;">Last Updated: {s_info.get('last_updated')}</div>
        </div>
        <p style='font-size:0.8rem; font-weight:700; color:#38BDF8; margin-bottom:6px;'>ADMINISTRATOR CONTROLS</p>
        """)

        # Administrator Maintenance Management Toggle (Requirement 9)
        is_curr_maint = s_info.get("is_maintenance", False)
        
        if is_curr_maint:
            if st.button(f"✅   Clear Maintenance for {selected_slot}", key=f"btn_maint_off_{selected_slot}", use_container_width=True):
                st.session_state.slot_manager.toggle_maintenance(selected_slot, enable=False)
                st.toast(f"Slot {selected_slot} restored to active operational pool!", icon="✅")
                st.rerun()
        else:
            if st.button(f"🔧   Mark {selected_slot} as UNDER MAINTENANCE", key=f"btn_maint_on_{selected_slot}", use_container_width=True):
                st.session_state.slot_manager.toggle_maintenance(selected_slot, enable=True)
                st.toast(f"Slot {selected_slot} marked UNDER MAINTENANCE! Excluded from allocation.", icon="🔧")
                st.rerun()

# --------------------------------------------------------------------
# PAGE 4: EMERGENCY MODE
# --------------------------------------------------------------------
elif st.session_state.active_page == "Emergency Mode":
    render_header("Emergency Mode", "Priority AI parking recommendation for medical and urgent ingress situations")

    col_emg_l, col_emg_r = st.columns([1.1, 1.3])

    with col_emg_l:
        render_html("""
        <div class="figma-card">
            <div class="figma-card-header">
                <span style="color:#EF4444;">🚨</span>
                <span>Select Emergency Situation</span>
            </div>
            <p style="font-size:0.85rem; color:#94A3B8; margin-bottom:16px;">
                Choose the emergency scenario to compute the optimal priority parking route:
            </p>
        </div>
        """)

        sit_options = [
            ("Medical Emergency", "🚑   Medical Emergency"),
            ("Accessibility / Elderly", "♿   Accessibility / Elderly"),
            ("Patient / Pregnancy", "🤰   Patient / Pregnancy"),
            ("Urgent Appointment", "⏰   Urgent Appointment"),
            ("Other Emergency", "⚠   Other Emergency")
        ]

        for code, lbl in sit_options:
            is_c = (st.session_state.selected_emergency_cat == code)
            k_type = "primary" if is_c else "secondary"
            if st.button(lbl, key=f"btn_sit_{code}", use_container_width=True, type=k_type):
                st.session_state.selected_emergency_cat = code
                st.session_state.emergency_submitted = True
                st.rerun()

        render_html("<div style='height:14px;'></div>")

        if st.button("🔍   Find Fastest Emergency Bay", key="btn_run_fastest", type="primary", use_container_width=True):
            st.session_state.emergency_submitted = True
            st.rerun()

    with col_emg_r:
        rec_e = recommender.recommend_slot(mode="emergency", emergency_category=st.session_state.selected_emergency_cat)
        
        if rec_e:
            reasons_html = "".join([
                f"<div style='font-size:0.84rem; color:#CBD5E1; margin-bottom:4px;'>{r_item}</div>"
                for r_item in rec_e.get("reasons", [])
            ])
            
            render_html(f"""
            <div style="border:2px solid #8B5CF6; background:rgba(139,92,246,0.12); border-radius:12px; padding:22px; width:100%; text-align:left;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span style="font-family:'JetBrains Mono'; font-size:2.2rem; font-weight:800; color:#FFFFFF;">{rec_e['slot_id']}</span>
                        <span style="margin-left:10px; font-size:0.85rem; color:#38BDF8; font-weight:700;">Zone {rec_e['zone']}</span>
                    </div>
                    <span style="background:#8B5CF6; color:#FFFFFF; font-size:0.75rem; font-weight:800; padding:5px 12px; border-radius:6px; letter-spacing:0.05em;">
                        PRIORITY LOCK
                    </span>
                </div>
                <div style="font-size:1.05rem; font-weight:800; color:#E2E8F0; margin-bottom:12px;">{rec_e['slot_name']}</div>
                
                <div style="margin-bottom:14px;">
                    <div style="font-size:0.78rem; font-weight:700; color:#A855F7; text-transform:uppercase; margin-bottom:6px;">WHY THIS BAY?</div>
                    {reasons_html}
                </div>
                <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:8px; font-family:'JetBrains Mono'; font-size:0.8rem; color:#94A3B8; margin-bottom:12px; background:rgba(0,0,0,0.3); padding:10px; border-radius:6px;">
                    <div>Distance: <strong style="color:#FFFFFF;">{rec_e['distance']}m</strong></div>
                    <div>YOLO Confidence: <strong style="color:#10B981;">{rec_e['score_breakdown']['YOLO Confidence']}%</strong></div>
                </div>
                <div style="padding:10px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:6px; font-size:0.82rem; color:#10B981;">
                    <strong>Navigation Guidance:</strong> {rec_e['nav_guidance']}
                </div>
            </div>
            """)

            render_html("<div style='height:14px;'></div>")
            if st.button("🚨   Confirm Emergency Parking Allocation & Issue Pass", key="btn_conf_pass", type="primary", use_container_width=True):
                st.session_state.slot_manager.log_reservation("EMERGENCY", rec_e['slot_id'], st.session_state.selected_emergency_cat)
                st.toast(f"Emergency Pass Issued for {rec_e['slot_id']}! North Gate barrier raised.", icon="🚨")
        else:
            st.warning("All emergency designated bays are currently occupied. Please alert ground parking marshals.")

        render_html("""
        <div style="margin-top:16px; font-size:0.75rem; color:#64748B; line-height:1.4; border-top:1px solid #142042; padding-top:10px;">
            ⚠️ <strong>IMPORTANT NOTICE:</strong> Emergency Mode is an intelligent parking slot allocation and priority ingress system. It is <strong>NOT</strong> an emergency response or medical ambulance dispatch service.
        </div>
        """)

# --------------------------------------------------------------------
# PAGE 5: SMART RECOMMENDATION
# --------------------------------------------------------------------
elif st.session_state.active_page == "Smart Recommendation":
    render_header("Smart Recommendation", "AI-prioritized parking allocation based on your vehicle profile & preference")

    # 3 Mode Buttons at Top
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        is_s = (st.session_state.smart_rec_mode == "normal")
        k_s = "primary" if is_s else "secondary"
        if st.button("🎯   NORMAL (Smart Parking)", key="btn_mode_smart", use_container_width=True, type=k_s):
            st.session_state.smart_rec_mode = "normal"
            st.rerun()

    with tc2:
        is_s = (st.session_state.smart_rec_mode == "accessible")
        k_s = "primary" if is_s else "secondary"
        if st.button("♿   ACCESSIBLE (Wheelchair / Elderly)", key="btn_mode_acc", use_container_width=True, type=k_s):
            st.session_state.smart_rec_mode = "accessible"
            st.rerun()

    with tc3:
        is_s = (st.session_state.smart_rec_mode == "quick_exit")
        k_s = "primary" if is_s else "secondary"
        if st.button("⚡   QUICK EXIT (Fastest Departure)", key="btn_mode_exit", use_container_width=True, type=k_s):
            st.session_state.smart_rec_mode = "quick_exit"
            st.rerun()

    render_html("<div style='height:18px;'></div>")

    rec_data = recommender.recommend_slot(mode=st.session_state.smart_rec_mode)

    if rec_data:
        reasons_html = "".join([
            f"<div style='font-size:0.88rem; color:#CBD5E1; margin-bottom:4px;'>{r_bullet}</div>"
            for r_bullet in rec_data.get("reasons", [])
        ])
        sb = rec_data.get("score_breakdown", {})

        render_html(f"""
        <div style="border:1.5px solid #3B82F6; background:rgba(59,130,246,0.08); border-radius:14px; padding:24px; max-width:750px; margin:0 auto; text-align:left;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <span style="font-size:0.78rem; font-weight:800; color:#38BDF8; letter-spacing:0.1em; text-transform:uppercase;">BEST SLOT FOR YOU</span>
                    <div style="font-family:'JetBrains Mono'; font-size:2.4rem; font-weight:800; color:#FFFFFF; line-height:1.1;">{rec_data['slot_id']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono'; font-size:1.5rem; color:#10B981; font-weight:800;">{rec_data['score']}/100</div>
                    <span style="font-size:0.75rem; color:#94A3B8;">PRIORITY MATCH</span>
                </div>
            </div>
            
            <div style="font-size:1.1rem; font-weight:800; color:#E2E8F0; margin-bottom:12px;">{rec_data['slot_name']} (Zone {rec_data['zone']})</div>
            
            <div style="margin-bottom:16px;">
                <div style="font-size:0.82rem; font-weight:800; color:#38BDF8; text-transform:uppercase; margin-bottom:6px;">WHY THIS SLOT?</div>
                {reasons_html}
            </div>

            <div style="font-size:0.82rem; font-weight:800; color:#94A3B8; text-transform:uppercase; margin-bottom:8px;">PRIORITY SCORE BREAKDOWN</div>
            <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; font-family:'JetBrains Mono'; font-size:0.75rem; text-align:center; margin-bottom:16px;">
                <div style="background:#0E1838; padding:8px; border-radius:6px; border:1px solid #162450;">
                    <div style="color:#64748B;">Avail</div>
                    <strong style="color:#10B981; font-size:0.95rem;">{sb.get('Availability', 100)}%</strong>
                </div>
                <div style="background:#0E1838; padding:8px; border-radius:6px; border:1px solid #162450;">
                    <div style="color:#64748B;">YOLO Conf</div>
                    <strong style="color:#38BDF8; font-size:0.95rem;">{sb.get('YOLO Confidence', 95)}%</strong>
                </div>
                <div style="background:#0E1838; padding:8px; border-radius:6px; border:1px solid #162450;">
                    <div style="color:#64748B;">Zone Match</div>
                    <strong style="color:#8B5CF6; font-size:0.95rem;">{sb.get('Zone Suitability', 90)}%</strong>
                </div>
                <div style="background:#0E1838; padding:8px; border-radius:6px; border:1px solid #162450;">
                    <div style="color:#64748B;">Proximity</div>
                    <strong style="color:#F59E0B; font-size:0.95rem;">{sb.get('Distance Proximity', 88)}%</strong>
                </div>
                <div style="background:#0E1838; padding:8px; border-radius:6px; border:1px solid #162450;">
                    <div style="color:#64748B;">Preference</div>
                    <strong style="color:#10B981; font-size:0.95rem;">{sb.get('User Preference', 95)}%</strong>
                </div>
            </div>

            <div style="padding:10px; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:6px; font-size:0.82rem; color:#38BDF8;">
                <strong>Nav:</strong> {rec_data['nav_guidance']}
            </div>
        </div>
        """)

        render_html("<div style='height:16px;'></div>")
        col_res1, col_res2, col_res3 = st.columns([1, 1.2, 1])
        with col_res2:
            if st.button("🅿️   Reserve Recommended Bay", key="btn_res_rec_live", type="primary", use_container_width=True):
                st.session_state.slot_manager.log_reservation(st.session_state.smart_rec_mode.upper(), rec_data['slot_id'], "Smart User Reservation")
                st.toast(f"Bay {rec_data['slot_id']} successfully reserved! Barrier pass active.", icon="🅿️")
    else:
        st.warning("No matching available parking slot found for this mode. All matching slots may be occupied or in maintenance.")

# --------------------------------------------------------------------
# PAGE 6: ANALYTICS
# --------------------------------------------------------------------
elif st.session_state.active_page == "Analytics":
    render_header("Analytics", "Comprehensive parking occupancy telemetry, zone comparisons & historical trends")

    kpi = st.session_state.slot_manager.get_kpi_summary()
    df_analytics = generate_analytics_dataframe(st.session_state.slot_manager.slots)

    # Top Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Slots", 14)
    with m2:
        st.metric("Free Bays", kpi["available"])
    with m3:
        st.metric("Occupancy Rate", f"{kpi['occupancy_pct']}%")
    with m4:
        st.metric("Avg YOLO Confidence", f"{kpi['avg_confidence']}%")
    with m5:
        st.metric("Stability Score", f"{kpi['stability_score']}%")

    render_html("<div style='height:10px;'></div>")

    st.download_button(
        label="📥   DOWNLOAD CSV REPORT",
        data=export_analytics_csv(df_analytics),
        file_name=f"SmartPark_Telemetry_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )

    render_html("<div style='height:18px;'></div>")

    # 4 Visualizations in 2 Rows
    an_c1, an_c2 = st.columns(2)
    with an_c1:
        render_html("<div class='figma-card-header'>Occupancy Distribution</div>")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Occupied", "Available", "Verifying", "Maintenance"],
            values=[kpi["occupied"], kpi["available"], kpi["verifying"], kpi["maintenance"]],
            hole=0.6,
            marker=dict(colors=["#EF4444", "#10B981", "#F59E0B", "#64748B"])
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF', family='Plus Jakarta Sans'),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with an_c2:
        render_html("<div class='figma-card-header'>Zone A vs Zone B Comparison</div>")
        fig_zone = go.Figure()
        fig_zone.add_trace(go.Bar(
            name='Occupied', x=['Zone A', 'Zone B'], y=[kpi['zone_a_occ'], kpi['zone_b_occ']],
            marker_color='#EF4444'
        ))
        fig_zone.add_trace(go.Bar(
            name='Available', x=['Zone A', 'Zone B'], y=[kpi['zone_a_avail'], kpi['zone_b_avail']],
            marker_color='#10B981'
        ))
        fig_zone.update_layout(
            barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF', family='Plus Jakarta Sans'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    render_html("<div style='height:14px;'></div>")

    an_c3, an_c4 = st.columns(2)
    with an_c3:
        render_html("<div class='figma-card-header'>Occupancy Over Time (Surveillance Timeline)</div>")
        df_timeline = get_occupancy_timeline_df(current_occupied=kpi["occupied"], total_usable=13)
        fig_time = px.area(
            df_timeline, x="Time", y="Occupied",
            labels={"Occupied": "Occupied Slots", "Time": "Surveillance Time"},
            color_discrete_sequence=["#8B5CF6"]
        )
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF', family='Plus Jakarta Sans'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with an_c4:
        render_html("<div class='figma-card-header'>Individual Slot YOLO Detection Confidence</div>")
        slot_names = [f"P{i:02d}" for i in range(1, 15)]
        confs = [round(st.session_state.slot_manager.slots[s]["confidence"] * 100, 1) for s in slot_names]
        colors = ['#64748B' if s == 'P07' else ('#3B82F6' if s in ['P02', 'P08'] else ('#A855F7' if s == 'P11' else '#10B981')) for s in slot_names]
        
        fig_bar = go.Figure(data=[go.Bar(x=slot_names, y=confs, marker_color=colors)])
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF', family='Plus Jakarta Sans'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[60, 100]),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------------------------
# PAGE 7: SETTINGS
# --------------------------------------------------------------------
elif st.session_state.active_page == "Settings":
    render_header("Settings", "AI vision parameters, temporal verification thresholds & pipeline architecture")

    # 1. AI Vision Model Tuning Card
    render_html("<div class='figma-card-header'>⚙ AI Vision Parameters & Filter Tuning</div>")

    set_c1, set_c2 = st.columns(2)
    with set_c1:
        new_conf = st.slider("YOLO Confidence Threshold", 0.20, 0.90, float(st.session_state.conf_threshold), 0.05)
        st.session_state.conf_threshold = new_conf

        new_skip = st.slider("Frame Skip Rate", 1, 5, int(st.session_state.frame_skip), 1)
        st.session_state.frame_skip = new_skip

    with set_c2:
        new_temp = st.slider("Temporal Confirmation Frames", 1, 5, int(st.session_state.temporal_frames), 1)
        st.session_state.temporal_frames = new_temp
        st.session_state.slot_manager.temporal_frames = new_temp

        st.session_state.show_boxes = st.toggle("Show YOLO Detection Bounding Boxes", value=st.session_state.show_boxes)
        st.session_state.enable_temporal = st.toggle("Enable 3-Frame Temporal Verification Filter", value=st.session_state.enable_temporal)

    render_html("<div style='height:24px;'></div>")

    # 2. Administrator Maintenance Management Section (Requirement 9 & 15)
    render_html("<div class='figma-card-header'>🔧 Parking Slot Maintenance Administration</div>")

    adm_c1, adm_c2 = st.columns([1.2, 1])
    with adm_c1:
        admin_sel_slot = st.selectbox(
            "Select Parking Bay to Configure",
            [f"P{i:02d}" for i in range(1, 15)],
            index=6, # Default to P07
            key="admin_settings_maint_slot"
        )
        curr_maint_state = st.session_state.slot_manager.slots[admin_sel_slot].get("is_maintenance", False)
        
        render_html(f"""
        <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">
            Current Status for <strong style="color:#FFFFFF;">{admin_sel_slot}</strong>: 
            <strong style="color:{'#F59E0B' if curr_maint_state else '#10B981'};">
                {'UNDER MAINTENANCE (🔧)' if curr_maint_state else 'AVAILABLE / OPERATIONAL (✅)'}
            </strong>
        </div>
        """)

    with adm_c2:
        render_html("<div style='height:28px;'></div>")
        if curr_maint_state:
            if st.button(f"✅   Set {admin_sel_slot} as AVAILABLE", key="btn_adm_avail", type="primary", use_container_width=True):
                st.session_state.slot_manager.toggle_maintenance(admin_sel_slot, enable=False)
                st.toast(f"Slot {admin_sel_slot} restored to operational status!", icon="✅")
                st.rerun()
        else:
            if st.button(f"🔧   Set {admin_sel_slot} as UNDER MAINTENANCE", key="btn_adm_maint", use_container_width=True):
                st.session_state.slot_manager.toggle_maintenance(admin_sel_slot, enable=True)
                st.toast(f"Slot {admin_sel_slot} marked as UNDER MAINTENANCE!", icon="🔧")
                st.rerun()

    render_html("<div style='height:24px;'></div>")

    # 3. Visual Explanation of AI Pipeline Architecture (Requirement 15)
    render_html("""
    <div class="figma-card">
        <div class="figma-card-header">🌐 End-to-End AI Vision Pipeline Architecture</div>
        <div class="pipeline-step-grid">
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 01</div>
                <div class="pipeline-step-title">CCTV Video Ingestion</div>
                <div class="pipeline-step-desc">High-angle surveillance video feeds ingested at 1080p/720p 25 FPS from overhead camera nodes.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 02</div>
                <div class="pipeline-step-title">OpenCV Frame Decoding</div>
                <div class="pipeline-step-desc">Frame extraction, noise reduction, aspect ratio standardization, and BGR color space transformation.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 03</div>
                <div class="pipeline-step-title">YOLOv8 Detection</div>
                <div class="pipeline-step-desc">Ultra-fast CNN inference detecting vehicles (Car, Motorcycle, Bus, Truck) with bounding box coordinates.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 04</div>
                <div class="pipeline-step-title">Slot Spatial Mapping</div>
                <div class="pipeline-step-desc">Centroid and polygon intersection analysis mapping detected vehicle bounding boxes to bays P01–P14.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 05</div>
                <div class="pipeline-step-title">Temporal Verification</div>
                <div class="pipeline-step-desc">3 consecutive frames confirmation buffer eliminating sensor noise, lighting glints, and temporary occlusions.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 06</div>
                <div class="pipeline-step-title">Parking Status Lock</div>
                <div class="pipeline-step-desc">State machine updates status: Available (Green), Occupied (Red), Verifying (Amber), Maintenance (Grey).</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 07</div>
                <div class="pipeline-step-title">Restriction Engine</div>
                <div class="pipeline-step-desc">Strict enforcement of Accessible (♿ P02), Emergency (🚨 P11), and Maintenance (🔧 P07) exclusions.</div>
            </div>
            <div class="pipeline-step-card">
                <div class="pipeline-step-num">STAGE 08</div>
                <div class="pipeline-step-title">Smart Recommendation</div>
                <div class="pipeline-step-desc">Multi-criteria allocation engine scoring proximity, YOLO confidence, and driver preference.</div>
            </div>
        </div>
    </div>
    """)
