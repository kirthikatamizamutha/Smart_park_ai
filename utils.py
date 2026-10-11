"""
SmartPark AI - Utility Functions
Synthetic CCTV Parking Video Generator, Video Processing Helpers, and Analytics Exporter.
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from config import SLOT_DEFINITIONS

def draw_topdown_car(image, center_x, center_y, width=240, height=48, angle=0, color=(180, 180, 180), car_type="Sedan"):
    """
    Draw a realistic top-down overhead view of a car with shadow, windshield, roof, and lights.
    """
    # 1. Subtle soft shadow
    shadow_offset = 5
    shadow_pts = np.array([
        [center_x - width // 2 + shadow_offset, center_y - height // 2 + shadow_offset],
        [center_x + width // 2 + shadow_offset, center_y - height // 2 + shadow_offset],
        [center_x + width // 2 + shadow_offset, center_y + height // 2 + shadow_offset],
        [center_x - width // 2 + shadow_offset, center_y + height // 2 + shadow_offset],
    ], np.int32)
    cv2.fillPoly(image, [shadow_pts], (15, 18, 24))

    # 2. Main Car Chassis (rounded rect)
    x1 = int(center_x - width // 2)
    y1 = int(center_y - height // 2)
    x2 = int(center_x + width // 2)
    y2 = int(center_y + height // 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    # Outline
    border_color = tuple(max(0, c - 40) for c in color)
    cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2)

    # 3. Windshield & Windows (dark tinted glass)
    glass_color = (40, 45, 55)
    # Front windshield (near right if facing right, or left if facing left)
    f_w1 = x1 + int(width * 0.22)
    f_w2 = x1 + int(width * 0.32)
    cv2.rectangle(image, (f_w1, y1 + 5), (f_w2, y2 - 5), glass_color, -1)

    # Rear window
    r_w1 = x1 + int(width * 0.68)
    r_w2 = x1 + int(width * 0.78)
    cv2.rectangle(image, (r_w1, y1 + 5), (r_w2, y2 - 5), glass_color, -1)

    # Roof panel (between front and rear windows)
    roof_color = tuple(min(255, c + 15) for c in color)
    cv2.rectangle(image, (f_w2, y1 + 4), (r_w1, y2 - 4), roof_color, -1)

    # Side mirrors
    mirror_color = border_color
    cv2.rectangle(image, (f_w1, y1 - 4), (f_w1 + 10, y1), mirror_color, -1)
    cv2.rectangle(image, (f_w1, y2), (f_w1 + 10, y2 + 4), mirror_color, -1)

    # Headlights & Taillights
    # Front lights (yellowish/white)
    cv2.rectangle(image, (x1, y1 + 2), (x1 + 6, y1 + 10), (220, 240, 255), -1)
    cv2.rectangle(image, (x1, y2 - 10), (x1 + 6, y2 - 2), (220, 240, 255), -1)
    # Taillights (red)
    cv2.rectangle(image, (x2 - 6, y1 + 2), (x2, y1 + 10), (30, 30, 220), -1)
    cv2.rectangle(image, (x2 - 6, y2 - 10), (x2, y2 - 2), (30, 30, 220), -1)

def generate_synthetic_cctv_video(output_path, duration_seconds=8, fps=25):
    """
    Generates a realistic 1280x720 overhead CCTV surveillance parking video file (MP4).
    Contains realistic asphalt markings, parking slots, static cars, moving vehicle,
    and live CCTV HUD overlay.
    """
    width = 1280
    height = 720
    total_frames = int(duration_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Car definitions in specific slots to simulate realistic scenario
    # P01: Metallic Grey Sedan
    # P03: Deep Blue SUV
    # P05: Red Hatchback
    # P09: Pearl White Sedan
    # P10: Black Compact
    # P13: Silver Sedan
    # P02: VACANT (Accessible)
    # P04: Car pulls in halfway through video!
    # P06, P08, P11, P12, P14: VACANT
    # P07: MAINTENANCE (under repair)
    
    parked_cars = {
        "P01": {"color": (160, 160, 170), "type": "Sedan"},
        "P03": {"color": (140, 70, 30), "type": "SUV"},     # BGR deep blue
        "P05": {"color": (40, 40, 210), "type": "Hatchback"}, # BGR red
        "P09": {"color": (220, 220, 225), "type": "Sedan"}, # Pearl white
        "P10": {"color": (45, 45, 50), "type": "Compact"},   # Charcoal
        "P13": {"color": (175, 180, 185), "type": "Sedan"}, # Silver
    }

    for frame_idx in range(total_frames):
        # Base Asphalt Background with slight CCTV noise/texture
        frame = np.full((height, width, 3), 32, dtype=np.uint8)
        
        # Add asphalt texture variation
        noise = np.random.randint(-3, 4, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw Drive Lane in the middle (x=500 to x=780)
        lane_x1, lane_x2 = 500, 780
        cv2.rectangle(frame, (lane_x1, 0), (lane_x2, height), (38, 38, 42), -1)

        # Draw Lane divider lines (dashed yellow)
        center_x = (lane_x1 + lane_x2) // 2
        dash_len = 30
        dash_gap = 20
        y_curr = 0
        while y_curr < height:
            cv2.line(frame, (center_x, y_curr), (center_x, min(height, y_curr + dash_len)), (0, 190, 230), 2)
            y_curr += dash_len + dash_gap

        # Draw Lane edge lines (solid white)
        cv2.line(frame, (lane_x1, 0), (lane_x1, height), (180, 180, 180), 2)
        cv2.line(frame, (lane_x2, 0), (lane_x2, height), (180, 180, 180), 2)

        # Entrance Markings at Top
        cv2.putText(frame, ">>> ENTRANCE >>>", (lane_x1 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        cv2.arrowedLine(frame, (center_x, 55), (center_x, 90), (0, 230, 255), 3, tipLength=0.35)

        # Exit Markings at Bottom
        cv2.putText(frame, ">>> EXIT GATE >>>", (lane_x1 + 30, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 100), 2)
        cv2.arrowedLine(frame, (center_x, height - 45), (center_x, height - 15), (80, 220, 100), 3, tipLength=0.35)

        # Draw Parking Bays & Markings
        for slot_id, slot in SLOT_DEFINITIONS.items():
            x1, y1, x2, y2 = slot["coords"]

            # Slot border outline (white painted line on asphalt)
            border_color = (130, 135, 140)
            if slot["is_accessible"]:
                border_color = (220, 130, 40) # Blue highlight
            elif slot["is_emergency"]:
                border_color = (200, 50, 180) # Purple highlight
            elif slot["is_maintenance"]:
                border_color = (90, 90, 90)   # Grey

            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

            # Slot Label (painted on ground)
            label = slot_id
            cv2.putText(frame, label, (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            # Special ground markings
            if slot["is_accessible"]:
                # Wheelchair accessible symbol tag (left edge)
                cv2.rectangle(frame, (x1 + 10, y1 + 32), (x1 + 70, y1 + 52), (200, 100, 30), -1)
                cv2.putText(frame, "ACC", (x1 + 18, y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            elif slot["is_emergency"]:
                # Emergency reserved symbol tag (left edge)
                cv2.rectangle(frame, (x1 + 10, y1 + 32), (x1 + 70, y1 + 52), (160, 40, 160), -1)
                cv2.putText(frame, "EMG", (x1 + 18, y1 + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            elif slot["is_maintenance"]:
                # Maintenance diagonal warning stripes on ground
                stripe_w = 20
                for sx in range(x1, x2, stripe_w * 2):
                    cv2.line(frame, (sx, y2), (min(x2, sx + stripe_w), y1), (60, 60, 60), 2)
                cv2.putText(frame, "[MAINTENANCE]", (x1 + 90, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

            # Draw static parked cars
            if slot_id in parked_cars:
                c_x = (x1 + x2) // 2
                c_y = (y1 + y2) // 2
                draw_topdown_car(frame, c_x, c_y, width=230, height=44, color=parked_cars[slot_id]["color"], car_type=parked_cars[slot_id]["type"])

        # Dynamic Event: A car drives down the central lane and parks into Slot P04!
        # Frames 0 -> 70: Drives down center lane
        # Frames 70 -> 120: Turns left into P04 and reverses into bay
        # Frames 120 -> end: Stationary parked in P04
        slot_p4 = SLOT_DEFINITIONS["P04"]["coords"]
        p4_target_x = (slot_p4[0] + slot_p4[2]) // 2
        p4_target_y = (slot_p4[1] + slot_p4[3]) // 2

        if frame_idx < 60:
            # Driving down central lane from entrance
            progress = frame_idx / 60.0
            moving_x = center_x
            moving_y = int(20 + progress * (p4_target_y - 20))
            draw_topdown_car(frame, moving_x, moving_y, width=44, height=180, color=(35, 140, 230), car_type="Sedan")
        elif frame_idx < 110:
            # Turning into Slot P04 (Simulates transition / Verifying state!)
            turn_progress = (frame_idx - 60) / 50.0
            moving_x = int(center_x + turn_progress * (p4_target_x - center_x))
            moving_y = p4_target_y
            car_w = int(44 + turn_progress * (230 - 44))
            car_h = int(180 - turn_progress * (180 - 44))
            draw_topdown_car(frame, moving_x, moving_y, width=car_w, height=car_h, color=(35, 140, 230), car_type="Sedan")
        else:
            # Fully settled in P04
            draw_topdown_car(frame, p4_target_x, p4_target_y, width=230, height=44, color=(35, 140, 230), car_type="Sedan")

        # Realistic CCTV HUD Overlay
        # 1. Top HUD Bar (translucent dark banner)
        cv2.rectangle(frame, (0, 0), (width, 36), (10, 12, 18), -1)
        cv2.line(frame, (0, 36), (width, 36), (50, 60, 80), 1)

        # Flashing Recording Dot (every 15 frames)
        if (frame_idx // 12) % 2 == 0:
            cv2.circle(frame, (20, 18), 6, (40, 40, 230), -1)
        cv2.putText(frame, "REC", (32, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        cv2.putText(frame, "CAM-04 [OVERHEAD NORTH-LOT]  |  1080p FHD 30.0 FPS", (75, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1)

        # Dynamic Timestamp
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:") + f"{(frame_idx // fps) % 60:02d} EST"
        cv2.putText(frame, curr_time, (width - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)

        # Bottom HUD Bar
        cv2.rectangle(frame, (0, height - 30), (width, height), (10, 12, 18), -1)
        cv2.line(frame, (0, height - 30), (width, height - 30), (50, 60, 80), 1)
        cv2.putText(frame, "SMARTPARK AI  ::  REAL-TIME OCCUPANCY MONITORING GRID  ::  AI MODEL: YOLOV8N", (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 180), 1)
        cv2.putText(frame, "STATUS: ONLINE - SENSOR STABLE", (width - 260, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 100), 1)

        out.write(frame)

    out.release()
    return True

def generate_analytics_dataframe(slot_states):
    """
    Creates a detailed Pandas DataFrame of current parking lot states and metrics.
    """
    records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for slot_id, state in slot_states.items():
        definition = SLOT_DEFINITIONS.get(slot_id, {})
        records.append({
            "Timestamp": timestamp,
            "Slot_ID": slot_id,
            "Zone": definition.get("zone", "A"),
            "Slot_Type": definition.get("type", "regular").upper(),
            "Current_Status": state.get("status", "AVAILABLE"),
            "Confidence_%": round(state.get("confidence", 0.0) * 100, 1),
            "Vehicle_Detected": state.get("vehicle_type", "None"),
            "Consecutive_Frames": state.get("consecutive_count", 3),
            "Distance_To_Entrance_Meters": definition.get("dist_entrance", 0.0),
            "Distance_To_Exit_Meters": definition.get("dist_exit", 0.0),
            "Is_Accessible": definition.get("is_accessible", False),
            "Is_Emergency_Reserve": definition.get("is_emergency", False),
            "Is_Maintenance": definition.get("is_maintenance", False)
        })
    return pd.DataFrame(records)

def export_analytics_csv(df):
    """
    Returns CSV bytes for download.
    """
    return df.to_csv(index=False).encode('utf-8')

def get_video_metadata(video_path):
    """
    Extracts metadata from a video file: width, height, fps, total_frames, duration, codec.
    """
    meta = {
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "total_frames": 200,
        "duration_seconds": 8.0,
        "codec": "H.264 / MP4V",
        "file_size_mb": 0.0
    }
    if not os.path.exists(video_path):
        return meta

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            meta["fps"] = round(fps, 1)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 200
            meta["total_frames"] = frames
            meta["duration_seconds"] = round(frames / max(1.0, fps), 1)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc:
                c1 = chr(fourcc & 0XFF)
                c2 = chr((fourcc >> 8) & 0XFF)
                c3 = chr((fourcc >> 16) & 0XFF)
                c4 = chr((fourcc >> 24) & 0XFF)
                meta["codec"] = f"{c1}{c2}{c3}{c4}".strip() or "MP4V"
            cap.release()
        meta["file_size_mb"] = round(os.path.getsize(video_path) / (1024 * 1024), 2)
    except Exception as e:
        print(f"Metadata extract note: {e}")

    return meta

def get_occupancy_timeline_df(current_occupied=6, total_usable=13):
    """
    Generates time-series telemetry data representing occupancy trend over video frames.
    """
    now = datetime.now()
    records = []
    # 10 time intervals over the last hour
    steps = 12
    base_occ = max(2, current_occupied - 2)
    for i in range(steps):
        t_delta = (steps - 1 - i) * 5 # minutes ago
        t_stamp = (now - pd.Timedelta(minutes=t_delta)).strftime("%H:%M")
        # progressive trend reaching current_occupied
        prog = i / float(steps - 1)
        sim_occ = int(round(base_occ + prog * (current_occupied - base_occ)))
        sim_avail = total_usable - sim_occ
        sim_pct = round((sim_occ / total_usable) * 100, 1)
        records.append({
            "Time": t_stamp,
            "Occupied": sim_occ,
            "Available": sim_avail,
            "Occupancy_Pct": sim_pct
        })
    return pd.DataFrame(records)

