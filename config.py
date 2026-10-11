"""
SmartPark AI - System Configuration and Constants
Theme, Slot Definitions, Coordinate Mapping, and Scoring Weights
"""

# Color Palette (Deep Space Navy Theme)
THEME_COLORS = {
    "bg_dark": "#050B2B",
    "bg_card": "rgba(13, 22, 58, 0.75)",
    "bg_glass": "rgba(18, 30, 76, 0.65)",
    "border_glass": "rgba(80, 120, 255, 0.25)",
    "neon_green": "#00FF87",       # Available
    "neon_red": "#FF3366",         # Occupied
    "neon_amber": "#FFB800",       # Verifying
    "color_maintenance": "#64748B",# Maintenance (Grey)
    "accessible_blue": "#3B82F6",  # Accessible Slot outline/tag
    "emergency_purple": "#A855F7", # Emergency Reserved outline/tag
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
    "gradient_primary": "linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%)",
    "gradient_accent": "linear-gradient(135deg, #00FF87 0%, #60EFFF 100%)",
    "gradient_danger": "linear-gradient(135deg, #FF3366 0%, #FF6584 100%)",
}

# 14 Defined Parking Slots: Zone A (P01-P07), Zone B (P08-P14)
# Coordinates map to standard 1280 x 720 video frames
SLOT_DEFINITIONS = {
    "P01": {
        "zone": "A",
        "type": "regular",
        "name": "Slot P-01",
        "coords": [130, 110, 480, 175], # [x1, y1, x2, y2]
        "dist_entrance": 12.5, # meters
        "dist_exit": 68.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P02": {
        "zone": "A",
        "type": "accessible",
        "name": "Slot P-02 (Accessible)",
        "coords": [130, 185, 480, 250],
        "dist_entrance": 8.0, # Closest to Entrance in Zone A
        "dist_exit": 62.5,
        "is_accessible": True,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P03": {
        "zone": "A",
        "type": "regular",
        "name": "Slot P-03",
        "coords": [130, 260, 480, 325],
        "dist_entrance": 17.5,
        "dist_exit": 52.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P04": {
        "zone": "A",
        "type": "regular",
        "name": "Slot P-04",
        "coords": [130, 335, 480, 400],
        "dist_entrance": 26.0,
        "dist_exit": 42.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P05": {
        "zone": "A",
        "type": "regular",
        "name": "Slot P-05",
        "coords": [130, 410, 480, 475],
        "dist_entrance": 35.0,
        "dist_exit": 32.5,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P06": {
        "zone": "A",
        "type": "regular",
        "name": "Slot P-06",
        "coords": [130, 485, 480, 550],
        "dist_entrance": 45.0,
        "dist_exit": 22.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P07": {
        "zone": "A",
        "type": "maintenance",
        "name": "Slot P-07 (Under Maintenance)",
        "coords": [130, 560, 480, 625],
        "dist_entrance": 55.0,
        "dist_exit": 12.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": True, # Permanent Maintenance
        "initial_status": "MAINTENANCE"
    },
    "P08": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-08",
        "coords": [800, 110, 1150, 175],
        "dist_entrance": 10.0,
        "dist_exit": 66.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P09": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-09",
        "coords": [800, 185, 1150, 250],
        "dist_entrance": 18.0,
        "dist_exit": 56.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P10": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-10",
        "coords": [800, 260, 1150, 325],
        "dist_entrance": 27.5,
        "dist_exit": 46.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P11": {
        "zone": "B",
        "type": "emergency",
        "name": "Slot P-11 (Emergency Reserved)",
        "coords": [800, 335, 1150, 400],
        "dist_entrance": 38.0,
        "dist_exit": 24.0, # Rapid exit access
        "is_accessible": False,
        "is_emergency": True, # Permanent Emergency Reserve
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P12": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-12",
        "coords": [800, 410, 1150, 475],
        "dist_entrance": 48.0,
        "dist_exit": 18.0,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    },
    "P13": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-13",
        "coords": [800, 485, 1150, 550],
        "dist_entrance": 57.0,
        "dist_exit": 10.5,
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "OCCUPIED"
    },
    "P14": {
        "zone": "B",
        "type": "regular",
        "name": "Slot P-14",
        "coords": [800, 560, 1150, 625],
        "dist_entrance": 65.0,
        "dist_exit": 5.2, # Closest slot to Exit
        "is_accessible": False,
        "is_emergency": False,
        "is_maintenance": False,
        "initial_status": "AVAILABLE"
    }
}

# Temporal Verification Defaults
DEFAULT_TEMPORAL_FRAMES = 3 # 3 consecutive frames required for status transition
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_FRAME_SKIP = 2 # Process every 2nd frame for smooth speed

# Emergency Categories
EMERGENCY_CATEGORIES = [
    "Medical Emergency",
    "Accessibility/Elderly",
    "Patient/Pregnancy",
    "Urgent Appointment",
    "Other Emergency"
]

# System Metadata
APP_METADATA = {
    "title": "SmartPark AI",
    "subtitle": "Automated Parking Slot Occupancy Detection System Using Overhead CCTV for Smart City Parking Management",
    "version": "v2.4-PROTOTYPE",
    "camera_id": "CCTV-NORTH-GATE-04",
    "resolution": "1080p FHD (1920x1080 downscaled to 1280x720)",
    "ai_engine": "YOLOv8n + OpenCV Temporal State Machine",
    "academic_context": "College S5 AIML Capstone Project"
}
