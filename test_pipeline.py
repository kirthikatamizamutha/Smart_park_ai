"""
SmartPark AI - Comprehensive Pipeline Verification Script
Tests end-to-end video reading, YOLO/vision detection, slot mapping,
temporal verification, and smart recommendations.
"""

import os
import cv2
from config import SLOT_DEFINITIONS
from detector import VehicleDetector
from slot_manager import SlotManager
from recommender import SmartRecommender
from utils import generate_analytics_dataframe

def test_pipeline():
    print("Testing pipeline initialization...")
    video_path = os.path.join(os.path.dirname(__file__), "assets", "demo_parking_video.mp4")
    assert os.path.exists(video_path), f"Video missing: {video_path}"

    detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.45)
    slot_mgr = SlotManager(temporal_frames=3)
    recommender = SmartRecommender(slot_mgr)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Cannot open video"

    frames_tested = 0
    while frames_tested < 15:
        ret, frame = cap.read()
        if not ret:
            break
        frames_tested += 1

        # Resize
        h, w = frame.shape[:2]
        if w != 1280 or h != 720:
            frame = cv2.resize(frame, (1280, 720))

        # Detect
        detections = detector.detect_vehicles(frame)
        # Map
        raw_mappings = detector.map_detections_to_slots(detections)
        # Update temporal state
        slot_mgr.update_frame(raw_mappings, enable_temporal=True)
        # Annotate
        annotated = detector.annotate_frame(frame, detections, slot_mgr.slots)
        assert annotated.shape == frame.shape, "Annotated frame shape mismatch"

    cap.release()
    print(f"Processed {frames_tested} frames successfully.")

    # Check KPIs
    kpi = slot_mgr.get_kpi_summary()
    print("KPI summary:", kpi)
    assert kpi["total_slots"] == 14, "Total slots should be 14"
    assert kpi["maintenance"] == 1, "P07 should be maintenance"

    # Test Recommendations
    rec_normal = recommender.recommend_slot("normal")
    print("Normal mode recommendation:", rec_normal["slot_id"] if rec_normal else "None")
    assert rec_normal is not None, "Normal recommendation should succeed"
    assert rec_normal["slot_id"] not in ["P02", "P07", "P11"], "Normal mode cannot pick P02, P07, or P11"

    rec_acc = recommender.recommend_slot("accessible")
    print("Accessible mode recommendation:", rec_acc["slot_id"] if rec_acc else "None")
    assert rec_acc is not None, "Accessible recommendation should succeed"

    rec_exit = recommender.recommend_slot("quick_exit")
    print("Quick Exit mode recommendation:", rec_exit["slot_id"] if rec_exit else "None")
    assert rec_exit is not None, "Quick exit recommendation should succeed"

    rec_emg = recommender.recommend_slot("emergency", emergency_category="Medical Emergency")
    print("Emergency mode recommendation:", rec_emg["slot_id"] if rec_emg else "None")
    assert rec_emg is not None, "Emergency recommendation should succeed"

    # Test DataFrame
    df = generate_analytics_dataframe(slot_mgr.slots)
    assert len(df) == 14, "DataFrame should contain exactly 14 records"
    print("DataFrame generated with columns:", list(df.columns))

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
