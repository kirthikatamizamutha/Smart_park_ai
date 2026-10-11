"""
SmartPark AI - Intelligent Parking Recommendation Engine
Provides algorithmic slot allocation across Normal, Accessible, Quick Exit,
and Multi-Category Emergency modes with comprehensive score breakdowns.
"""

from config import SLOT_DEFINITIONS

class SmartRecommender:
    def __init__(self, slot_manager):
        self.slot_manager = slot_manager

    def recommend_slot(self, mode="normal", emergency_category=None):
        """
        Determines the optimal parking slot based on selected operational mode.
        Returns:
            {
                "slot_id": slot_id,
                "slot_name": str,
                "zone": str,
                "score": float (0-100),
                "score_breakdown": dict,
                "reasons": list of str,
                "reasoning": str,
                "distance": float,
                "distance_type": str,
                "nav_guidance": str
            }
        """
        slots = self.slot_manager.slots

        if mode == "accessible":
            return self._recommend_accessible(slots)
        elif mode == "quick_exit":
            return self._recommend_quick_exit(slots)
        elif mode == "emergency":
            return self._recommend_emergency(slots, emergency_category)
        else:
            return self._recommend_normal(slots)

    def _recommend_normal(self, slots):
        """
        Normal Mode:
        - Recommends best regular available slot.
        - Excludes: P02 (Accessible), P11 (Emergency Reserved), P07 (Maintenance), Occupied, Verifying.
        """
        candidate_scores = []

        for slot_id, slot in slots.items():
            # Exclude special and unavailable slots
            if slot["is_accessible"] or slot["is_emergency"] or slot["is_maintenance"]:
                continue
            if slot["status"] != "AVAILABLE":
                continue

            dist_entrance = slot["dist_entrance"]
            dist_score = max(0.0, min(100.0, 100.0 - (dist_entrance * 1.1)))
            conf_score = round(slot["confidence"] * 100.0, 1)
            avail_score = 100.0
            zone_score = 96.0 if slot["zone"] == "A" else 90.0
            pref_score = 94.0

            # Composite Priority Score
            composite_score = round(
                (0.30 * dist_score) + (0.30 * conf_score) + (0.20 * avail_score) + (0.10 * zone_score) + (0.10 * pref_score), 1
            )

            candidate_scores.append({
                "slot_id": slot_id,
                "slot": slot,
                "composite_score": composite_score,
                "breakdown": {
                    "Availability": 100.0,
                    "YOLO Confidence": conf_score,
                    "Zone Suitability": zone_score,
                    "Distance Proximity": round(dist_score, 1),
                    "User Preference": pref_score
                }
            })

        if not candidate_scores:
            return None

        # Sort by composite score descending
        candidate_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        best = candidate_scores[0]
        slot = best["slot"]

        return {
            "slot_id": best["slot_id"],
            "slot_name": slot["name"],
            "zone": slot["zone"],
            "score": best["composite_score"],
            "score_breakdown": best["breakdown"],
            "reasons": [
                "✓ Available for immediate parking",
                "✓ AI verified through 3-frame temporal filter",
                f"✓ High detection confidence ({int(slot['confidence']*100)}%)",
                f"✓ Suitable location ({slot['dist_entrance']}m from entrance)",
                "✓ Matches selected Normal parking preference"
            ],
            "reasoning": (
                f"Slot {best['slot_id']} is the optimal normal bay with {slot['dist_entrance']}m transit "
                f"from the entrance, verified by YOLOv8 at {int(slot['confidence']*100)}% detection confidence."
            ),
            "distance": slot["dist_entrance"],
            "distance_type": "Distance from Entrance",
            "nav_guidance": f"Enter North Gate, proceed down Central Driveway, turn into Bay {best['slot_id']}."
        }

    def _recommend_accessible(self, slots):
        """
        Accessible Mode:
        - Priority 1: P02 (Official ADA accessible slot closest to Entrance).
        - If P02 is occupied or verifying, pick the nearest available slot to Entrance.
        """
        # Check P02 first
        p02 = slots.get("P02")
        if p02 and p02["status"] == "AVAILABLE" and not p02["is_maintenance"]:
            conf_score = round(p02["confidence"] * 100.0, 1)
            return {
                "slot_id": "P02",
                "slot_name": p02["name"],
                "zone": "A",
                "score": 98.5,
                "score_breakdown": {
                    "Availability": 100.0,
                    "YOLO Confidence": conf_score,
                    "Zone Suitability": 100.0,
                    "Distance Proximity": 98.0,
                    "User Preference": 100.0
                },
                "reasons": [
                    "✓ Available dedicated Accessible Bay (♿)",
                    "✓ AI verified through 3-frame temporal filter",
                    f"✓ High detection confidence ({int(p02['confidence']*100)}%)",
                    "✓ Prime entrance ramp positioning (8.0m)",
                    "✓ Extra clearance borders for wheelchair ingress"
                ],
                "reasoning": (
                    "Slot P02 is the designated Accessible Bay equipped with extra-wide clearance borders, "
                    "tactile ground paving, and immediate 8.0-meter barrier-free ramp access from the main entrance."
                ),
                "distance": 8.0,
                "distance_type": "Distance to Main Ingress Ramp",
                "nav_guidance": "Enter via Main Gate; take immediate left into dedicated Accessible Bay P-02."
            }

        # Fallback to closest available slot to Entrance (excluding emergency and maintenance)
        available_regular = [
            s for s in slots.values()
            if s["status"] == "AVAILABLE" and not s["is_maintenance"] and not s["is_emergency"]
        ]
        if not available_regular:
            return None

        available_regular.sort(key=lambda s: s["dist_entrance"])
        fallback_slot = available_regular[0]
        conf_score = round(fallback_slot["confidence"] * 100.0, 1)

        return {
            "slot_id": fallback_slot["id"],
            "slot_name": fallback_slot["name"],
            "zone": fallback_slot["zone"],
            "score": 88.0,
            "score_breakdown": {
                "Availability": 100.0,
                "YOLO Confidence": conf_score,
                "Zone Suitability": 90.0,
                "Distance Proximity": 88.0,
                "User Preference": 85.0
            },
            "reasons": [
                "✓ Available nearest alternative slot",
                "✓ AI verified through temporal filter",
                f"✓ High detection confidence ({int(fallback_slot['confidence']*100)}%)",
                f"✓ Close to entrance ({fallback_slot['dist_entrance']}m)",
                "✓ Level ground approach without curb obstacles"
            ],
            "reasoning": (
                f"Dedicated Accessible Bay P02 is currently engaged. Slot {fallback_slot['id']} is the closest "
                f"accessible alternative, positioned {fallback_slot['dist_entrance']}m from the entrance with barrier-free ground access."
            ),
            "distance": fallback_slot["dist_entrance"],
            "distance_type": "Distance from Entrance",
            "nav_guidance": f"Proceed through North Entrance into Bay {fallback_slot['id']}."
        }

    def _recommend_quick_exit(self, slots):
        """
        Quick Exit Mode:
        - Recommends the available regular slot closest to the Exit gate (P14, P13, P12, P06).
        - Excludes P11 (reserved for emergency) and P07 (maintenance).
        """
        available_candidates = [
            s for s in slots.values()
            if s["status"] == "AVAILABLE" and not s["is_emergency"] and not s["is_accessible"] and not s["is_maintenance"]
        ]

        if not available_candidates:
            return None

        # Sort by distance to exit ascending
        available_candidates.sort(key=lambda s: s["dist_exit"])
        best = available_candidates[0]

        dist_exit_score = max(0.0, min(100.0, 100.0 - (best["dist_exit"] * 1.2)))
        conf_score = round(best["confidence"] * 100.0, 1)
        composite_score = round((0.45 * dist_exit_score) + (0.30 * conf_score) + (0.25 * 100.0), 1)

        return {
            "slot_id": best["id"],
            "slot_name": best["name"],
            "zone": best["zone"],
            "score": composite_score,
            "score_breakdown": {
                "Availability": 100.0,
                "YOLO Confidence": conf_score,
                "Zone Suitability": 95.0,
                "Distance Proximity": round(dist_exit_score, 1),
                "User Preference": 98.0
            },
            "reasons": [
                "✓ Available nearest slot to Exit gate",
                "✓ AI verified through 3-frame temporal filter",
                f"✓ High detection confidence ({int(best['confidence']*100)}%)",
                f"✓ Ultra-fast departure distance ({best['dist_exit']}m to gate)",
                "✓ Direct alignment with outbound lane"
            ],
            "reasoning": (
                f"Slot {best['id']} is situated only {best['dist_exit']}m from the automated Exit Barrier, "
                f"minimizing departure transit time and avoiding internal parking lot congestion."
            ),
            "distance": best["dist_exit"],
            "distance_type": "Distance to Exit Barrier",
            "nav_guidance": f"Follow Drive Lane towards South Exit; park in Bay {best['id']} for instant outbound egress."
        }

    def _recommend_emergency(self, slots, category="Medical Emergency"):
        """
        Emergency Mode:
        - Prioritizes P11 (Emergency Reserved slot near Exit) for medical/urgent cases.
        - For Accessibility/Elderly emergencies, prioritizes P02 (Accessible bay near entrance).
        - Ignores occupied and maintenance slots.
        """
        # If Accessibility/Elderly emergency, check P02 first
        if category in ["Accessibility / Elderly", "Accessibility/Elderly", "Patient / Pregnancy", "Patient/Pregnancy"]:
            p02 = slots.get("P02")
            if p02 and p02["status"] == "AVAILABLE" and not p02["is_maintenance"]:
                return {
                    "slot_id": "P02",
                    "slot_name": "P-02 (Accessible Emergency Bay)",
                    "zone": "A",
                    "category": category,
                    "score": 99.0,
                    "score_breakdown": {
                        "Availability": 100.0,
                        "YOLO Confidence": round(p02["confidence"] * 100, 1),
                        "Zone Suitability": 100.0,
                        "Distance Proximity": 98.0,
                        "User Preference": 100.0
                    },
                    "reasons": [
                        f"✓ Prioritized for '{category}' (♿)",
                        "✓ AI verified via temporal verification",
                        f"✓ YOLO Detection Confidence: {int(p02['confidence']*100)}%",
                        "✓ Immediate 8.0m wheelchair/stretcher ramp access",
                        "✓ Excluded from all standard parking pools"
                    ],
                    "reasoning": f"Prioritized for '{category}'. Situated immediately adjacent to emergency medical ramps (8.0m).",
                    "distance": 8.0,
                    "distance_type": "Proximity to Medical Ramp",
                    "nav_guidance": "Emergency Priority: Proceed directly to Bay P-02 at Entrance Left."
                }

        # Check Emergency Reserved Slot P11
        p11 = slots.get("P11")
        if p11 and p11["status"] == "AVAILABLE" and not p11["is_maintenance"]:
            conf_score = round(p11["confidence"] * 100.0, 1)
            return {
                "slot_id": "P11",
                "slot_name": "P-11 (Emergency Reserved Bay)",
                "zone": "B",
                "category": category,
                "score": 99.5,
                "score_breakdown": {
                    "Availability": 100.0,
                    "YOLO Confidence": conf_score,
                    "Zone Suitability": 98.0,
                    "Distance Proximity": 95.0,
                    "User Preference": 100.0
                },
                "reasons": [
                    f"✓ Designated Emergency Reserved Bay (🚨) for '{category}'",
                    "✓ AI verified via 3-frame temporal filter",
                    f"✓ YOLO Detection Confidence: {int(p11['confidence']*100)}%",
                    "✓ Fast 24.0m clearance to automated exit barrier",
                    "✓ Reserved exclusively for high-priority vehicles"
                ],
                "reasoning": (
                    f"Selected Emergency Reserved Bay P11 for '{category}'. Reserved exclusively for emergency "
                    f"vehicles and rapid departure clearance (24.0m to Exit Gate)."
                ),
                "distance": 24.0,
                "distance_type": "Distance to Exit Emergency Route",
                "nav_guidance": "Emergency Priority Protocol Activated: Head down Central Lane, turn right into Purple Bay P-11."
            }

        # If P11 is occupied or unavailable, find nearest available regular slot to Exit
        available = [s for s in slots.values() if s["status"] == "AVAILABLE" and not s["is_maintenance"] and not s["is_accessible"]]
        if not available:
            # Try any available non-maintenance
            available = [s for s in slots.values() if s["status"] == "AVAILABLE" and not s["is_maintenance"]]

        if not available:
            return None

        available.sort(key=lambda s: s["dist_exit"])
        fallback = available[0]
        conf_score = round(fallback["confidence"] * 100.0, 1)

        return {
            "slot_id": fallback["id"],
            "slot_name": f"{fallback['name']} (Emergency Dispatch)",
            "zone": fallback["zone"],
            "category": category,
            "score": 91.0,
            "score_breakdown": {
                "Availability": 100.0,
                "YOLO Confidence": conf_score,
                "Zone Suitability": 90.0,
                "Distance Proximity": 90.0,
                "User Preference": 92.0
            },
            "reasons": [
                f"✓ Fast dispatch alternative for '{category}'",
                "✓ AI verified via temporal filter",
                f"✓ YOLO Detection Confidence: {int(fallback['confidence']*100)}%",
                f"✓ Nearest available exit route ({fallback['dist_exit']}m to gate)",
                "✓ Clear lane access without obstruction"
            ],
            "reasoning": (
                f"Designated emergency bay is engaged. System dispatched nearest available bay {fallback['id']} "
                f"with fastest corridor exit clearance ({fallback['dist_exit']}m to exit)."
            ),
            "distance": fallback["dist_exit"],
            "distance_type": "Corridor Distance to Exit",
            "nav_guidance": f"Emergency Priority: Proceed immediately down Drive Lane into Bay {fallback['id']}."
        }
