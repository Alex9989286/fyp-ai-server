# business/rules.py

SOUND_RULES = {

    "siren": {
        "action": "EMERGENCY_ALERT",
        "priority": 3,
        "threshold": 0.70
    },

    "car_horn": {
        "action": "WARNING",
        "priority": 2,
        "threshold": 0.65
    },

    "door_wood_knock": {
        "action": "CHECK",
        "priority": 1,
        "threshold": 0.60
    },

    "clock_alarm": {
        "action": "NOTIFY",
        "priority": 1,
        "threshold": 0.60
    },

    "footsteps": {
        "action": "MONITOR",
        "priority": 1,
        "threshold": 0.55
    },

    "dog": {
        "action": "IGNORE",
        "priority": 0,
        "threshold": 0.50
    }
}
