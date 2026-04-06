from .rules import SOUND_RULES
from . import actions

def decide_and_execute(label, confidence):
    if label not in SOUND_RULES:
        print(f"Unknown sound detected: {label}")
        return

    rule = SOUND_RULES[label]
    action = rule["action"]

    # 不再完全忽略低信心
    if confidence < rule["threshold"]:
        print(f"⚠️ Low confidence sound: {label} ({confidence:.2f})")
    else:
        print(f"✅ Detected: {label} ({confidence:.2f})")

    # 决策路由
    if action == "EMERGENCY_ALERT":
        actions.emergency_alert(label, confidence)

    elif action == "WARNING":
        actions.warning_alert(label, confidence)

    elif action == "NOTIFY":
        actions.notify_user(label, confidence)

    elif action == "MONITOR":
        actions.monitor_event(label, confidence)

    elif action == "CHECK":
        actions.check_environment(label, confidence)

    else:
        actions.ignore_event(label, confidence)
