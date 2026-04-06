# business/actions.py

def emergency_alert(label, confidence):
    print(f"🚨 EMERGENCY DETECTED: {label} ({confidence:.2f})")

def warning_alert(label, confidence):
    print(f"⚠️ Warning sound: {label}")

def notify_user(label, confidence):
    print(f"📱 Notification: {label}")

def monitor_event(label, confidence):
    print(f"👀 Monitoring: {label}")

def check_environment(label, confidence):
    print(f"🔎 Checking environment: {label}")

def ignore_event(label, confidence):
    print(f"✅ Ignored: {label}")
