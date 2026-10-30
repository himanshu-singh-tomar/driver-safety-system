
import time
from src.alerts import AlertManager

alerts = AlertManager(cooldown_seconds=2.0)

print("First alert: you should hear a beep.")
alerts.trigger_drowsiness_alert()

time.sleep(0.5)

print("Second immediate alert: should be suppressed.")
alerts.trigger_drowsiness_alert()

time.sleep(2.0)

print("Third alert: you should hear another beep.")
alerts.trigger_drowsiness_alert()

time.sleep(1.0)
print("Alert test complete.")