
import time
import threading

try:
    import winsound
except ImportError:
    winsound = None


class AlertManager:
    """Plays audible drowsiness alerts with a cooldown."""

    def __init__(self, cooldown_seconds=2.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0
        self._lock = threading.Lock()

    def _play_sound(self):
        try:
            if winsound is not None:
                winsound.Beep(1000, 500)
        except RuntimeError as exc:
            print(f"Could not play drowsiness alert: {exc}")

    def trigger_drowsiness_alert(self):
        now = time.monotonic()

        with self._lock:
            if now - self.last_alert_time < self.cooldown_seconds:
                return

            # Record the trigger time before starting the sound.
            self.last_alert_time = now

        print("DROWSINESS ALERT: Playing audible warning.")
        threading.Thread(
            target=self._play_sound,
            daemon=True,
        ).start()