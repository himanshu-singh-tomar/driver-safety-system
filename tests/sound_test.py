
import winsound
import time

print("Testing Windows sound...")

# Play a standard Windows notification sound.
winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

time.sleep(1)

# Play a short beep at a specified frequency.
winsound.Beep(1000, 700)

print("Sound test finished.")