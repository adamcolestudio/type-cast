#!/bin/bash
# kiosk.sh — launch one synced video-wall screen in Chromium kiosk mode.
# Place at /home/pi/kiosk.sh, make executable: chmod +x /home/pi/kiosk.sh
# Identical on every Pi; each Pi differs only by /boot/screen.txt (the number).

# ====== EDIT THIS ONE LINE ======
BASE_URL="https://YOUR-SITE.netlify.app/"
# ================================

# --- which screen am I? read a number from the boot (FAT) partition ---
# (Bullseye: /boot/screen.txt   Bookworm: /boot/firmware/screen.txt)
SCREENFILE=/boot/screen.txt
[ -f /boot/firmware/screen.txt ] && SCREENFILE=/boot/firmware/screen.txt
N=$(tr -dc '0-9' < "$SCREENFILE" 2>/dev/null)
[ -z "$N" ] && N=1
URL="${BASE_URL}?screen=${N}"

# --- wait for network: the page must fetch its video on first load ---
for i in $(seq 1 30); do
  ping -c1 -W1 8.8.8.8 >/dev/null 2>&1 && break
  sleep 2
done

# --- stop the screen ever blanking / sleeping ---
xset s off
xset -dpms
xset s noblank

# --- hide the mouse cursor ---
unclutter -idle 0 -root &

# --- neutralise Chromium's "didn't shut down correctly" restore bar ---
# (so a yanked power cord doesn't leave a dialog on screen next boot)
for PREF in "$HOME/.config/chromium/Default/Preferences"; do
  if [ -f "$PREF" ]; then
    sed -i 's/"exited_cleanly":false/"exited_cleanly":true/' "$PREF"
    sed -i 's/"exit_type":"[^"]*"/"exit_type":"Normal"/' "$PREF"
  fi
done

# --- find the chromium binary (name differs across OS versions) ---
CHROME=$(command -v chromium-browser || command -v chromium)

# --- launch kiosk ---
exec "$CHROME" \
  --kiosk \
  --start-fullscreen \
  --noerrordialogs \
  --disable-infobars \
  --disable-translate \
  --disable-features=TranslateUI \
  --disable-session-crashed-bubble \
  --disable-component-update \
  --check-for-update-interval=31536000 \
  --no-first-run \
  --autoplay-policy=no-user-gesture-required \
  --app="$URL"
