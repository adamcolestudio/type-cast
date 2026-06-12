# Video-wall Pi setup — golden image + clone to 8

Goal: build **one** SD card that boots straight into a fullscreen Chromium
playing its assigned synced video, then clone it to 8 and change one number per card.

---

## 0. What you need
- Raspberry Pi OS **(with Desktop)** — for a Pi 3 B+, the **Bullseye (Legacy)**
  build is the least-friction choice for kiosk Chromium. (Bookworm works too;
  see the note at the end.)
- Raspberry Pi Imager, an SD card, the `kiosk.sh` file, your hosted page URL.

---

## 1. Flash the card (Raspberry Pi Imager)
In Imager, before writing, open the **gear / advanced settings** and set:
- **Hostname:** e.g. `wall` (fine if all clones share it; see note 4)
- **Enable SSH**, set a username (`pi`) + password (so you can fix things remotely)
- **Configure Wi-Fi:** the gallery SSID + password (baked in, so every clone
  auto-connects — no per-Pi Wi-Fi setup)
- **Locale / timezone**

Write the card, boot the Pi once, let it finish first-boot + connect to Wi-Fi.

---

## 2. Make the panel show 800×480 (Waveshare 5″ HDMI)
These panels usually need explicit HDMI timings or they show a wrong mode / blank.
Edit the boot config (`/boot/config.txt`, or `/boot/firmware/config.txt` on Bookworm)
and add the Waveshare-recommended lines for the **5inch HDMI LCD (H)** — typically:

```
hdmi_group=2
hdmi_mode=87
hdmi_cvt 800 480 60 6 0 0 0
hdmi_drive=2
```

Confirm the exact values on the Waveshare wiki for your panel revision, reboot,
and check it fills the screen at 800×480.

---

## 3. Auto-login + never blank
Run `sudo raspi-config`:
- **System Options → Boot / Auto Login → Desktop Autologin**
- **Display Options → Screen Blanking → Disable**

(Or non-interactively:
`sudo raspi-config nonint do_boot_behaviour B4` and
`sudo raspi-config nonint do_blanking 1`.)

## 3.5 Adjust GPU Memory Split

By default, the Raspberry Pi 3 only allocates 64MB of RAM to its graphics chip. You must increase this to give the HTML5 player enough room to process the video frame buffer:
- Open the terminal and run sudo raspi-config.
- Navigate to Performance Options (or Advanced Options depending on the OS version).
- Find GPU Memory.Change the value from 64 to 128.


(Or non-interactively)
sudo raspi-config nonint do_memory_split 128

---

## 4. Install the cursor-hider
```
sudo apt update && sudo apt install -y unclutter
```

---

## 5. Drop in the kiosk script
Copy `kiosk.sh` to `/home/pi/kiosk.sh`, edit the one `BASE_URL` line to your
hosted page, then:
```
chmod +x /home/pi/kiosk.sh
```

---

## 6. Run it on login (Bullseye / LXDE)
Create the per-user autostart so the desktop launches the kiosk:
```
mkdir -p ~/.config/lxsession/LXDE-pi
cat > ~/.config/lxsession/LXDE-pi/autostart <<'EOF'
@xset s off
@xset -dpms
@xset s noblank
@/home/pi/kiosk.sh
EOF
```

---

## 7. Set THIS card's screen number, then test
Create the screen-number file on the boot partition:
```
echo 1 | sudo tee /boot/screen.txt     # (or /boot/firmware/screen.txt on Bookworm)
```
Reboot. The Pi should come up fullscreen, fetch `?screen=1`, and start playing.
Verify with `?...&debug=1` behaviour if needed by temporarily editing BASE_URL.

**This is your golden card. Get it perfect before cloning.**

---

## 8. Clone to 8
1. Power off, remove the golden SD card.
2. Image it on another computer:
   - Linux/macOS: `sudo dd if=/dev/sdX of=wall.img bs=4M status=progress`
   - or use Raspberry Pi Imager / Win32DiskImager to *read* the card to a `.img`.
3. Write `wall.img` to 7 more cards (Imager → "Use custom image", or Etcher/dd).

---

## 9. Differentiate each card (the only per-Pi step)
For each of the 8 cards, plug it into any computer — the **boot partition mounts
as a normal FAT drive** on Windows/macOS/Linux — and edit `screen.txt` to that
card's number (`1`…`8`). No need to boot the Pi to do this.

Label each card + Pi with its number so you keep them straight at install.

---

## 10. Deploy
Insert each card into its Pi, connect panel (HDMI + power), power on. Each Pi:
auto-connects Wi-Fi → auto-logs in → runs `kiosk.sh` → reads its `screen.txt` →
opens `…/?screen=N` → downloads its clip once → loops it, clock-synced to the others.

---

## Notes / gotchas
1. **Binary name:** `kiosk.sh` auto-detects `chromium-browser` vs `chromium`.
2. **Power:** give each Pi a solid 5V/2.5–3A supply; under-volting causes flaky
   Wi-Fi and stutter. Power the panel from your USB supply, not off the Pi.
3. **First-boot load:** 8 Pis fetching video at once is the one heavy moment;
   it's fine — each snaps to the correct clock position whenever it finishes.
4. **Duplicate hostname** (from cloning) is usually harmless for kiosks; if you
   SSH in, use IP addresses. To make hostnames unique, also edit `/etc/hostname`
   per card, or set it from `screen.txt` at boot.
5. **Bookworm instead of Bullseye:** auto-login + raspi-config steps are the same,
   but the autostart in step 6 differs (it uses labwc/wayfire, not LXDE). Put the
   launch in `~/.config/wayfire.ini` `[autostart]` or `~/.config/labwc/autostart`.
   If you want zero surprises under time pressure, use **Bullseye**.
6. **Recover a stuck Pi:** SSH in (Wi-Fi + SSH are enabled) and `pkill chromium`
   or `sudo reboot`.
