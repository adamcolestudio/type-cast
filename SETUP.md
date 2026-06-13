# Video-wall Pi setup — golden image + clone to 8

Goal: build **one** SD card that boots straight into a fullscreen Chromium
playing its assigned synced video, then clone it to 8 and change one number per card.

---

## 0. What you need
- **Raspberry Pi OS (with Desktop), Bullseye, 32-bit (armhf).** Least-friction
  choice for a Pi 3 B+ kiosk — it's the last release built around the X11 +
  legacy video stack this board runs best. The current Imager **no longer lists
  Bullseye**: its menu now offers Trixie (default) and Bookworm ("Legacy"), and
  both default to Wayland/labwc, which would need different kiosk steps (see
  note 5). So download Bullseye directly and flash it as a **custom image**:
  ```
  https://downloads.raspberrypi.com/raspios_armhf/images/raspios_armhf-2023-05-03/2023-05-03-raspios-bullseye-armhf.img.xz
  ```
  (Last/most-patched Bullseye desktop build — the "oldstable" track has since
  rolled over to Bookworm, so this is effectively the final Bullseye.)
- Raspberry Pi Imager, an SD card, the `kiosk.sh` file, your hosted page URL.

---

## 0.5 Encode the 8 videos (on your Mac)
The Pi 3 plays these best as **H.264 / yuv420p** clips at the panel's native
**480×800**, all authored to the **same exact length** so they loop in lockstep.

> ⚠️ **Two values are still experimental — revisit them as you test:**
> - **Frame rate = 8 fps.** These are a stream of animation frames with no
>   inherent rate; 8 fps is a low-decode-load starting point. Raise it if motion
>   reads too choppy, lower it for more Pi headroom.
> - **`-tune grain`.** Preserves the subtle AI-glitch texture (stops x264 from
>   smoothing high-frequency detail away) but *raises decode load*. Confirm it
>   still plays smoothly on the Waveshare; drop it (or try `-tune film`) if not.

**Best quality — encode straight from the frame sequence** (no resampling judder):
```
ffmpeg -framerate 8 -i screen1/frame_%05d.png \
  -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p \
  -crf 16 -preset veryslow -tune grain \
  -g 32 -an -movflags +faststart \
  encoded/screen1.mp4
```

**If you only have rendered mp4s**, batch-convert (forces a constant 8 fps):
```
mkdir -p encoded
for f in screen{1..8}.mp4; do
  ffmpeg -i "$f" \
    -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p \
    -crf 16 -preset veryslow -tune grain \
    -g 32 -r 8 -an -movflags +faststart \
    "encoded/$f"
done
```
Then copy `encoded/screen1.mp4 … screen8.mp4` into `webui/videos/`.

The settled parts, and why:
- **`-crf 16 -preset veryslow`** — quality-targeted (not fixed bitrate), near-
  transparent, preserves fine detail. File size doesn't matter here.
- **`high` profile / `level 4.0`** — best detail compression; the Pi 3 hardware-
  decodes it fine. Don't drop to Main/Level 3.1 (worse fine-detail handling, and
  3.1 caps bitrate on complex frames).
- **`-g 32`** — keyframe interval (~4 s at 8 fps). Fine; lower it only if the
  sync's hard-seeks ever look laggy.
- **H.264 only** — the Pi 3 can't hardware-decode H.265/VP9/AV1 (they'd software-
  decode and stutter). **`-an`** drops audio (kiosk is muted). **`+faststart`**
  is web-friendly.
- **No `-vf scale`** — author the sources at 480×800; re-scaling softens detail.

Then verify all 8 are **identical length** (required for clean sync + `LOOP_SECONDS`):
```
for f in encoded/screen*.mp4; do ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done
```
If they match, set `LOOP_SECONDS` in `webui/index.html` to that exact value
instead of relying on auto-detect.

---

## 1. Flash the card + configure it (post-flash, on the boot partition)
Because we're using a **custom image** (§0), Imager's OS customization (the gear
/ "Edit Settings") is **greyed out** — that's expected; it only works for images
picked from Imager's own list. We apply the same settings by hand after flashing,
by dropping a few files on the card's **boot** partition (the classic Bullseye
headless method).

1. In Imager: **Choose OS → Use custom → the Bullseye `.img.xz`**, pick the SD
   card, **Write**. (Flash the `.img.xz` directly; no need to unzip.)
2. When it finishes, re-insert the card — the **boot** partition mounts as a FAT
   drive. On macOS it appears under `/Volumes/` as either `boot` or `bootfs`
   depending on the build (this Bullseye image mounts as **`/Volumes/bootfs`**) —
   use whichever name shows up in the commands below. Add these three files to it:

   **a) (Optional) Enable SSH** — only if you want to recover a wall-mounted Pi
   remotely (`pkill chromium` / `reboot` from a laptop) instead of power-cycling
   it. Not required for the kiosk to run. To enable, drop an empty file literally
   named `ssh`, no extension:
   ```
   touch /Volumes/bootfs/ssh
   ```

   **b) Create the login user** — modern Bullseye ships with **no default `pi`
   user**, so without this there's nothing to log in as (and autologin in §3
   can't work). Write `userconf.txt` as `pi:<hashed-password>`:
   ```
   echo "pi:$(echo 'password' | openssl passwd -6 -stdin)" > /Volumes/bootfs/userconf.txt
   ```

   **c) Configure Wi-Fi** — create `/Volumes/bootfs/wpa_supplicant.conf` with
   **both** networks, so the same card works at the studio now *and* the gallery
   later (it auto-picks whichever is in range). `country=` is required or the
   Pi 3 keeps Wi-Fi soft-blocked:
   ```
   country=GR
   # ^ Regulatory domain of the INSTALL site (GR=Greece). Set to wherever the Pi
   #   physically runs — a wrong code can hide channels so the AP never appears.
   #   NOTE: wpa_supplicant has NO inline comments — keep every # on its own line.
   ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
   update_config=1

   # Gallery network — OPEN (no password): no psk, use key_mgmt=NONE.
   # Tested clear of a captive portal (note 7); still confirm a Pi cold-boots to
   # video and that NTP isn't blocked during the on-site test.
   # SSID is case-sensitive — must match the broadcast name EXACTLY.
   # (The network broadcasts as all-caps "ONASSIS"; "Onassis" will NOT connect.)
   network={
       ssid="ONASSIS - Free WiFi"
       key_mgmt=NONE
   }

   # Template for a future WPA2 (password) network — uncomment + edit:
   #network={
   #    ssid="StudioWiFi"
   #    psk="studio-password"
   #}
   ```

3. Eject, boot the Pi once, let it finish first-boot + connect to Wi-Fi.

> **Hostname / locale / timezone** aren't set by these files. They're cosmetic
> here — timezone doesn't affect the clock-sync (it runs on UTC epoch time) — so
> set them later in `raspi-config` (§3) if you care. Duplicate hostnames across
> clones are fine (note 4).
>
> Alternatively, since the golden Pi has its panel + a USB keyboard attached
> anyway, you can skip (a)/(b) and just create the user via the on-screen
> first-boot wizard, then SSH in afterwards. The clones inherit the user from the
> image, so they never need this step — only `wpa_supplicant.conf` matters for
> them, and it's already baked into the golden card before cloning (§8).

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
6. **Recover a stuck Pi:** simplest is to power-cycle it (a switched power strip
   makes this painless for a wall of 8). If you enabled SSH in §1a, you can
   instead SSH in and `pkill chromium` / `sudo reboot` without touching the wall.
7. **Captive portal (checked — looks clear):** open institutional Wi-Fi like
   "Onassis - Free WiFi" *often* sits behind a captive portal (a "click to accept"
   splash page), which a kiosk Chromium in `--app` mode can't click through — the
   Pis would associate to Wi-Fi but never reach Netlify (no video) or NTP (no
   clock sync → drift). **Tested on macOS via Forget + reconnect: no portal
   appeared**, so this network looks like plain open Wi-Fi. Two things still worth
   confirming, since a laptop can't fully stand in for a headless Pi:
   - **A Pi (new MAC) cold-boots straight to playing video** with zero clicks —
     the real proof. Do this during the on-site test.
   - **NTP (UDP 123) isn't blocked** — without it the wall drifts apart even with
     video loading fine. `timedatectl` on a Pi should show "System clock
     synchronized: yes" within a minute of boot.

   If both hold, no IT involvement is needed. If a portal ever turns up, the
   durable fix is to have IT **MAC-whitelist the 8 Pis** (survives reboots,
   unlike manually clicking "accept," which most portals expire).
