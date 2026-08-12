#!/usr/bin/env python3
"""
Perseid meteor detector for a live telescope screen/window.

Watches a screen region or a specific window for the bright, short
streaks of Perseid meteors. On a detection it plays a chime, flashes
the streak on screen, logs the event (JSONL) and saves a crop image.

Usage:
  python perseid_detector.py --window "SharpCap"
  python perseid_detector.py --pick
  python perseid_detector.py --region 0,0,800,600 --sensitivity high

Keys in the live window:
  q / Esc  quit      m  mute      d  toggle detection mask
  s        save snapshot
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

SENSITIVITY = {
    "high":   dict(bright=60,  diff=25, min_area=60,  min_frames=2),
    "normal": dict(bright=80,  diff=35, min_area=120, min_frames=2),
    "low":    dict(bright=110, diff=60, min_area=300, min_frames=2),
}

MAX_TRACK_FRAMES = 100   # longer than this => slow object (plane), not a meteor
SCENE_CHANGE_FRAC = 0.25  # fraction of frame changed => camera resync/clouds
WARMUP_FRAMES = 30


# ---------------------------------------------------------------- audio

def synth_chime(sr=44100, dur=0.55, vol=0.85):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = np.sin(2 * np.pi * 880.0 * t) + 0.55 * np.sin(2 * np.pi * 1320.0 * t)
    tone *= np.exp(-t * 5.5)
    return (tone * vol).astype(np.float32)


def write_wav(path, samples, sr=44100):
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


class Sound:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self.mode = None
        if not enabled:
            return
        if sd is not None:
            self.mode = "sounddevice"
        else:
            self.mode = "afplay"
            self.wav = Path(__file__).resolve().parent / "chime.wav"
            if not self.wav.exists():
                write_wav(self.wav, synth_chime())

    def beep(self):
        if not self.enabled or self.mode is None:
            return
        with self._lock:
            try:
                if self.mode == "sounddevice":
                    sd.play(synth_chime(), 44100)
                else:
                    subprocess.Popen(
                        ["afplay", str(self.wav)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
            except Exception as e:
                print(f"[audio error] {e}", file=sys.stderr)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled


# ---------------------------------------------------------------- bridge

class EventBridge:
    """Minimal local WebSocket server (127.0.0.1) that pushes meteor
    events to a browser page (the GitHub Pages watch page) in real time."""

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, port, hello=None):
        self.port = port
        self.hello = hello or {}
        self._clients = set()
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.port))
        self._server.listen(8)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"[bridge] events at ws://127.0.0.1:{self.port}")

    def stop(self):
        if self._server:
            self._server.close()
        with self._lock:
            for c in list(self._clients):
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()

    def broadcast(self, message: dict):
        data = json.dumps(message)
        dead = []
        with self._lock:
            for c in list(self._clients):
                try:
                    self._send_text(c, data)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.discard(c)

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn):
        try:
            conn.settimeout(5)
            self._handshake(conn)
            with self._lock:
                self._clients.add(conn)
            self._send_text(conn, json.dumps(self.hello))
            conn.settimeout(60)
            while True:
                try:
                    data = conn.recv(2)
                except socket.timeout:
                    continue
                if not data:
                    break
                opcode = data[0] & 0x0F
                if opcode == 0x8:  # close
                    break
                if opcode == 0x9:  # ping -> pong
                    try:
                        conn.sendall(bytes([0x8A, 0x00]))
                    except OSError:
                        break
                self._drain(conn, data)
        except OSError:
            pass
        finally:
            with self._lock:
                self._clients.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _handshake(conn):
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(1024)
            if not chunk:
                raise OSError("no handshake")
            buf += chunk
        head, _, _ = buf.decode("latin1").partition("\r\n")
        if head.upper().startswith("OPTIONS"):
            # Private Network Access preflight from a browser on an https
            # page (e.g. GitHub Pages) talking to localhost — Chrome blocks
            # the WebSocket unless this is answered.
            conn.sendall((
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Private-Network: true\r\n"
                "Access-Control-Allow-Headers: *\r\n"
                "Access-Control-Max-Age: 86400\r\n"
                "Connection: close\r\n\r\n").encode())
            raise OSError("preflight answered")
        key = None
        for line in buf.decode("latin1").split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
        if not key:
            raise OSError("bad handshake")
        accept = base64.b64encode(hashlib.sha1((key + EventBridge.GUID).encode()).digest()).decode()
        conn.sendall((
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())

    @staticmethod
    def _drain(conn, first):
        """Skip the payload of an incoming client frame (we ignore their data)."""
        while len(first) < 2:
            chunk = conn.recv(1)
            if not chunk:
                raise OSError("closed")
            first += chunk
        b0, b1 = first[0], first[1]
        ln = b1 & 0x7F
        if ln == 126:
            ln = int.from_bytes(conn.recv(2), "big")
        elif ln == 127:
            ln = int.from_bytes(conn.recv(8), "big")
        if b1 & 0x80:
            ln += 4
        while ln > 0:
            chunk = conn.recv(min(ln, 4096))
            if not chunk:
                raise OSError("closed")
            ln -= len(chunk)

    @staticmethod
    def _send_text(conn, text):
        data = text.encode()
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += n.to_bytes(2, "big")
        else:
            header.append(127)
            header += n.to_bytes(8, "big")
        conn.sendall(bytes(header) + data)


# ---------------------------------------------------------------- capture

def find_window_region(title):
    """Locate an on-screen window by title substring, return mss region."""
    if mss is None:
        raise SystemExit("pip install mss")
    try:
        import Quartz
    except ImportError:
        raise SystemExit("pip install pyobjc-framework-Quartz (needed for --window)")
    infos = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    best, best_area = None, 0
    for w in infos:
        owner = w.get("kCGWindowOwnerName", "") or ""
        name = w.get("kCGWindowName", "") or ""
        if title.lower() in (owner + " " + name).lower():
            b = w.get("kCGWindowBounds", {})
            a = b.get("Width", 0) * b.get("Height", 0)
            if a > best_area:
                best, best_area = b, a
    if best is None:
        raise SystemExit(f"No on-screen window matching {title!r}")
    main = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    mon = mss.MSS().monitors[1]
    sx = mon["width"] / main.size.width
    sy = mon["height"] / main.size.height
    return {
        "left":   int(best["X"] * sx),
        "top":    int(best["Y"] * sy),
        "width":  int(best["Width"] * sx),
        "height": int(best["Height"] * sy),
    }


def clip_region(region, monitor):
    left = max(region["left"], monitor["left"])
    top = max(region["top"], monitor["top"])
    right = min(region["left"] + region["width"], monitor["left"] + monitor["width"])
    bottom = min(region["top"] + region["height"], monitor["top"] + monitor["height"])
    return {"left": left, "top": top,
            "width": max(0, right - left), "height": max(0, bottom - top)}


# ---------------------------------------------------------------- detector

class Tracker:
    def __init__(self, box, area):
        self.box = box          # (x, y, w, h)
        self.life = 1
        self.max_area = area

    def update(self, box, area):
        x1 = min(self.box[0], box[0])
        y1 = min(self.box[1], box[1])
        x2 = max(self.box[0] + self.box[2], box[0] + box[2])
        y2 = max(self.box[1] + self.box[3], box[1] + box[3])
        self.box = (x1, y1, x2 - x1, y2 - y1)
        self.life += 1
        self.max_area = max(self.max_area, area)


def _rects_touch(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    return not (ax1 + aw < bx1 or bx1 + bw < ax1 or ay1 + ah < by1 or by1 + bh < ay1)


class Detector:
    def __init__(self, sens="normal", bright=None, diff=None, area=None):
        p = SENSITIVITY[sens]
        self.min_bright = bright if bright is not None else p["bright"]
        self.bg_diff = diff if diff is not None else p["diff"]
        self.min_area = area if area is not None else p["min_area"]
        self.min_frames = 2
        self.bg = None
        self.tracker = None
        self.frame_no = 0

    def process(self, gray):
        """Return dict: mask, bbox (tracking now), scene_change, events fired."""
        self.frame_no += 1
        out = {"mask": None, "bbox": None, "scene_change": False, "events": []}

        if self.bg is None:
            self.bg = gray.astype(np.float32)
            return out

        diff = cv2.absdiff(gray, self.bg.astype(np.uint8))
        mask = (((diff > self.bg_diff) & (gray > self.min_bright)) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        out["mask"] = mask

        frac = float(np.mean(mask > 0))
        if self.frame_no < WARMUP_FRAMES or frac > SCENE_CHANGE_FRAC:
            self.bg = gray.astype(np.float32)
            self.tracker = None
            out["scene_change"] = frac > SCENE_CHANGE_FRAC
            return out

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= self.min_area * 0.5:
                x, y, w, h = cv2.boundingRect(c)
                if self.tracker is not None and _rects_touch(self.tracker.box, (x, y, w, h)):
                    self.tracker.update((x, y, w, h), area)
                else:
                    self._finalize(out)
                    self.tracker = Tracker((x, y, w, h), area)
                out["bbox"] = self.tracker.box
        else:
            self._finalize(out)
            self.tracker = None

        if self.tracker is None:
            self.bg = self.bg * 0.92 + gray.astype(np.float32) * 0.08
        return out

    def _finalize(self, out):
        t = self.tracker
        if t is None:
            return
        if (t.life >= self.min_frames and t.max_area >= self.min_area
                and t.life <= MAX_TRACK_FRAMES):
            out["events"].append({
                "box": t.box,
                "area": t.max_area,
                "duration_frames": t.life,
                "frame": self.frame_no,
            })
        self.tracker = None


# ---------------------------------------------------------------- overlay

def draw_overlay(frame, view, flashes, tonight, total, muted, show_mask,
                 scene_ts, fps, now):
    h, w = frame.shape[:2]
    if view.get("bbox"):
        x, y, bw, bh = view["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    for t0, box in flashes:
        age = now - t0
        if age > 2.5:
            continue
        x, y, bw, bh = box
        x2, y2 = x + bw, y + bh
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
        cx, cy = x + bw // 2, y + bh // 2
        r = int(30 + age * 120)
        cv2.circle(frame, (cx, cy), min(r, max(w, h)), (0, 0, 255), 2)
        cv2.putText(frame, "METEOR!", (cx - 60, max(cy - 40, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if now - scene_ts < 1.5:
        cv2.putText(frame, "scene change / clouds", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, f"Tonight {tonight}   All-time {total}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{'MUTED' if muted else 'ON'}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 120, 255) if muted else (255, 255, 255), 2)
    cv2.putText(frame, now.strftime("%H:%M:%S"), (w - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if show_mask and view.get("mask") is not None:
        cv2.imshow("mask", view["mask"])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", help="capture an on-screen window by title substring")
    ap.add_argument("--region", help="capture region as X,Y,W,H")
    ap.add_argument("--pick", action="store_true", help="drag-select the region on screen")
    ap.add_argument("--sensitivity", choices=list(SENSITIVITY), default="normal")
    ap.add_argument("--bright", type=int, help="override minimum streak brightness (0-255)")
    ap.add_argument("--diff", type=int, help="override background-difference threshold")
    ap.add_argument("--area", type=int, help="override minimum streak area (px)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="run headless (log + audio only)")
    ap.add_argument("--bridge-port", type=int, default=0,
                    help="broadcast events to browsers via a local WebSocket on this port "
                         "(e.g. 8765; the GitHub Pages page connects to it)")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "perseid_data"))
    args = ap.parse_args()

    if mss is None:
        raise SystemExit("pip install -r requirements.txt")

    data_dir = Path(args.data_dir)
    (data_dir / "crops").mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "config.json"
    stats_path = data_dir / "stats.json"
    jsonl_path = data_dir / f"events-{date.today().isoformat()}.jsonl"

    stats = {"all_time": 0}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except Exception:
            pass
    tonight = 0
    if jsonl_path.exists():
        tonight = sum(1 for _ in jsonl_path.open())

    with mss.MSS() as sct:
        monitor = sct.monitors[1]

        region = None
        if args.region:
            x, y, w, h = map(int, args.region.split(","))
            region = {"left": x, "top": y, "width": w, "height": h}
        elif args.window:
            retry_delay = float(os.environ.get("PERSEID_WINDOW_RETRY", "30"))
            while True:
                try:
                    region = find_window_region(args.window)
                    break
                except SystemExit as e:
                    if retry_delay <= 0:
                        raise
                    print(f"[init] {e} — retrying in {retry_delay:.0f}s…", file=sys.stderr)
                    time.sleep(retry_delay)
        elif args.pick:
            frame = np.array(sct.grab(monitor))[:, :, :3][:, :, ::-1]
            cv2.imshow("Drag to select region, press ENTER (ESC = cancel)", frame)
            r = cv2.selectROI("Drag to select region, press ENTER (ESC = cancel)", frame,
                              showCrosshair=True)
            cv2.destroyAllWindows()
            if r[2] > 20 and r[3] > 20:
                region = {"left": monitor["left"] + r[0], "top": monitor["top"] + r[1],
                          "width": r[2], "height": r[3]}
        elif config_path.exists():
            try:
                saved = json.loads(config_path.read_text())
                region = saved.get("region")
            except Exception:
                region = None

        if region is None:
            region = dict(monitor)  # fall back to the whole main display
        region = clip_region(region, monitor)
        if region["width"] < 32 or region["height"] < 32:
            raise SystemExit("Captured region too small")

        config_path.write_text(json.dumps({"region": region}, indent=2))
        print(f"[init] capturing {region['width']}x{region['height']} at "
              f"({region['left']},{region['top']}) -> {data_dir}")
        print(f"[init] sensitivity={args.sensitivity} bright={args.bright or SENSITIVITY[args.sensitivity]['bright']} "
              f"diff={args.diff or SENSITIVITY[args.sensitivity]['diff']} "
              f"area={args.area or SENSITIVITY[args.sensitivity]['min_area']}")
        print("[init] keys: q quit | m mute | d mask | s snapshot")

        detector = Detector(args.sensitivity, args.bright, args.diff, args.area)
        sound = Sound(enabled=not args.no_audio)
        bridge = None
        if args.bridge_port:
            bridge = EventBridge(
                args.bridge_port,
                hello={"type": "hello", "tonight": tonight,
                       "all_time": stats.get("all_time", 0)},
            )
            bridge.start()
        flashes = deque(maxlen=8)
        show_mask = False
        scene_ts = 0.0
        fps = 0.0
        last_t = time.time()

        try:
            while True:
                t0 = time.time()
                shot = sct.grab(region)
                frame = np.array(shot)[:, :, :3][:, :, ::-1].copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                view = detector.process(gray)
                if view["scene_change"]:
                    scene_ts = t0

                for ev in view["events"]:
                    now = datetime.now()
                    x, y, bw, bh = ev["box"]
                    x1, y1 = max(0, x - 10), max(0, y - 10)
                    x2 = min(frame.shape[1], x + bw + 10)
                    y2 = min(frame.shape[0], y + bh + 10)
                    crop_name = f"crop_{now.strftime('%Y%m%d_%H%M%S')}_f{ev['frame']}.png"
                    crop_path = data_dir / "crops" / crop_name
                    cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])

                    record = {
                        "type": "meteor",
                        "ts": now.isoformat(),
                        "unix": now.timestamp(),
                        "frame": ev["frame"],
                        "bbox": [x, y, bw, bh],
                        "area": ev["area"],
                        "duration_frames": ev["duration_frames"],
                        "crop": crop_name,
                    }
                    with jsonl_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")

                    stats["all_time"] = stats.get("all_time", 0) + 1
                    stats_path.write_text(json.dumps(stats))
                    tonight += 1
                    flashes.append((t0, ev["box"]))
                    sound.beep()
                    if bridge:
                        bridge.broadcast(record)
                    print(f"[{now.strftime('%H:%M:%S')}] METEOR  area={ev['area']} "
                          f"life={ev['duration_frames']}f bbox={ev['box']} crop={crop_name}")

                if not args.no_window:
                    draw_overlay(frame, view, flashes, tonight, stats.get("all_time", 0),
                                 not sound.enabled, show_mask, scene_ts, fps, datetime.now())
                    cv2.imshow("Perseid Detector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("m"):
                        print(f"[ui] audio {'ON' if sound.toggle() else 'MUTED'}")
                    elif key == ord("d"):
                        show_mask = not show_mask
                    elif key == ord("s"):
                        snap = data_dir / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(str(snap), frame)
                        print(f"[ui] snapshot {snap}")

                dt = time.time() - t0
                fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
                time.sleep(max(0.0, 1.0 / 30 - (time.time() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            if bridge:
                bridge.stop()
            if not args.no_window:
                cv2.destroyAllWindows()
            print(f"[done] tonight: {tonight} | all-time: {stats.get('all_time', 0)}")


if __name__ == "__main__":
    main()
