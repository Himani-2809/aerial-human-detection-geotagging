#!/usr/bin/env python3

"""
Real-time Human Detection and Geotagging Pipeline

- YOLO-based human detection
- MAVLink GPS + attitude fusion
- LiDAR-based ground intersection
- Temporal filtering and identity management
- Outputs geotagged detections and final person locations

Tested on Jetson Orin Nano with Pixhawk + RealSense.
"""


import os, sys, time, threading
from datetime import datetime
from math import cos, radians, degrees
from collections import defaultdict
import torch
import cv2
import pandas as pd
import numpy as np
from pymavlink import mavutil

# ================= USER CONFIG =================
MODEL_PATH = "/workspace/models/bestbest.pt"
OUTPUT_BASE_FOLDER = "./detections_07012"
CSV_OUT_BASE = "human_detections_geotagged"

DST_TIME_STR = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, f"run_{DST_TIME_STR}")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RS_COLOR_W, RS_COLOR_H, RS_FPS = 1280, 720, 30
PIXHAWK_PORT = "/dev/ttyACM0"
PIXHAWK_BAUD = 115200

MIN_CONFIDENCE = 0.25
LIDAR_ENABLED = True
GROUND_ALT_AMSL = 0.0
EARTH_R = 6378137.0

# --- FILTER PARAMS ---
TIME_WINDOW_SEC = 2.0
MIN_POINTS_FILTER = 3
PERSON_MERGE_DIST_M = 2.5
MIN_TRACK_AGE = 5
MIN_VALID_ALTITUDE_M = 3.0
OCCLUSION_GRACE_SEC = 0.3

MAX_TRACK_IDLE = 0.5  # seconds

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= TIME SYNC =================
rs_t0 = None
sys_t0 = None

# ================= LOAD YOLO =================
from ultralytics import YOLO
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    model.to("cuda:0")
    print("[INFO] Running on CUDA")
else:
    print("[WARN] CUDA not available")

# ================= DATA =================
columns = [
    "Serial_No","Track_ID","Person_ID","Frame_Number","Frame_Timestamp_s",
    "Center_X","Center_Y",
    "Geo_Lat_Raw","Geo_Lon_Raw",
    "Geo_Lat_Filtered","Geo_Lon_Filtered"
]
df = pd.DataFrame(columns=columns)

# ================= GEOMETRY =================
def pixel_to_camera_ray(u,v,K):
    x = (u-K[0,2])/K[0,0]
    y = (v-K[1,2])/K[1,1]
    r = np.array([x,y,1.0])
    return r/np.linalg.norm(r)

def rpy_to_R(roll,pitch,yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

def ned_to_enu(v):
    return np.array([v[1],v[0],-v[2]])

def enu_to_latlon(lat0,lon0,e,n):
    return (
        lat0 + degrees(n/EARTH_R),
        lon0 + degrees(e/(EARTH_R*cos(radians(lat0))))
    )

def intersect(lat,lon,alt,ray_ned,ground):
    e,n,u = ned_to_enu(ray_ned)
    if abs(u) < 1e-6: return None
    t = (ground-alt)/u
    if t < 0: return None
    return enu_to_latlon(lat,lon,e*t,n*t)

# ================= FILTERING =================
def latlon_to_enu(lat0,lon0,lat,lon):
    return (
        radians(lon-lon0)*EARTH_R*cos(radians(lat0)),
        radians(lat-lat0)*EARTH_R
    )

def median_latlon(pts):
    return float(np.median([p["lat"] for p in pts])), \
           float(np.median([p["lon"] for p in pts]))

def ransac_latlon(pts,thresh=3.0):
    if len(pts) < 3: return None
    lat0,lon0 = pts[0]["lat"], pts[0]["lon"]
    enu = np.array([latlon_to_enu(lat0,lon0,p["lat"],p["lon"]) for p in pts])
    best=[]
    for i in range(len(enu)):
        d = np.linalg.norm(enu-enu[i],axis=1)
        inl = np.where(d<thresh)[0]
        if len(inl) > len(best): best=inl
    if len(best) < 3: return None
    return median_latlon([pts[i] for i in best])

    # ================= FINAL PID ESTIMATION =================
def estimate_final_geotag(buf):
    if len(buf) < MIN_POINTS_FILTER:
        return None

    pts = [(p["lat"], p["lon"]) for p in buf]

    r = ransac_latlon(buf)
    if r is not None:
        return r

    return median_latlon(buf)


# ================= ID MANAGEMENT =================
track_to_person = {}
person_buffers = defaultdict(list)
track_age = defaultdict(int)
track_last_seen = {}
next_person_id = 1
final_pid_results = {}  # pid -> {"lat":..., "lon":..., "n":...}


def prune_buffer(buf, now):
    return [p for p in buf if now - p["t"] <= TIME_WINDOW_SEC]

def recently_seen_person(lat,lon,now,used_pids):
    for pid, buf in person_buffers.items():
        if pid in used_pids or not buf:
            continue
        last = buf[-1]
        dt = now - last["t"]
        if dt > OCCLUSION_GRACE_SEC:
            continue
        e,n = latlon_to_enu(last["lat"], last["lon"], lat, lon)
        if np.hypot(e,n) < PERSON_MERGE_DIST_M:
            return pid
    return None

def assign_person(track_id, lat, lon, now, used_pids):
    global next_person_id

    stable = track_id is not None and track_age.get(track_id,0) >= MIN_TRACK_AGE

    if stable and track_id in track_to_person:
        pid = track_to_person[track_id]
        if pid not in used_pids:
            return pid

    pid = recently_seen_person(lat,lon,now,used_pids)
    if pid is not None:
        if track_id is not None:
            track_to_person[track_id] = pid
        return pid

    if not stable:
        return None

    pid = next_person_id
    next_person_id += 1
    track_to_person[track_id] = pid
    person_buffers[pid] = []
    return pid

# ================= VALIDATION =================
def valid_detection(box, tid, gps_alt):
    if float(box.conf[0]) < MIN_CONFIDENCE:
        return False
    if gps_alt < MIN_VALID_ALTITUDE_M:
        return False
    if tid is not None and track_age[tid] < MIN_TRACK_AGE:
        return "UNSTABLE"
    return True

# ================= MAVLINK =================
gps_buf, att_buf, lidar_buf = [],[],[]

def get_nearest(buf,t,dt=0.5):
    if not buf: return None
    s = min(buf,key=lambda x:abs(x["t"]-t))
    return s if abs(s["t"]-t)<=dt else None

def mav_reader():
    while True:
        m = mav.recv_match(blocking=True,timeout=1)
        if not m: continue
        t=time.time()
        if m.get_type()=="GLOBAL_POSITION_INT":
            gps_buf.append({"t":t,"lat":m.lat/1e7,"lon":m.lon/1e7,"alt":m.alt/1000})
        elif m.get_type()=="ATTITUDE":
            att_buf.append({"t":t,"r":m.roll,"p":m.pitch,"y":m.yaw})
        elif m.get_type()=="DISTANCE_SENSOR":
            lidar_buf.append({"t":t,"d":m.current_distance/100})

# ================= START SENSORS =================
import pyrealsense2 as rs
pipe=rs.pipeline()
cfg=rs.config()
cfg.enable_stream(rs.stream.color,RS_COLOR_W,RS_COLOR_H,rs.format.bgr8,RS_FPS)
profile=pipe.start(cfg)
intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])

mav=mavutil.mavlink_connection(PIXHAWK_PORT,baud=PIXHAWK_BAUD)
mav.wait_heartbeat()
threading.Thread(target=mav_reader,daemon=True).start()

# ================= MAIN LOOP =================
serial = 1
frame = 0
try:
    while True:
        f = pipe.wait_for_frames().get_color_frame()
        img = np.asanyarray(f.get_data())
        vis = img.copy()

        rs_ts = f.get_timestamp() * 1e-3
        if rs_t0 is None:
            rs_t0 = rs_ts
            sys_t0 = time.time()
        ts = sys_t0 + (rs_ts - rs_t0)

        res = model.track(img, persist=True, verbose=False, half=True, imgsz=640)[0]
        if not res.boxes:
            frame += 1
            continue

        used_pids = set()
        detection = False

        for box in res.boxes:
            if int(box.cls[0]) != 0:
                continue

            tid = int(box.id[0]) if box.id is not None else None
            if tid is not None:
                track_age[tid] += 1
                track_last_seen[tid] = ts

            gps = get_nearest(gps_buf, ts, 5)
            att = get_nearest(att_buf, ts, 5)
            if not gps or not att:
                continue

            status = valid_detection(box, tid, gps["alt"])
            if status is False:
                continue

            l = get_nearest(lidar_buf, ts)
            if not l or l["d"] <= 0.3 or l["d"] > 50:
                continue
            ground = gps["alt"] - l["d"]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            ray = pixel_to_camera_ray(cx, cy, K)
            ray_ned = rpy_to_R(att["r"], att["p"], att["y"]) @ ray
            hit = intersect(gps["lat"], gps["lon"], gps["alt"], ray_ned, ground)
            if not hit:
                continue

            lat, lon = hit
            pid = assign_person(tid, lat, lon, ts, used_pids)
            if pid is None:
                continue

            used_pids.add(pid)
            person_buffers[pid].append({"lat": lat, "lon": lon, "t": ts})
            person_buffers[pid] = prune_buffer(person_buffers[pid], ts)

            fl, fn = lat, lon
    # Only compute filtered geotag for stable PIDs
            stable_pid = track_age.get(tid, 0) >= MIN_TRACK_AGE if tid is not None else False
            if stable_pid and len(person_buffers[pid]) >= MIN_POINTS_FILTER:
                r = ransac_latlon(person_buffers[pid])
                if r: fl, fn = r


            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"PID:{pid} TID:{tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            df.loc[len(df)] = [serial, tid, pid, frame, round(ts, 3),
                               cx, cy, lat, lon, fl, fn]
            serial += 1
            detection = True

        # ================= TRACK CLEANUP + PID FINALIZATION =================
        for tid in list(track_last_seen.keys()):
            if ts - track_last_seen[tid] > MAX_TRACK_IDLE:
                pid = track_to_person.get(tid)

                if pid is not None and pid not in final_pid_results:
                    buf = person_buffers.get(pid, [])
                    r = estimate_final_geotag(buf)
                    if r:
                        final_pid_results[pid] = {
                            "lat": r[0],
                            "lon": r[1],
                            "n": len(buf)
                        }

                track_last_seen.pop(tid, None)
                track_age.pop(tid, None)
                track_to_person.pop(tid, None)

        # Save frames
        if detection:
            cv2.imwrite(f"{OUTPUT_FOLDER}/frame_{frame:06d}.jpg", vis)

        frame += 1

except KeyboardInterrupt:
    pass
finally:
    pipe.stop()

    # Finalize any remaining PIDs
    for pid, buf in person_buffers.items():
        if pid in final_pid_results:
            continue
        r = estimate_final_geotag(buf)
        if r:
            final_pid_results[pid] = {
                "lat": r[0],
                "lon": r[1],
                "n": len(buf)
            }

    # Save final PID geotags
    final_df = pd.DataFrame([
        {
            "Person_ID": pid,
            "Final_Lat": round(v["lat"],7),
            "Final_Lon": round(v["lon"],7),
            "Num_Detections": v["n"]
        }
        for pid, v in final_pid_results.items()
    ])

    final_df.to_csv(
        f"{OUTPUT_FOLDER}/final_person_geotags_{DST_TIME_STR}.csv",
        index=False
    )

    print(f"[OK] Saved frames to {OUTPUT_FOLDER}")
    print(f"[OK] Saved final geotags to {OUTPUT_FOLDER}")
