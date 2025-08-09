#!/usr/bin/env python3
"""
OpenSky Sanford Antenna-Direction Filter with Aircraft Metadata
---------------------------------------------------------------
Queries OpenSky's REST API for aircraft in a bounding box around a sensor,
filters by antenna azimuth/beam, and logs results to CSV with aircraft type info.

Features:
- config.json or CLI args (CLI overrides config)
- Auto-fetch/refresh bearer token from client_id/client_secret
- Fetch aircraft metadata (model, manufacturer, registration, typecode)
- Caches metadata in memory + persistent JSON cache
- One CSV per run or loop iteration
- Prints summary after each run

Default config.json (Sanford test):
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "sensor_lat": 28.7775,
  "sensor_lon": -81.3070,
  "antenna_azimuth": 90,
  "beam_width": 180,
  "radius_km": 15
}
"""

import os
import math
import json
import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests

CACHE_FILE = "aircraft_cache.json"


# ----------------------------- Authentication -----------------------------

def get_opensky_token(client_id: str, client_secret: str) -> Tuple[str, float]:
    url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(url, data=data, headers=headers, timeout=10)
    r.raise_for_status()
    resp = r.json()
    token = resp.get("access_token")
    expires_in = resp.get("expires_in", 300)
    if not token:
        raise RuntimeError(f"No access_token in OpenSky response: {resp}")
    expiry = time.time() + expires_in - 10
    return token, expiry


# ----------------------------- Geometry utils -----------------------------

def deg_box_from_radius_km(lat: float, radius_km: float) -> Tuple[float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return dlat, dlon


def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def is_within_beam(sensor_lat, sensor_lon, target_lat, target_lon, azimuth, beam_width) -> bool:
    if target_lat is None or target_lon is None:
        return False
    bearing = calculate_bearing(sensor_lat, sensor_lon, target_lat, target_lon)
    diff = (bearing - azimuth + 360) % 360
    return diff <= beam_width / 2 or diff >= 360 - beam_width / 2


# ----------------------------- API Calls -----------------------------

def query_opensky_states(token: str, lamin: float, lamax: float, lomin: float, lomax: float) -> Dict[str, Any]:
    url = "https://opensky-network.org/api/states/all"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_aircraft_metadata(icao24: str, token: str, cache: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Look up aircraft metadata, using in-memory + persistent cache.
    """
    if icao24 in cache:
        return cache[icao24]

    url = f"https://opensky-network.org/api/metadata/aircraft/icao/{icao24}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        result = {
            "registration": data.get("registration"),
            "manufacturericao": data.get("manufacturericao"),
            "model": data.get("model"),
            "typecode": data.get("typecode")
        }
    except Exception:
        result = {
            "registration": None,
            "manufacturericao": None,
            "model": None,
            "typecode": None
        }

    cache[icao24] = result
    return result


# ----------------------------- Filtering -----------------------------

def filter_in_beam(data: Dict[str, Any], sensor_lat: float, sensor_lon: float,
                   azimuth: float, beam_width: float,
                   min_alt: Optional[float], max_alt: Optional[float]) -> List[Dict[str, Any]]:
    states = data.get("states", []) or []
    results = []
    for s in states:
        callsign = (s[1] or "").strip() or "N/A"
        lon = s[5]
        lat = s[6]
        alt = s[7]
        if not is_within_beam(sensor_lat, sensor_lon, lat, lon, azimuth, beam_width):
            continue
        if min_alt is not None and (alt is None or alt < min_alt):
            continue
        if max_alt is not None and (alt is not None and alt > max_alt):
            continue
        bearing = calculate_bearing(sensor_lat, sensor_lon, lat, lon) if lat and lon else None
        results.append({
            "icao24": s[0],
            "callsign": callsign,
            "lat": lat,
            "lon": lon,
            "baro_alt_m": alt,
            "bearing_deg": bearing,
            "velocity_ms": s[9],
            "heading_deg": s[10],
        })
    return results


# ----------------------------- Config & CLI -----------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    client_id = args.client_id or cfg.get("client_id")
    client_secret = args.client_secret or cfg.get("client_secret")
    token = args.token or os.getenv("OPENSKY_TOKEN") or cfg.get("bearer_token")

    lat = args.lat if args.lat is not None else cfg.get("sensor_lat")
    lon = args.lon if args.lon is not None else cfg.get("sensor_lon")
    azimuth = args.azimuth if args.azimuth is not None else cfg.get("antenna_azimuth")
    beam = args.beam if args.beam is not None else cfg.get("beam_width")

    radius_km = args.radius_km if args.radius_km is not None else cfg.get("radius_km", 25.0)
    min_alt = args.min_alt if args.min_alt is not None else cfg.get("min_alt")
    max_alt = args.max_alt if args.max_alt is not None else cfg.get("max_alt")

    missing = []
    if not token and not (client_id and client_secret):
        missing.append("token OR client_id+client_secret")
    for key, value in [("lat", lat), ("lon", lon), ("azimuth", azimuth), ("beam", beam)]:
        if value is None:
            missing.append(key)
    if missing:
        raise SystemExit("Missing: " + ", ".join(missing))

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "token": token,
        "lat": float(lat),
        "lon": float(lon),
        "azimuth": float(azimuth) % 360.0,
        "beam": float(beam),
        "radius_km": float(radius_km),
        "min_alt": None if min_alt is None else float(min_alt),
        "max_alt": None if max_alt is None else float(max_alt),
    }


def print_examples():
    msg = r"""
Examples:

  # Using CLI args only (token from env var)
  export OPENSKY_TOKEN="YOUR_BEARER_TOKEN"
  python opensky_sanford_cli.py --lat 28.7775 --lon -81.3070 --azimuth 90 --beam 180 --radius-km 15

  # Using client_id/secret from config.json
  python opensky_sanford_cli.py --config config.json

  # Loop every 60 seconds
  python opensky_sanford_cli.py --config config.json --loop 60
"""
    print(msg.strip())


# ----------------------------- Main -----------------------------

def main():
    p = argparse.ArgumentParser(description="OpenSky Sanford antenna-direction filter with metadata")
    p.add_argument("--config", help="Path to config.json")
    p.add_argument("--examples", action="store_true", help="Show usage examples and exit")
    p.add_argument("--token", help="Bearer token")
    p.add_argument("--client-id", dest="client_id", help="OpenSky client_id")
    p.add_argument("--client-secret", dest="client_secret", help="OpenSky client_secret")
    p.add_argument("--lat", type=float, help="Sensor latitude")
    p.add_argument("--lon", type=float, help="Sensor longitude")
    p.add_argument("--azimuth", type=float, help="Antenna azimuth degrees")
    p.add_argument("--beam", type=float, help="Beam width degrees")
    p.add_argument("--radius-km", type=float, help="Search radius km")
    p.add_argument("--min-alt", type=float, help="Min altitude m")
    p.add_argument("--max-alt", type=float, help="Max altitude m")
    p.add_argument("--loop", type=int, default=0, help="Loop every N seconds (0 = run once)")
    args = p.parse_args()

    if args.examples:
        print_examples()
        return

    cfg = load_config(args.config)
    settings = resolve_settings(args, cfg)

    token = settings["token"]
    expiry = 0
    if settings["client_id"] and settings["client_secret"]:
        token, expiry = get_opensky_token(settings["client_id"], settings["client_secret"])

    os.makedirs("logs", exist_ok=True)

    # Load persistent cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            metadata_cache = json.load(f)
    else:
        metadata_cache = {}

    def save_cache():
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_cache, f)

    def run_once():
        nonlocal token, expiry
        if settings["client_id"] and settings["client_secret"] and time.time() >= expiry:
            token, expiry = get_opensky_token(settings["client_id"], settings["client_secret"])

        dlat, dlon = deg_box_from_radius_km(settings["lat"], settings["radius_km"])
        lamin, lamax = settings["lat"] - dlat, settings["lat"] + dlat
        lomin, lomax = settings["lon"] - dlon, settings["lon"] + dlon

        data = query_opensky_states(token, lamin, lamax, lomin, lomax)
        filtered = filter_in_beam(data, settings["lat"], settings["lon"],
                                  settings["azimuth"], settings["beam"],
                                  settings["min_alt"], settings["max_alt"])

        # Add metadata
        for ac in filtered:
            meta = get_aircraft_metadata(ac["icao24"], token, metadata_cache)
            ac.update(meta)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_name = f"logs/opensky_results_{ts}_az{int(settings['azimuth'])}_beam{int(settings['beam'])}.csv"

        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "icao24", "callsign", "lat", "lon", "baro_alt_m",
                "bearing_deg", "velocity_ms", "heading_deg",
                "registration", "manufacturericao", "model", "typecode"
            ])
            writer.writeheader()
            for row in filtered:
                writer.writerow(row)

        save_cache()

        total_states = len(data.get("states", []) or [])
        print(f"Found {total_states} total aircraft; {len(filtered)} within antenna beam.")
        print(f"Saved results to: {csv_name}\n")

    if args.loop > 0:
        while True:
            run_once()
            time.sleep(args.loop)
    else:
        run_once()


if __name__ == "__main__":
    main()
