# OpenSky Sanford Antenna-Direction Filter with Aircraft Metadata

This Python tool queries the [OpenSky Network REST API](https://opensky-network.org/apidoc/rest.html) for aircraft within a specified bounding box, filters them based on antenna azimuth and beam width, and logs the results to CSV.  
It also fetches **aircraft metadata** (registration, manufacturer, model, type code) and caches it locally.

The default configuration is set for a **Sanford, FL** test site with a **180° beam width** and **15 km radius**.

## Features
- **Location-based aircraft queries** using OpenSky's `/states/all` endpoint
- **Antenna beam filtering** by azimuth and beam width
- **Aircraft metadata lookup** via OpenSky's `/metadata/aircraft/icao/{icao24}` endpoint
- **Persistent metadata caching** (`aircraft_cache.json`)
- **CSV output** with one file per run or loop iteration:
```

logs/opensky\_results\_YYYY-MM-DD\_HH-MM-SS\_az{AZ}\_beam{BEAM}.csv

````
- **Auto-fetch and refresh** bearer token from `client_id` / `client_secret`
- **Configurable via JSON config file or CLI args**
- **Rich help** and `--examples` option

---

## Requirements

- Python 3.8+
- [Requests](https://pypi.org/project/requests/) library
- OpenSky API **client credentials** (`client_id` and `client_secret`)

Install dependencies:
```bash
pip install requests
````

---

## Setup

### 1. Create `config.json`

```json
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "sensor_lat": 28.7775,
  "sensor_lon": -81.3070,
  "antenna_azimuth": 90,
  "beam_width": 180,
  "radius_km": 15
}
```

* `client_id` / `client_secret` → provided by OpenSky when you register for API access
* `sensor_lat` / `sensor_lon` → location of your sensor
* `antenna_azimuth` → center of the beam in degrees (0° = north, 90° = east, etc.)
* `beam_width` → beam coverage in degrees
* `radius_km` → search radius

---

### 2. Run the script

**Single run** (uses `config.json`):

```bash
python opensky_sanford_cli.py --config config.json
```

**Looped run** (every 60 seconds):

```bash
python opensky_sanford_cli.py --config config.json --loop 60
```

---

## CSV Output

Example row:

```csv
icao24,callsign,lat,lon,baro_alt_m,bearing_deg,velocity_ms,heading_deg,registration,manufacturericao,model,typecode
a1b2c3,N123AB,28.8000,-81.3000,1524,87.4,65.2,90.0,N123AB,CESSNA,172S Skyhawk SP,C172
```

* **icao24** → unique hex identifier
* **callsign** → flight ID if present
* **lat/lon** → position in decimal degrees
* **baro\_alt\_m** → barometric altitude in meters
* **bearing\_deg** → bearing from sensor to aircraft
* **velocity\_ms** → ground speed in m/s
* **heading\_deg** → aircraft heading in degrees
* **registration** → tail number
* **manufacturericao** → ICAO manufacturer code
* **model** → model name
* **typecode** → ICAO type designator

---

## Caching

* Aircraft metadata is cached in `aircraft_cache.json`
* On subsequent runs, known aircraft are loaded from the cache instead of calling the API again

---

## Help & Examples

Show usage examples:

```bash
python opensky_sanford_cli.py --examples
```

---

## Notes

* The script requires **client credentials** for token-based authentication
* OpenSky tokens expire after \~5 minutes; the script auto-refreshes as needed
* One CSV file is created **per run or loop iteration**
* Metadata lookups are rate-limited — caching helps avoid API limits
* Default config is set for Sanford Airport, FL with a 180° east-facing beam

```

---

If you want, I can also add a **diagram** to the README showing how the antenna beam filtering works relative to the sensor location and bounding box. That can make it easier for new users to visualize what’s happening.  

Do you want me to add that diagram?
```
