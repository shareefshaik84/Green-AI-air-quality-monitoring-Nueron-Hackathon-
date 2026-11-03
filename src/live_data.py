
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import math
import time

import pandas as pd
import requests


BASE = "https://api.openaq.org/v3"
GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

GEOCODING_HEADERS = {
    "User-Agent": "GreenAI-AirQuality-Intelligence/1.0"
}

MAP = {
    "pm25": "pm25",
    "pm2.5": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "nitrogen dioxide": "no2",
    "so2": "so2",
    "sulfur dioxide": "so2",
    "sulphur dioxide": "so2",
    "o3": "o3",
    "ozone": "o3",
    "co": "co",
    "carbon monoxide": "co",
}

_last_geocode_request = 0.0


# =====================================================
# COMMON API FUNCTIONS
# =====================================================

def get(path, key, params=None):

    response = requests.get(
        BASE + path,
        headers={"X-API-Key": key},
        params=params or {},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


@lru_cache(maxsize=128)
def geocode_city(city, country):

    global _last_geocode_request

    elapsed = time.monotonic() - _last_geocode_request

    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    try:

        response = requests.get(
            GEOCODING_URL,
            params={
                "q": city,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": country.lower(),
            },
            headers=GEOCODING_HEADERS,
            timeout=25,
        )

        _last_geocode_request = time.monotonic()

        response.raise_for_status()

        results = response.json()

    except requests.RequestException as error:

        raise ValueError(
            f"City geocoding service failed: {error}"
        ) from error

    if not results:

        raise ValueError(
            f"Could not find {city} in country {country}."
        )

    result = results[0]

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result.get("display_name", city),
    }


def distance_km(lat1, lon1, lat2, lon2):

    earth_radius_km = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


def _location_coordinates(location):

    coords = location.get("coordinates") or {}

    lat = coords.get("latitude")
    lon = coords.get("longitude")

    if lat is None or lon is None:
        return None

    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _reading_timestamp(item):

    dt = item.get("datetime")

    if isinstance(dt, dict):
        return dt.get("utc") or dt.get("local") or ""

    if dt:
        return str(dt)

    period = item.get("period") or {}

    period_end = period.get("datetimeTo") or {}

    if isinstance(period_end, dict):
        return (
            period_end.get("utc")
            or period_end.get("local")
            or ""
        )

    return str(period_end or "")


def _city_locations(city, country, api_key, radius_km=25):

    city_info = geocode_city(city, country)

    latitude = city_info["latitude"]
    longitude = city_info["longitude"]

    payload = get(
        "/locations",
        api_key,
        {
            "iso": country,
            "coordinates": f"{latitude:.6f},{longitude:.6f}",
            "radius": int(radius_km * 1000),
            "limit": 100,
            "page": 1,
        },
    )

    nearby = []

    for location in payload.get("results", []):

        coords = _location_coordinates(location)

        if coords is None:
            continue

        loc_lat, loc_lon = coords

        distance = distance_km(
            latitude,
            longitude,
            loc_lat,
            loc_lon,
        )

        if distance <= radius_km:

            location["_greenai_distance_km"] = distance

            nearby.append(location)

    return city_info, nearby


# =====================================================
# LIVE AIR QUALITY DATA
# =====================================================

def fetch_openaq_latest(
    city="Chennai",
    country="IN",
    api_key="",
):

    city = str(city).strip()
    country = str(country).strip().upper()

    if not api_key:
        raise ValueError("OpenAQ API key is required.")

    if not city:
        raise ValueError("Please enter a city name.")

    if len(country) != 2 or not country.isalpha():
        raise ValueError("Enter a valid two-letter country code.")

    city_info, locations = _city_locations(
        city,
        country,
        api_key,
    )

    if not locations:
        raise ValueError(
            f"No verified monitoring stations found within "
            f"25 km of {city}."
        )

    rows = []

    locations_checked = 0
    locations_with_readings = 0

    for loc in locations[:30]:

        loc_id = loc.get("id")

        if not loc_id:
            continue

        locations_checked += 1

        sensors = loc.get("sensors") or []

        sensor_map = {
            str(sensor["id"]): sensor
            for sensor in sensors
            if sensor.get("id") is not None
        }

        try:

            latest_payload = get(
                f"/locations/{loc_id}/latest",
                api_key,
                {
                    "limit": 100,
                    "page": 1,
                },
            )

        except requests.RequestException as error:

            print(f"Latest API error at {loc_id}: {error}")
            continue

        added = False

        for item in latest_payload.get("results", []):

            sensor_id = str(item.get("sensorsId", ""))

            sensor = sensor_map.get(sensor_id)

            if sensor is None:
                continue

            parameter = sensor.get("parameter") or {}

            name = str(
                parameter.get("name", "")
            ).lower().strip()

            pollutant = MAP.get(name)

            if pollutant is None:
                continue

            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue

            timestamp = _reading_timestamp(item)

            if not timestamp:
                continue

            loc_lat, loc_lon = _location_coordinates(loc)

            rows.append({
                "station": loc.get("name", str(loc_id)),
                "location_id": loc_id,
                "parameter": pollutant,
                "value": value,
                "unit": parameter.get("units", ""),
                "timestamp": timestamp,
                "latitude": loc_lat,
                "longitude": loc_lon,
                "distance_km": round(
                    loc["_greenai_distance_km"], 2
                ),
            })

            added = True

        if added:
            locations_with_readings += 1

    if not rows:
        raise ValueError(
            f"Stations were found near {city}, but no supported "
            "latest pollutant observations were available."
        )

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        ["location_id", "parameter", "timestamp", "unit"]
    )

    metadata = {
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "stations": int(df["location_id"].nunique()),
        "locations_found": len(locations),
        "locations_checked": locations_checked,
        "locations_with_readings": locations_with_readings,

        "source": "OpenAQ API v3",
        "city": city,
        "country": country,

        "resolved_location": city_info["display_name"],

        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],

        "search_radius_km": 25,
        "geocoding_source": "OpenStreetMap Nominatim",
    }

    return df, metadata


# =====================================================
# HISTORICAL PM2.5 DATA
# =====================================================

def fetch_openaq_historical(
    city="Chennai",
    country="IN",
    api_key="",
    days=365,
):

    city = str(city).strip()
    country = str(country).strip().upper()

    if not api_key:
        raise ValueError("OpenAQ API key is required.")

    if not city:
        raise ValueError("Please enter a city name.")

    if len(country) != 2 or not country.isalpha():
        raise ValueError("Enter a valid two-letter country code.")

    try:
        days = int(days)
    except (TypeError, ValueError):
        raise ValueError("Historical period must be a number of days.")

    if days < 1 or days > 365:
        raise ValueError("Historical period must be between 1 and 365 days.")

    # ---------------------------------------------
    # 1. FIND VERIFIED CITY LOCATIONS
    # ---------------------------------------------

    city_info, locations = _city_locations(
        city,
        country,
        api_key,
    )

    if not locations:
        raise ValueError(
            f"No verified monitoring stations found near {city}."
        )

    date_to = datetime.now(timezone.utc).date()
    date_from = date_to - timedelta(days=days)

    rows = []
    sensors_checked = 0
    sensors_with_data = 0
    units_found = set()

    # ---------------------------------------------
    # 2. FIND PM2.5 SENSORS AND FETCH DAILY DATA
    # ---------------------------------------------

    for loc in locations[:30]:

        sensors = loc.get("sensors") or []

        for sensor in sensors:

            sensor_id = sensor.get("id")

            if not sensor_id:
                continue

            parameter = sensor.get("parameter") or {}

            parameter_name = str(
                parameter.get("name", "")
            ).lower().strip()

            if MAP.get(parameter_name) != "pm25":
                continue

            sensors_checked += 1

            try:

                daily_payload = get(
                    f"/sensors/{sensor_id}/days",
                    api_key,
                    {
                        "date_from": date_from.isoformat(),
                        "date_to": date_to.isoformat(),
                        "limit": 1000,
                        "page": 1,
                    },
                )

            except requests.RequestException as error:

                print(
                    f"Historical API error for sensor "
                    f"{sensor_id}: {error}"
                )
                continue

            sensor_rows = []

            for item in daily_payload.get("results", []):

                value = item.get("value")

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                period = item.get("period") or {}

                datetime_to = period.get("datetimeTo") or {}

                if isinstance(datetime_to, dict):
                    timestamp = (
                        datetime_to.get("utc")
                        or datetime_to.get("local")
                    )
                else:
                    timestamp = str(datetime_to or "")

                if not timestamp:
                    datetime_from = period.get("datetimeFrom") or {}

                    if isinstance(datetime_from, dict):
                        timestamp = (
                            datetime_from.get("utc")
                            or datetime_from.get("local")
                        )

                if not timestamp:
                    continue

                unit = (
                    item.get("parameter", {}).get("units")
                    or parameter.get("units")
                    or ""
                )

                units_found.add(unit)

                sensor_rows.append({
                    "timestamp": timestamp,
                    "pm25": value,
                    "unit": unit,
                    "station": loc.get("name", str(loc.get("id"))),
                    "sensor_id": sensor_id,
                    "location_id": loc.get("id"),
                    "latitude": loc.get("coordinates", {}).get("latitude"),
                    "longitude": loc.get("coordinates", {}).get("longitude"),
                    "distance_km": round(
                        loc["_greenai_distance_km"], 2
                    ),
                })

            if sensor_rows:
                rows.extend(sensor_rows)
                sensors_with_data += 1

    # ---------------------------------------------
    # 3. VALIDATE AND PREPARE DATA
    # ---------------------------------------------

    if not rows:
        raise ValueError(
            f"No historical PM2.5 daily measurements were found "
            f"for {city} in the last {days} days. "
            "This city may not have sufficient historical coverage."
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "pm25"]
    )

    if df.empty:
        raise ValueError(
            "Historical measurements were returned, "
            "but none had valid timestamps and PM2.5 values."
        )

    # The model expects one PM2.5 value per timestamp.
    # Average readings from multiple stations for each day.
    model_df = (
        df.groupby(
            df["timestamp"].dt.floor("D"),
            as_index=False,
        )
        .agg(
            pm25=("pm25", "mean")
        )
        .rename(
            columns={"timestamp": "timestamp"}
        )
    )

    model_df = model_df.rename(
        columns={"timestamp": "timestamp"}
    )

    # Explicitly ensure the model receives the expected columns.
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)

    if len(model_df) < 40:
        raise ValueError(
            f"Only {len(model_df)} daily PM2.5 observations "
            f"were available for {city}. At least 40 are needed "
            "for the current ML model."
        )

    # ---------------------------------------------
    # 4. RETURN HISTORICAL DATA AND METADATA
    # ---------------------------------------------

    metadata = {
        "source": "OpenAQ API v3 daily measurements",
        "city": city,
        "country": country,
        "resolved_location": city_info["display_name"],
        "period_days": days,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "locations_found": len(locations),
        "sensors_checked": sensors_checked,
        "sensors_with_data": sensors_with_data,
        "daily_rows": len(model_df),
        "unit": ", ".join(
            str(unit) for unit in units_found if unit
        ),
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return model_df, metadata
