"""
Full-cohort support for the official Amazon Last Mile Routing Research Challenge (ALMRRC 2021) training set.

The official travel_times.json (1.8 GB) cannot be json.load-ed alongside thousands of RouteInstance objects,
so the dataset is split once into one gzip-compressed shard per route. Each shard keeps the four official
records of that route unchanged, keyed by route_id, so build_route_instance() sees exactly what
load_official_amazon_dataset() would see.
"""
import os
import re
import gzip
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .amazon_adapter import build_route_instance
from .schemas import RouteInstance

_WS = re.compile(r"\s*")

OFFICIAL_FILES = ["route_data.json", "package_data.json", "travel_times.json", "actual_sequences.json"]

def iter_top_level_items(path: str) -> Iterator[Tuple[str, Any]]:
    """
    Yields (key, value) for each member of a top-level JSON object without materialising the whole object.
    Peak memory is the file text plus one decoded value.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    decoder = json.JSONDecoder()
    i = _WS.match(text, 0).end()
    if text[i] != "{":
        raise ValueError(f"'{path}' does not contain a top-level JSON object.")
    i += 1
    while True:
        i = _WS.match(text, i).end()
        if text[i] == "}":
            return
        key, i = decoder.raw_decode(text, i)
        i = _WS.match(text, i).end()
        if text[i] != ":":
            raise ValueError(f"Malformed JSON object in '{path}' at offset {i}.")
        i = _WS.match(text, i + 1).end()
        value, i = decoder.raw_decode(text, i)
        yield key, value
        i = _WS.match(text, i).end()
        if text[i] == ",":
            i += 1

def shard_path(shard_dir: str, route_id: str) -> str:
    return os.path.join(shard_dir, f"{route_id}.json.gz")

def split_official_dataset(source_dir: str, cohort_dir: str) -> pd.DataFrame:
    """
    Splits the four official files in source_dir into per-route shards under cohort_dir/shards and writes
    cohort_dir/manifest.csv (one row per route). Existing shards are overwritten.
    """
    for fname in OFFICIAL_FILES:
        if not os.path.exists(os.path.join(source_dir, fname)):
            raise FileNotFoundError(f"Official file '{fname}' not found in '{source_dir}'.")
    shard_dir = os.path.join(cohort_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    with open(os.path.join(source_dir, "route_data.json"), "r", encoding="utf-8") as f:
        routes = json.load(f)
    with open(os.path.join(source_dir, "actual_sequences.json"), "r", encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"[Cohort] route_data: {len(routes)} routes; actual_sequences: {len(sequences)} routes.")

    # Packages are kept as compact JSON text per route to bound memory while travel times are streamed.
    packages: Dict[str, str] = {}
    for route_id, value in iter_top_level_items(os.path.join(source_dir, "package_data.json")):
        packages[route_id] = json.dumps(value, separators=(",", ":"))
    print(f"[Cohort] package_data: {len(packages)} routes.")

    written = 0
    for route_id, travel in iter_top_level_items(os.path.join(source_dir, "travel_times.json")):
        if route_id not in routes:
            continue
        pkg_text = packages.get(route_id, "{}")
        payload = (
            '{"route_id":' + json.dumps(route_id)
            + ',"route_data":' + json.dumps(routes[route_id], separators=(",", ":"))
            + ',"package_data":' + pkg_text
            + ',"travel_times":' + json.dumps(travel, separators=(",", ":"))
            + ',"actual_sequences":' + json.dumps(sequences.get(route_id), separators=(",", ":"))
            + "}"
        )
        with gzip.open(shard_path(shard_dir, route_id), "wt", encoding="utf-8", compresslevel=6) as out:
            out.write(payload)
        written += 1
        if written % 500 == 0:
            print(f"[Cohort] {written} shards written...", flush=True)

    rows = []
    for route_id, r_info in routes.items():
        stops = r_info.get("stops", {})
        rows.append({
            "route_id": route_id,
            "route_date": r_info.get("date_YYYY_MM_DD"),
            "station_code": r_info.get("station_code"),
            "route_score": r_info.get("route_score"),
            "departure_time_utc": r_info.get("departure_time_utc"),
            "num_stops": len(stops),
            "num_dropoffs": sum(1 for s in stops.values() if s.get("type") != "Station"),
            "has_packages": route_id in packages,
            "has_actual_sequence": route_id in sequences,
            "has_shard": os.path.exists(shard_path(shard_dir, route_id))
        })
    manifest = pd.DataFrame(rows).sort_values(["route_date", "route_id"]).reset_index(drop=True)
    manifest.to_csv(os.path.join(cohort_dir, "manifest.csv"), index=False)
    print(f"[Cohort] Wrote {written} shards and manifest for {len(manifest)} routes to '{cohort_dir}'.")
    return manifest

def read_shard(shard_dir: str, route_id: str) -> Dict[str, Any]:
    with gzip.open(shard_path(shard_dir, route_id), "rt", encoding="utf-8") as f:
        return json.load(f)

def build_instance_from_shard(
    shard: Dict[str, Any],
    derived_threshold_hours: Optional[float],
    strict_mode: bool = True
) -> RouteInstance:
    """Builds the RouteInstance exactly as load_official_amazon_dataset() would for this route."""
    route_id = shard["route_id"]
    packages = {route_id: shard["package_data"]} if shard.get("package_data") is not None else {}
    travel = {route_id: shard["travel_times"]} if shard.get("travel_times") is not None else {}
    seqs = {route_id: shard["actual_sequences"]} if shard.get("actual_sequences") is not None else {}
    return build_route_instance(
        route_id, shard["route_data"], packages, travel, seqs,
        strict_mode=strict_mode, derived_threshold_hours=derived_threshold_hours
    )

def write_cohort_subset_as_official_dir(shard_dir: str, route_ids: List[str], out_dir: str) -> None:
    """Writes the four official files for a route subset (used to check shard fidelity against the adapter)."""
    os.makedirs(out_dir, exist_ok=True)
    merged = {"route_data": {}, "package_data": {}, "travel_times": {}, "actual_sequences": {}}
    for route_id in route_ids:
        shard = read_shard(shard_dir, route_id)
        for key in merged:
            if shard.get(key) is not None:
                merged[key][route_id] = shard[key]
    for key, value in merged.items():
        with open(os.path.join(out_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(value, f)
