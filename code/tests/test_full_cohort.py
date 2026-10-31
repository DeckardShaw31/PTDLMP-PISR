import os
import json

import pytest

from src.data.amazon_adapter import load_official_amazon_dataset
from src.data.cohort import iter_top_level_items, split_official_dataset, read_shard, build_instance_from_shard

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(RAW_DIR, "route_data.json")),
    reason="Official Amazon sample data not available"
)

def test_streaming_reader_matches_json_load():
    path = os.path.join(RAW_DIR, "travel_times.json")
    with open(path, "r", encoding="utf-8") as f:
        expected = json.load(f)
    streamed = dict(iter_top_level_items(path))
    assert list(streamed) == list(expected)
    assert streamed == expected

@pytest.mark.parametrize("threshold", [None, 4.0])
def test_shards_rebuild_identical_instances(tmp_path, threshold):
    manifest = split_official_dataset(RAW_DIR, str(tmp_path))
    assert manifest["has_shard"].all()

    reference = load_official_amazon_dataset(RAW_DIR, strict_mode=True, derived_threshold_hours=threshold)
    assert set(manifest["route_id"]) == set(reference)
    for route_id, ref in reference.items():
        inst = build_instance_from_shard(read_shard(str(tmp_path / "shards"), route_id), derived_threshold_hours=threshold)
        assert inst.depot_id == ref.depot_id
        assert inst.departure_time == ref.departure_time
        assert inst.route_date == ref.route_date
        assert inst.travel_times == ref.travel_times
        assert inst.distances == ref.distances
        assert inst.actual_sequence == ref.actual_sequence
        assert list(inst.stops) == list(ref.stops)
        for stop_id, stop in ref.stops.items():
            other = inst.stops[stop_id]
            assert (other.promised_time, other.service_seconds, other.package_volume_cm3, other.deadline_source) == \
                   (stop.promised_time, stop.service_seconds, stop.package_volume_cm3, stop.deadline_source)
