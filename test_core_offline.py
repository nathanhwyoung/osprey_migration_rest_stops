# -*- coding: utf-8 -*-

"""
test_core_offline.py  (reference version, 2026-08-20)

runs the whole stopover_core chain against the Movebank CSVs, no arcpy,
and checks every number against BASELINE.txt

the loading here mirrors detect_stopovers.py steps 1-3, except that the
season filter and the no-origin drop are left to find_candidate_fixes,
which is why the intermediate counts differ from the original's
"""

import os

import pandas as pd

import stopover_core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "Osprey in North and South America 1995-2002 (Martell).csv")
REF_FILE = os.path.join(HERE, "Osprey in North and South America 1995-2002 (Martell)-reference-data.csv")

# baseline parameters, from the constants at the top of detect_stopovers.py
SEASON_START_MONTH = 8
SEASON_END_MONTH = 12
WINDOW_DAYS = 7
FLAT_THRESHOLD_KM = 100
MIN_STOPOVER_DAYS = 3
MIN_DEPARTURE_KM = 100
CLUSTER_RADIUS_KM = 75
MIN_CLUSTER_FIXES = 3
MAX_STOPOVER_DAYS = 45
MIN_CLUSTER_DISPLACEMENT_KM = 250
SHARED_RADIUS_KM = 50


def load_tracks():
    """
    read both CSVs and return the df stopover_core expects:
    oid, id, timestamp, lat, lon, origin_lat, origin_lon

    stands in for _read_points in the .pyt
    """
    tracks = pd.read_csv(DATA_FILE, low_memory=False)
    ref = pd.read_csv(REF_FILE)

    # force to str: pandas reads numeric-looking ids as int, which breaks the
    # merge below if the other side is str
    for col in ["tag-local-identifier", "individual-local-identifier"]:
        tracks[col] = tracks[col].astype(str)
    for col in ["tag-id", "animal-id"]:
        ref[col] = ref[col].astype(str)

    # 41 of 80 tags were redeployed on a second bird, so tag alone is not a key
    tracks["deployment_key"] = (
        tracks["tag-local-identifier"] + "_" + tracks["individual-local-identifier"]
    )

    # movebank's own quality flag, drops Z-class fixes
    tracks = tracks[tracks["visible"].astype(str).str.upper() == "TRUE"].copy()
    tracks = tracks.dropna(subset=["location-lat", "location-long"])

    ref_coords = ref[
        ["tag-id", "animal-id", "deploy-on-latitude", "deploy-on-longitude"]
    ].copy()

    tracks = tracks.merge(
        ref_coords,
        left_on=["tag-local-identifier", "individual-local-identifier"],
        right_on=["tag-id", "animal-id"],
        how="left",
    )

    # note: rows with no deployment coords are NOT dropped here. find_candidate_fixes
    # drops them and reports the count, which is what fixes_dropped_no_origin is
    df = pd.DataFrame({
        "oid": range(len(tracks)),
        "id": tracks["deployment_key"].values,
        "timestamp": pd.to_datetime(tracks["timestamp"].values),
        "lat": tracks["location-lat"].values,
        "lon": tracks["location-long"].values,
        "origin_lat": tracks["deploy-on-latitude"].values,
        "origin_lon": tracks["deploy-on-longitude"].values,
    })

    return df.sort_values(["id", "timestamp"]).reset_index(drop=True)


def check(label, got, want, results):
    """
    record one comparison and print it
    """
    status = "MATCH" if got == want else "MISMATCH"
    print(f"  {label:<28} {got:>8}  expected {want:>8}   {status}")
    results.append(status == "MATCH")


def main():
    results = []

    print("=" * 62)
    print("STOPOVER CORE, OFFLINE CHECK")
    print("=" * 62)

    print("\n[1] loading")
    df = load_tracks()
    print(f"  rows loaded                  {len(df):>8}")

    print("\n[2] find_candidate_fixes")
    candidates, scored, info = stopover_core.find_candidate_fixes(
        df,
        origin_mode="fields",
        origin_days=7,
        season_start_month=SEASON_START_MONTH,
        season_end_month=SEASON_END_MONTH,
        window_days=WINDOW_DAYS,
        flat_threshold_km=FLAT_THRESHOLD_KM,
        min_stopover_days=MIN_STOPOVER_DAYS,
        min_departure_km=MIN_DEPARTURE_KM,
    )

    check("fixes_in", info["fixes_in"], 44995, results)
    check("individuals_in", info["individuals_in"], 127, results)
    check("fixes_dropped_no_origin", info["fixes_dropped_no_origin"], 1350, results)
    check("fixes_after_season", info["fixes_after_season"], 21812, results)
    check("candidate_fixes", info["candidate_fixes"], 10145, results)
    check("individuals_with_candidates", info["individuals_with_candidates"], 89, results)
    print(f"  individuals_without_origin   {info['individuals_without_origin']}")

    print("\n[3] build_stopovers")
    stopovers, clustered, cluster_info = stopover_core.build_stopovers(
        candidates,
        cluster_radius_km=CLUSTER_RADIUS_KM,
        min_cluster_fixes=MIN_CLUSTER_FIXES,
        max_stopover_days=MAX_STOPOVER_DAYS,
        min_cluster_displacement_km=MIN_CLUSTER_DISPLACEMENT_KM,
        shared_radius_km=SHARED_RADIUS_KM,
    )

    check("clusters_found", cluster_info["clusters_found"], 138, results)
    check("clusters_dropped_duration", cluster_info["clusters_dropped_duration"], 49, results)
    check("clusters_dropped_displacement", cluster_info["clusters_dropped_displacement"], 7, results)
    check("stopovers_final", cluster_info["stopovers_final"], 82, results)
    check("individuals_with_stopovers", cluster_info["individuals_with_stopovers"], 60, results)
    check("shared_sites", cluster_info["shared_sites"], 26, results)
    check("max_individuals_at_one_site", cluster_info["max_individuals_at_one_site"], 5, results)
    print(f"  noise_fixes                  {cluster_info['noise_fixes']:>8}")

    print("\n[4] attach_cluster_labels")
    out_fixes = stopover_core.attach_cluster_labels(scored, clustered)
    check("out_fixes rows == scored", len(out_fixes), len(scored), results)
    print(f"  cluster_id dtype             {out_fixes['cluster_id'].dtype}")
    print(f"  stopover_id nulls            {out_fixes['stopover_id'].isna().sum():>8}")

    print("\n" + "=" * 62)
    if all(results):
        print(f"ALL {len(results)} CHECKS MATCH")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
    print("=" * 62)


if __name__ == "__main__":
    main()
