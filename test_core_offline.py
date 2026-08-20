"""Offline validation of stopover_core against BASELINE.txt. No arcpy."""
import numpy as np
import pandas as pd

import stopover_core as core

DATA_FILE = "Osprey in North and South America 1995-2002 (Martell).csv"
REF_FILE = "Osprey in North and South America 1995-2002 (Martell)-reference-data.csv"

def build_input():
    """Stand-in for _read_points on osprey_fixes with visible = 'true'."""
    tracks = pd.read_csv(DATA_FILE, low_memory=False)
    ref = pd.read_csv(REF_FILE)

    for c in ["tag-local-identifier", "individual-local-identifier"]:
        tracks[c] = tracks[c].astype(str)
    for c in ["tag-id", "animal-id"]:
        ref[c] = ref[c].astype(str)

    tracks["id"] = (
        tracks["tag-local-identifier"] + "_" + tracks["individual-local-identifier"]
    )
    tracks = tracks[tracks["visible"].astype(str).str.upper() == "TRUE"]
    tracks = tracks.dropna(subset=["location-lat", "location-long"])

    tracks = tracks.merge(
        ref[["tag-id", "animal-id", "deploy-on-latitude", "deploy-on-longitude"]],
        left_on=["tag-local-identifier", "individual-local-identifier"],
        right_on=["tag-id", "animal-id"],
        how="left",
    )

    df = pd.DataFrame({
        "oid": np.arange(1, len(tracks) + 1),
        "id": tracks["id"].astype(str),
        "timestamp": pd.to_datetime(tracks["timestamp"]),
        "lat": tracks["location-lat"],
        "lon": tracks["location-long"],
        "origin_lat": tracks["deploy-on-latitude"],
        "origin_lon": tracks["deploy-on-longitude"],
    })
    return df.sort_values(["id", "timestamp"]).reset_index(drop=True)

def main():
    df = build_input()
    candidates, scored, info = core.find_candidate_fixes(
        df,
        origin_mode="fields",
        origin_days=7,
        season_start_month=8,
        season_end_month=12,
        window_days=7,
        flat_threshold_km=100.0,
        min_stopover_days=3,
        min_departure_km=100.0,
    )

    for k, v in info.items():
        print("{:30s}: {}".format(k, v))

    clustered = core.cluster_candidates(candidates, cluster_radius_km=75, min_cluster_fixes=3)
    stopovers = core.summarize_clusters(clustered)
    print("{:30s}: {}".format("clusters summarized", len(stopovers)))

    filtered, finfo = core.filter_clusters(stopovers, max_stopover_days=45, min_cluster_displacement_km=250)
    for k, v in finfo.items():
        print("{:30s}: {}".format(k, v))

    ok = (
        info["candidate_fixes"] == 10145 and info["individuals_with_candidates"] == 89
    )
    print("\n" + ("MATCH" if ok else "*** MISMATCH ***"))

if __name__ == "__main__":
    main()