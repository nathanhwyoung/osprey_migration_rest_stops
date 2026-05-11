"""
Osprey Stopover Detection Script

Dataset: Martell & Douglas (1995-2002) - Movebank
Pipeline: NSD flat-segment detection + DBSCAN spatial clustering.
Requires: pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

DATA_FILE = "Osprey in North and South America 1995-2002 (Martell).csv"
REF_FILE = "Osprey in North and South America 1995-2002 (Martell)-reference-data.csv"
OUTPUT_FILE = "stopover_candidates_v2.csv"

# Fall migration window only — avoids treating winter residency as stopover behavior.
MIGRATION_START_MONTH = 8
MIGRATION_END_MONTH = 12

# Max NSD range (km) within the window for a bird to count as stationary.
# 100 km absorbs Argos error (10–30 km) and within-stopover movement while
# still excluding active migration (hundreds of km/week).
NSD_FLAT_THRESHOLD_KM = 100

WINDOW_DAYS = 7

# Lowest threshold used in comparable raptor telemetry studies; matches the
# biological expectation that meaningful refueling takes 2–3 days.
MIN_STOPOVER_DAYS = 3

# Fix-level filter: excludes a bird sitting near its nest pre-departure,
# which would otherwise read as "stationary" with near-zero NSD.
MIN_DEPARTURE_KM = 100

# Cluster-level upper bound. Longest documented en-route osprey stopover
# (Kjellén et al. 2001) was ~44 days; above this is wintering residency.
# Set to None to disable.
MAX_STOPOVER_DAYS = 45

# Cluster-level complement to MIN_DEPARTURE_KM: catches clusters whose
# individual fixes pass the fix-level filter but whose centroid is still
# close to home (pre-departure staging just past 100 km). 250 km chosen
# from a visible gap in cluster-mean-NSD distribution during test runs.
# Set to None to disable.
MIN_CLUSTER_NSD_KM = 250

# Generous to keep fixes from one physical stopover together despite
# Argos error and within-site movement; tradeoff is that nearby distinct
# stopovers may merge.
CLUSTER_RADIUS_KM = 75
MIN_CLUSTER_FIXES = 3

# Tightened from 100 km after inspection showed median pair distance of
# 42.8 km at that threshold — many "shared" sites weren't actually close.
SHARED_RADIUS_KM = 50

EARTH_RADIUS_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorized over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def detect_stopover_fixes(bird_df, deploy_lat, deploy_lon):
    """
    Flag fixes belonging to an NSD plateau. For each fix, look forward
    WINDOW_DAYS days; if NSD range stays under NSD_FLAT_THRESHOLD_KM
    across at least MIN_STOPOVER_DAYS, flag it.
    """
    df = bird_df.copy()
    df["nsd_km"] = haversine(
        deploy_lat, deploy_lon, df["location-lat"].values, df["location-long"].values
    )
    df["is_stopover_candidate"] = False

    dates = df["date"].values
    nsds = df["nsd_km"].values

    for i in range(len(df)):
        if nsds[i] < MIN_DEPARTURE_KM:
            continue

        t_start = dates[i]
        t_end = dates[i] + np.timedelta64(WINDOW_DAYS, "D")
        window_mask = (dates >= t_start) & (dates <= t_end)
        window_nsds = nsds[window_mask]
        window_dates = dates[window_mask]

        if len(window_dates) < 2:
            continue

        span_days = (window_dates.max() - window_dates.min()) / np.timedelta64(1, "D")
        nsd_range = window_nsds.max() - window_nsds.min()

        if span_days >= MIN_STOPOVER_DAYS and nsd_range <= NSD_FLAT_THRESHOLD_KM:
            df.iloc[i, df.columns.get_loc("is_stopover_candidate")] = True

    return df


def cluster_stopover_fixes(candidate_df):
    """DBSCAN grouping of candidate fixes. Returns df with cluster_id (-1 = noise)."""
    if len(candidate_df) == 0:
        candidate_df = candidate_df.copy()
        candidate_df["cluster_id"] = pd.Series(dtype=int)
        return candidate_df

    # sklearn's haversine metric requires radians, not degrees, and returns
    # angular distances — so eps must also be in radians.
    coords_rad = np.radians(candidate_df[["location-lat", "location-long"]].values)
    eps_rad = CLUSTER_RADIUS_KM / EARTH_RADIUS_KM

    # ball_tree is required for haversine; the default 'auto' won't accept it.
    db = DBSCAN(
        eps=eps_rad,
        min_samples=MIN_CLUSTER_FIXES,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)

    candidate_df = candidate_df.copy()
    candidate_df["cluster_id"] = db.labels_
    return candidate_df


def summarize_clusters(clustered_df, bird_id):
    """One row per cluster summarizing centroid, dates, fix count, and mean NSD."""
    records = []
    valid_clusters = clustered_df[clustered_df["cluster_id"] >= 0]

    for cid, grp in valid_clusters.groupby("cluster_id"):
        records.append(
            {
                "bird_id": bird_id,
                "stopover_id": f"{bird_id}_stop{cid}",
                "lat": round(grp["location-lat"].mean(), 4),
                "lon": round(grp["location-long"].mean(), 4),
                "date_first": grp["date"].min().date(),
                "date_last": grp["date"].max().date(),
                # +1 to count days inclusive of both endpoints (a single-fix
                # stop has duration 1, not 0).
                "duration_days": (grp["date"].max() - grp["date"].min()).days + 1,
                "n_fixes": len(grp),
                "mean_nsd_km": round(grp["nsd_km"].mean(), 1),
            }
        )

    return pd.DataFrame(records)


def find_shared_stopovers(stopovers_df):
    """
    For each stopover, find other birds' stopovers within SHARED_RADIUS_KM.
    Same-bird pairs are skipped — we want cross-individual site sharing,
    not within-individual revisits.
    """
    lats = stopovers_df["lat"].values
    lons = stopovers_df["lon"].values
    birds = stopovers_df["bird_id"].values
    n = len(stopovers_df)

    shared_with = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if birds[i] == birds[j]:
                continue
            dist = haversine(lats[i], lons[i], lats[j], lons[j])
            if dist <= SHARED_RADIUS_KM:
                shared_with[i].append(birds[j])
                shared_with[j].append(birds[i])

    df = stopovers_df.copy()
    df["shared_with"] = [";".join(sorted(set(s))) for s in shared_with]
    df["n_birds_total"] = [len(set(s)) + 1 if s else 1 for s in shared_with]
    return df


def main():
    print("=" * 60)
    print("OSPREY STOPOVER DETECTION v2")
    print("=" * 60)

    print("\n[1/9] Loading data...")
    tracks = pd.read_csv(DATA_FILE)
    ref = pd.read_csv(REF_FILE)

    # Force to str: pandas reads numeric-looking IDs as int, which silently
    # breaks the join in Step 3 if the other side is str.
    tracks["tag-local-identifier"] = tracks["tag-local-identifier"].astype(str)
    tracks["individual-local-identifier"] = tracks[
        "individual-local-identifier"
    ].astype(str)
    ref["tag-id"] = ref["tag-id"].astype(str)
    ref["animal-id"] = ref["animal-id"].astype(str)

    # 41 of 80 tags in this dataset were physically redeployed on a second
    # bird. Grouping by tag ID alone blends two animals' tracks into one.
    # The composite tag+individual key keeps each deployment separate.
    tracks["deployment_key"] = (
        tracks["tag-local-identifier"] + "_" + tracks["individual-local-identifier"]
    )
    print(f"  Tracking records loaded : {len(tracks):,}")
    print(f"  Deployment records loaded: {len(ref):,}")

    print("\n[2/9] Filtering...")

    # Movebank's pre-computed quality flag — drops Z-class fixes (no position,
    # only Doppler) and other records flagged as unreliable.
    n_before = len(tracks)
    tracks = tracks[tracks["visible"].astype(str).str.upper() == "TRUE"].copy()
    print(
        f"  After visible=TRUE filter : {len(tracks):,} (dropped {n_before - len(tracks):,})"
    )

    tracks = tracks.dropna(subset=["location-lat", "location-long"])
    print(f"  After dropping no-coord rows: {len(tracks):,}")

    tracks["date"] = pd.to_datetime(tracks["timestamp"])

    tracks = tracks[
        tracks["date"].dt.month.between(MIGRATION_START_MONTH, MIGRATION_END_MONTH)
    ]
    print(f"  After Aug-Dec filter       : {len(tracks):,} fixes")

    print("\n[3/9] Merging deployment locations...")

    ref_coords = ref[
        ["tag-id", "animal-id", "deploy-on-latitude", "deploy-on-longitude"]
    ].copy()

    # Join on both tag and animal: joining on tag alone returns the first
    # deployment's coordinates for any reused tag.
    tracks = tracks.merge(
        ref_coords,
        left_on=["tag-local-identifier", "individual-local-identifier"],
        right_on=["tag-id", "animal-id"],
        how="left",
    )

    n_before = len(tracks)
    tracks = tracks.dropna(subset=["deploy-on-latitude", "deploy-on-longitude"])
    if n_before > len(tracks):
        print(
            f"  Warning: dropped {n_before - len(tracks)} fixes with no deployment coords"
        )

    birds = tracks["deployment_key"].unique()
    print(f"  Birds with valid deployment coords: {len(birds)}")

    print("\n[4/9] Detecting stopover candidates (NSD flat-segment method)...")
    all_candidates = []

    for bird_id in birds:
        bird_df = tracks[tracks["deployment_key"] == bird_id].sort_values("date").copy()
        deploy_lat = bird_df["deploy-on-latitude"].iloc[0]
        deploy_lon = bird_df["deploy-on-longitude"].iloc[0]

        result = detect_stopover_fixes(bird_df, deploy_lat, deploy_lon)

        candidates = result[result["is_stopover_candidate"]].copy()
        candidates["bird_id"] = bird_id
        all_candidates.append(candidates)

    if not all_candidates:
        print("  No stopover candidates found. Try loosening thresholds.")
        return

    candidates_df = pd.concat(all_candidates, ignore_index=True)
    print(f"  Candidate fixes found: {len(candidates_df):,}")
    print(
        f"  Birds with at least one candidate fix: "
        f"{candidates_df['bird_id'].nunique()}"
    )

    print("\n[5/9] Clustering candidate fixes spatially (DBSCAN)...")
    all_summaries = []

    # Cluster per-bird, not across all birds — cross-individual sharing is
    # handled in Step 8.
    for bird_id, grp in candidates_df.groupby("bird_id"):
        clustered = cluster_stopover_fixes(grp)
        summary = summarize_clusters(clustered, bird_id)
        all_summaries.append(summary)

    stopovers_df = pd.concat(all_summaries, ignore_index=True)
    print(f"  Stopover clusters identified: {len(stopovers_df)}")

    if stopovers_df.empty:
        print("  No clusters passed the minimum-fixes threshold.")
        print("  Try lowering MIN_CLUSTER_FIXES or increasing CLUSTER_RADIUS_KM.")
        return

    print("\n[6/9] Filtering likely wintering sites (duration)...")
    if MAX_STOPOVER_DAYS is not None:
        n_before = len(stopovers_df)
        stopovers_df = stopovers_df[stopovers_df["duration_days"] <= MAX_STOPOVER_DAYS]
        n_dropped = n_before - len(stopovers_df)
        print(
            f"  Dropped {n_dropped} clusters > {MAX_STOPOVER_DAYS} days (likely wintering sites)"
        )
        print(f"  Remaining after duration filter: {len(stopovers_df)}")

    if stopovers_df.empty:
        print(
            "  No stopovers remain after duration filter. Try raising MAX_STOPOVER_DAYS."
        )
        return

    print("\n[7/9] Filtering pre-departure staging (NSD threshold)...")
    if MIN_CLUSTER_NSD_KM is not None:
        n_before = len(stopovers_df)
        stopovers_df = stopovers_df[stopovers_df["mean_nsd_km"] >= MIN_CLUSTER_NSD_KM]
        n_dropped = n_before - len(stopovers_df)
        print(f"  Dropped {n_dropped} clusters with mean NSD < {MIN_CLUSTER_NSD_KM} km")
        print(f"  Remaining after NSD filter: {len(stopovers_df)}")

    if stopovers_df.empty:
        print(
            "  No stopovers remain after NSD filter. Try lowering MIN_CLUSTER_NSD_KM."
        )
        return

    print("\n[8/9] Identifying shared stopover sites...")
    stopovers_df = find_shared_stopovers(stopovers_df)
    n_shared = (stopovers_df["n_birds_total"] > 1).sum()
    print(f"  Stopovers used by 2+ birds: {n_shared}")

    print("\n[9/9] Writing output...")
    stopovers_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Columns: {list(stopovers_df.columns)}")

    print("\n" + "=" * 60)
    print("STOPOVER SUMMARY (sorted by number of birds sharing site)")
    print("=" * 60)
    display_cols = [
        "bird_id",
        "lat",
        "lon",
        "date_first",
        "date_last",
        "duration_days",
        "n_fixes",
        "mean_nsd_km",
        "n_birds_total",
        "shared_with",
    ]
    print(
        stopovers_df[display_cols]
        .sort_values(["n_birds_total", "mean_nsd_km"], ascending=[False, True])
        .to_string(index=False)
    )

    print("\n" + "-" * 40)
    print(f"Total stopovers detected       : {len(stopovers_df)}")
    print(f"Birds represented              : {stopovers_df['bird_id'].nunique()}")
    print(
        f"Median duration (days)         : {stopovers_df['duration_days'].median():.1f}"
    )
    print(f"Median fixes per stopover      : {stopovers_df['n_fixes'].median():.1f}")
    print(f"Stopovers shared by 2+ birds   : {n_shared}")
    print(f"Max birds at a single site     : {stopovers_df['n_birds_total'].max()}")


if __name__ == "__main__":
    main()
