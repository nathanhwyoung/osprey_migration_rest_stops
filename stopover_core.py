# -*- coding: utf-8 -*-

"""
stopover_core.py

stopover detection math, no arcpy/geoprocessing/messages
receives a df with columns: id, timestamp, lat, lon; returns a df
intended to be runnable/testable outside of arcgis pro

all distances are kilometers, all durations are days
unit conversion happens in StopoverTools.pyt before it gets sent here
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def resolve_origins_from_fields(df):
    """
    origin per individual bird, this runs if a bird has deploy dates/coords
    """
    have = df.dropna(subset=["origin_lat", "origin_lon"])
    return have.groupby("id")[["origin_lat", "origin_lon"]].first()

def resolve_origins_from_first_fixes(df, origin_days):
    """
    origin per individual bird, this runs if a bird does NOT have deploy dates/coords
    """
    out = {}
    for ind_id, grp in df.groupby("id"):
        t0 = grp["timestamp"].min()
        early = grp[grp["timestamp"] <= t0 + pd.Timedelta(days=origin_days)]
        out[ind_id] = (early["lat"].median(), early["lon"].median())

    return pd.DataFrame.from_dict(
        out, orient="index", columns=["origin_lat", "origin_lon"]
    ).rename_axis("id")

def filter_season(df, start_month, end_month):
    """
    keep fixes inside calendar window
    """
    if start_month is None or end_month is None:
        return df

    months = df["timestamp"].dt.month
    if start_month <= end_month:
        keep = months.between(start_month, end_month)
    else:
        keep = (months >= start_month) | (months <= end_month)
    return df[keep]

def flag_flat_segments(dates, displacement_km, window_days, flat_threshold_km, min_stopover_days, min_departure_km):
    """
    flag fixes that indicate bird is resting
    for each fix, look forward 'window_days' days, if displacement varies by no more than 'flat_threshold_km' across that window,
    and the window spans 'min_stopover_days', the fix is a stopover candidate

    'dates' MUST be sorted ascending. np.searchsorted assumes that, does not check

    returns boolean array 
    """
    n = len(dates)

    displacement_km = np.asarray(displacement_km)

    if displacement_km.shape != (n,):
        raise ValueError(
            "displacement km must be a 1-D array parallel to dates: "
            "got shape {} for {} dates.".format(displacement_km.shape, n)
        )
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags

    window = np.timedelta64(int(window_days), "D")

    lo = np.searchsorted(dates, dates, side="left")
    hi = np.searchsorted(dates, dates + window, side="right")

    span_days = (dates[hi - 1] - dates[lo]) / np.timedelta64(1, "D")

    eligible = (
        (displacement_km >= min_departure_km)
        & ((hi - lo) >= 2)
        & (span_days >= min_stopover_days)
    )

    for i in np.flatnonzero(eligible):
        window_disp = displacement_km[lo[i]:hi[i]]
        if window_disp.max() - window_disp.min() <= flat_threshold_km:
            flags[i] = True

    return flags

def find_candidate_fixes(
    df, 
    origin_mode, 
    origin_days, 
    season_start_month, 
    season_end_month, 
    window_days, 
    flat_threshold_km, 
    min_stopover_days, 
    min_departure_km
):
    """
    resolve origins, filter to season, compute displacement from origin, flag plateaus
    returns (candidates_df, info) -->> info is a dict of counts for report
    """
    info = {"fixes_in": len(df), "individuals_in": df["id"].nunique()}

    # origins come from unfiltered track (before season filter)
    if origin_mode == "fields":
        origins = resolve_origins_from_fields(df)
    else:
        origins = resolve_origins_from_first_fixes(df, origin_days)

    missing = sorted(set(df["id"].unique()) - set(origins.index))
    info["individuals_without_origin"] = missing

    df = df[df["id"].isin(origins.index)]
    info["fixes_dropped_no_origin"] = info["fixes_in"] - len(df)

    df = filter_season(df, season_start_month, season_end_month)
    info["fixes_after_season"] = len(df)

    pieces = []
    for ind_id, grp in df.groupby("id", sort=False):
        grp = grp.sort_values("timestamp")
        o_lat = origins.at[ind_id, "origin_lat"]
        o_lon = origins.at[ind_id, "origin_lon"]

        disp = haversine(o_lat, o_lon, grp["lat"].values, grp["lon"].values)

        flags = flag_flat_segments(
            grp["timestamp"].values,
            disp,
            window_days,
            flat_threshold_km,
            min_stopover_days,
            min_departure_km
        )

        grp = grp.copy()
        grp["displacement_km"] = disp
        grp["is_candidate"] = flags
        pieces.append(grp)

    # guard clause in case pieces is empty, avoids panda ValueError
    if not pieces:
        empty = df.copy()
        empty["displacement_km"] = pd.Series(dtype=float)
        empty["is_candidate"] = pd.Series(dtype=bool)
        info["candidate_fixes"] = 0
        info["individuals_with_candidates"] = 0
        return empty, empty.copy(), info

    scored = pd.concat(pieces, ignore_index=True)
    candidates = scored[scored["is_candidate"]].copy()

    info["candidate_fixes"] = len(candidates)
    info["individuals_with_candidates"] = candidates["id"].nunique()
    return candidates, scored, info

def cluster_candidates(candidates, cluster_radius_km, min_cluster_fixes):
    """
    assign dbscan cluster labels to fixes, per individual

    returns the same rows + cluster_id and stopover_id
    noise (-1) is kept, not filtered
    """

    pieces = []
    for ind_id, grp in candidates.groupby("id", sort=False):

        grp = grp.copy()

        coords_rad = np.radians(grp[["lat", "lon"]].values)
        eps_rad = cluster_radius_km / EARTH_RADIUS_KM

        labels = DBSCAN(
            eps=eps_rad,
            min_samples=min_cluster_fixes,
            algorithm="ball_tree",
            metric="haversine",
        ).fit(coords_rad).labels_

        grp["cluster_id"] = labels
        grp["stopover_id"] = [f"{ind_id}_stop{cid}" if cid >= 0 else "" for cid in labels]

        pieces.append(grp)  

    # again, guard clause
    if not pieces:
        empty = candidates.copy()
        empty["cluster_id"] = pd.Series(dtype=int)
        empty["stopover_id"] = pd.Series(dtype=str)
        return empty

    return pd.concat(pieces, ignore_index=True)



