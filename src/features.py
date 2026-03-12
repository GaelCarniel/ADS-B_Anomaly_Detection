import numpy as np
import pandas as pd
from dataclasses import dataclass

LFBO_LAT, LFBO_LON = 43.6293, 1.3673

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_features(track, cluster_id: int) -> dict:
    """
    Compute all features for a single TrackSegment.
    Returns a flat dictionary of feature_name -> value.
    """
    pts = track.points.dropna(subset=["lat", "lon"])
    if len(pts) < 3:
        return None
    
    features = {"icao24": track.icao24, "callsign": track.callsign}
    
    # Geometric ──────────────────────────────────────────
    distances = [
        haversine(pts.iloc[i]["lat"], pts.iloc[i]["lon"], pts.iloc[i+1]["lat"], pts.iloc[i+1]["lon"])
        for i in range(len(pts)-1)
    ]
    features["track_length_km"] = sum(distances)
    
    # Straight-line distance start→end
    features["straight_line_km"] = haversine(pts.iloc[0]["lat"],  pts.iloc[0]["lon"], pts.iloc[-1]["lat"], pts.iloc[-1]["lon"])
    
    # Sinuosity 1.0 = perfectly straight, higher = more curved
    features["sinuosity"] = (features["track_length_km"] / features["straight_line_km"] if features["straight_line_km"] > 0.1 else np.nan)
    
    # Closest approach to LFBO in km
    distances_to_lfbo = [haversine(row["lat"], row["lon"], LFBO_LAT, LFBO_LON)
        for _, row in pts.iterrows()
    ]
    features["min_dist_to_lfbo_km"] = min(distances_to_lfbo)
    
    # Lateral deviation from straight-line path
    start = pts.iloc[0][["lat", "lon"]].values
    end   = pts.iloc[-1][["lat", "lon"]].values
    mid   = pts.iloc[len(pts)//2][["lat", "lon"]].values
    
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len > 0:
        point_vec = mid - start
        cross = abs(np.cross(line_vec, point_vec)) / line_len
        features["lateral_deviation_deg"] = cross
    else:
        features["lateral_deviation_deg"] = np.nan
    
    # Altitude ───────────────────────────────────────────
    
    alt = pts["baroaltitude"].dropna()
    
    if len(alt) > 0:
        features["alt_mean_m"]   = alt.mean()
        features["alt_max_m"]    = alt.max()
        features["alt_min_m"]    = alt.min()
        features["alt_range_m"]  = alt.max() - alt.min()
        features["alt_std_m"]    = alt.std()
        
        # Net altitude change 
        features["alt_net_change_m"] = (alt.iloc[-1] - alt.iloc[0])
        
        # Altitude profile slope 
        x = np.arange(len(alt))
        slope = np.polyfit(x, alt.values, 1)[0] #linear
        features["alt_slope"] = slope  
        
    else:
        for f in ["alt_mean_m", "alt_max_m", "alt_min_m", "alt_range_m", "alt_std_m", "alt_net_change_m", "alt_slope"]:
            features[f] = np.nan
    
    # Kinematic ──────────────────────────────────────────
    
    if "velocity" in pts.columns:
        vel = pts["velocity"].dropna()
        if len(vel) > 0:
            features["speed_mean_ms"]  = vel.mean()
            features["speed_std_ms"]   = vel.std()
            features["speed_min_ms"]   = vel.min()
            features["speed_max_ms"]   = vel.max()
            # Speed variance ratio 
            features["speed_cv"] = (vel.std() / vel.mean() if vel.mean() > 0 else np.nan)
        else:
            for f in ["speed_mean_ms", "speed_std_ms", "speed_min_ms", "speed_max_ms", "speed_cv"]:
                features[f] = np.nan
    
    if "heading" in pts.columns:
        hdg = pts["heading"].dropna()
        if len(hdg) > 1:
            hdg_diff = hdg.diff().abs()
            hdg_diff = hdg_diff.apply(lambda x: x if x <= 180 else 360 - x)
            features["heading_change_rate"] = hdg_diff.mean()
        else:
            features["heading_change_rate"] = np.nan
    
    features["duration_min"] = track.duration_seconds / 60
    features["point_count"] = track.point_count
    
    # Contextual ─────────────────────────────────────────
    
    features["cluster_id"]    = cluster_id
    features["is_noise"]      = int(cluster_id == -1)
    features["source_date"]   = track.source_date
    #Hand-made be careful
    features["is_arriving"]   = int(track.is_arriving)
    features["is_departing"]  = int(track.is_departing)
    features["is_transiting"] = int(track.is_transiting)
        
    return features