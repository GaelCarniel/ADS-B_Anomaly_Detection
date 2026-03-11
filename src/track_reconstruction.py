import pandas as pd
import numpy as np
from dataclasses import dataclass

# Constants
GAP_THRESHOLD_SECONDS = 120      # split track if gap exceeds 2 minutes
ALTITUDE_JUMP_THRESHOLD_M = 1000 # 1000m sudden jump most likely -> new aircraft
MIN_TRACK_POINTS = 10            # discard very short tracks
MAX_INTERP_SECONDS = 60          # maximum gap to interpolate over

@dataclass
class TrackSegment:
    icao24: str
    callsign: str
    source_date: str
    points: pd.DataFrame      # time, lat, lon, baroaltitude, velocity, heading
    gap_count: int

    @property
    def duration_seconds(self) -> float:
        return (self.points["time"].iloc[-1] - self.points["time"].iloc[0]).total_seconds()

    @property
    def point_count(self) -> int:
        return len(self.points)


def split_on_gaps(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a single aircraft dataframe wherever time gap exceeds threshold."""
    df = df.sort_values("time").reset_index(drop=True)
    time_diffs = df["time"].diff().dt.total_seconds()
    split_indices = time_diffs[time_diffs > GAP_THRESHOLD_SECONDS].index

    segments = []
    prev = 0
    for idx in split_indices:
        segments.append(df.iloc[prev:idx].copy())
        prev = idx
    segments.append(df.iloc[prev:].copy())
    return [s for s in segments if len(s) > 0]

def split_on_gaps(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.sort_values("time").reset_index(drop=True)
    
    time_diffs = df["time"].diff().dt.total_seconds()
    alt_diffs  = df["baroaltitude"].diff().abs()
    
    # Also detect when altitude goes from valid to NaN to valid
    alt_was_valid  = df["baroaltitude"].notna()
    alt_gap_start  = alt_was_valid & (~alt_was_valid.shift(1).fillna(True))
    
    # Check altitude jump AFTER any NaN gaps by forward-filling
    alt_filled = df["baroaltitude"].ffill()
    alt_diffs_filled = alt_filled.diff().abs()
    
    split_mask = (
        (time_diffs > GAP_THRESHOLD_SECONDS) |
        (alt_diffs > ALTITUDE_JUMP_THRESHOLD_M) |
        (alt_diffs_filled > ALTITUDE_JUMP_THRESHOLD_M)
    )
    
    split_indices = split_mask[split_mask].index
    
    segments = []
    prev = 0
    for idx in split_indices:
        segments.append(df.iloc[prev:idx].copy())
        prev = idx
    segments.append(df.iloc[prev:].copy())
    return [s for s in segments if len(s) > 0]

def interpolate_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 10s grid and interpolate small gaps."""
    df = df.set_index("time")
    df = df.select_dtypes(include="number")  
    df = df.resample("10s").mean()
    df = df.interpolate(method="linear", limit=MAX_INTERP_SECONDS // 10)
    return df.reset_index()

def smooth_altitude(points: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Light smoothing for quantization noise only."""
    points = points.copy()
    points["baroaltitude"] = points["baroaltitude"].rolling(window=window_size, center=True, min_periods=1).mean()
    return points

def reconstruct_tracks(states_df: pd.DataFrame) -> list[TrackSegment]:
    """
    Input:  raw state vectors dataframe (filtered to bbox)
    Output: list of clean TrackSegment objects
    """
    tracks = []
    
    for icao24, aircraft_df in states_df.groupby("icao24"):
        aircraft_df = (aircraft_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        )
        
        segments = split_on_gaps(aircraft_df)
        
        for seg in segments:
            if len(seg) < MIN_TRACK_POINTS:
                continue

            #Metadata
            callsign_mode = seg["callsign"].mode()
            callsign = callsign_mode.iloc[0] if len(callsign_mode) > 0 else ""
            source_date = seg["source_date"].iloc[0] if "source_date" in seg.columns else ""
            gap_count = len(split_on_gaps(seg)) - 1
            
            points = interpolate_segment(seg[["time", "lat", "lon", "baroaltitude", "velocity", "heading"]])
            points = smooth_altitude(points)
            points = points.dropna(subset=["lat", "lon"])
            
            if len(points) < MIN_TRACK_POINTS:
                continue
            
            tracks.append(TrackSegment(
                icao24=icao24,
                callsign=callsign,
                source_date=source_date,
                points=points,
                gap_count=gap_count))
    
    return tracks


# Quality report
def reconstruction_report(tracks: list[TrackSegment]) -> None:
    durations = [t.duration_seconds / 60 for t in tracks]
    point_counts = [t.point_count for t in tracks]
    gap_counts = [t.gap_count for t in tracks]
    
    print(f"Total tracks reconstructed: {len(tracks)}")
    print(f"\nDuration (minutes):")
    print(f"  mean={np.mean(durations):.1f}  "
          f"median={np.median(durations):.1f}  "
          f"max={np.max(durations):.1f}")
    print(f"\nPoint count per track:")
    print(f"  mean={np.mean(point_counts):.1f}  "
          f"median={np.median(point_counts):.1f}  "
          f"max={np.max(point_counts):.1f}")
    print(f"\nTracks with gaps: "
          f"{sum(1 for g in gap_counts if g > 0)} "
          f"({sum(1 for g in gap_counts if g > 0)/len(tracks)*100:.1f}%)")