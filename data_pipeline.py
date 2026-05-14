import sqlite3
import pandas as pd
import numpy as np
import lancedb
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Retrieve the base path from environment variables
# Fallback to the current directory if not set
BASE_PATH = os.getenv("SPOTIFY_DATA_BASE_PATH", os.getcwd())

# Define relative paths based on the base path
MAIN_DB_PATH = os.path.join(BASE_PATH, "spotify_clean.sqlite3")
FEATURES_DB_PATH = os.path.join(BASE_PATH, "spotify_clean_audio_features.sqlite3")
VECTOR_DB_PATH = os.path.join(BASE_PATH, "spotify_vector_vault")
BOUNDS_FILE = os.path.join(BASE_PATH, "spotify_bounds.json")

def calculate_global_bounds(cursor):
    """
    Queries both databases for the global min/max of each feature that needs
    scaling (popularity, tempo, key, loudness, time_signature) and writes
    the results to BOUNDS_FILE so the same scale is applied across all chunks.

    Returns the bounds dict so the caller can use it without re-reading the file.
    """
    print("Calculating global bounds for normalization...")
    cursor.execute("SELECT MIN(popularity), MAX(popularity) FROM tracks")
    pop_min, pop_max = cursor.fetchone()

    cursor.execute("""
        SELECT
            MIN(time_signature), MAX(time_signature),
            MIN(tempo), MAX(tempo),
            MIN(key), MAX(key),
            MIN(loudness), MAX(loudness)
        FROM features_db.track_audio_features
    """)
    bounds = cursor.fetchone()

    bounds_data = {
        "popularity": {"min": pop_min, "max": pop_max},
        "time_signature": {"min": bounds[0], "max": bounds[1]},
        "tempo": {"min": bounds[2], "max": bounds[3]},
        "key": {"min": bounds[4], "max": bounds[5]},
        "loudness": {"min": bounds[6], "max": bounds[7]}
    }

    with open(BOUNDS_FILE, 'w') as f:
        json.dump(bounds_data, f)

    return bounds_data

def migrate_to_vector_db():
    """
    Reads MAIN_DB_PATH and FEATURES_DB_PATH in 1M-row chunks, normalises the
    five unbounded audio features to [0, 1] using global bounds, assembles a
    12-dimensional vector per track, and writes the result to a LanceDB table
    at VECTOR_DB_PATH.

    Raises FileNotFoundError if either source database is missing.
    The SQLite connection is always closed, even if an error occurs mid-migration.
    """
    if not os.path.exists(MAIN_DB_PATH):
        raise FileNotFoundError(
            f"Main database not found: {MAIN_DB_PATH}\n"
            "Set SPOTIFY_DATA_BASE_PATH in your .env to the folder containing the Anna's Archive files."
        )
    if not os.path.exists(FEATURES_DB_PATH):
        raise FileNotFoundError(
            f"Features database not found: {FEATURES_DB_PATH}\n"
            "Set SPOTIFY_DATA_BASE_PATH in your .env to the folder containing the Anna's Archive files."
        )

    conn = sqlite3.connect(MAIN_DB_PATH)
    try:
        cursor = conn.cursor()

        # Performance optimizations for SQLite
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        cursor.execute(f"ATTACH DATABASE '{FEATURES_DB_PATH}' AS features_db")

        bounds = calculate_global_bounds(cursor)

        # Fixed estimate used only for progress-bar display; actual count may differ
        total_tracks = 45000000
        db = lancedb.connect(VECTOR_DB_PATH)

        cols_to_scale = ["time_signature", "tempo", "key", "popularity", "loudness"]
        cols_already_scaled = [
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "danceability", "energy"
        ]

        query = """
            SELECT
                t.id, t.name, t.popularity,
                a.name as artist_name,
                f.time_signature, f.tempo, f.key, f.loudness,
                f.speechiness, f.acousticness, f.instrumentalness,
                f.liveness, f.valence, f.danceability, f.energy
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists a ON ta.artist_rowid = a.rowid
            JOIN features_db.track_audio_features f ON t.id = f.track_id
            WHERE t.popularity > 0
        """

        # 1M rows per chunk balances memory usage against SQLite read overhead
        chunk_size = 1000000
        first_chunk = True

        print(f"Starting chunked migration of ~{total_tracks:,} tracks...")

        with tqdm(total=total_tracks, desc="Migrating", unit="track") as pbar:
            for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
                # Apply normalization based on the pre-calculated global bounds
                for col in cols_to_scale:
                    g_min = bounds[col]["min"]
                    g_max = bounds[col]["max"]
                    chunk[col] = (chunk[col] - g_min) / (g_max - g_min) if g_max > g_min else 0.0

                # Combine all features into a single high-dimensional vector
                all_feature_cols = cols_to_scale + cols_already_scaled
                chunk["vector"] = chunk[all_feature_cols].values.tolist()

                clean_chunk = chunk[["id", "name", "artist_name", "vector", "popularity"]]

                if first_chunk:
                    db.create_table("tracks", data=clean_chunk, mode="overwrite")
                    first_chunk = False
                else:
                    db.open_table("tracks").add(clean_chunk)

                pbar.update(len(chunk))

    except Exception:
        print(f"Migration failed. Partial data may exist at: {VECTOR_DB_PATH}")
        raise
    finally:
        conn.close()

    print(f"Success! Vector database created at: {VECTOR_DB_PATH}")

if __name__ == "__main__":
    migrate_to_vector_db()
