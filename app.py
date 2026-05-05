import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lancedb
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv

# Initialize environment and local storage paths
load_dotenv()
BASE_PATH = os.getenv("SPOTIFY_DATA_BASE_PATH", os.getcwd())
VECTOR_DB_PATH = os.path.join(BASE_PATH, "spotify_vector_vault")

st.set_page_config(page_title="Signal to Song", layout="wide")

# --- DATABASE AND API CLIENTS ---

@st.cache_resource
def get_vector_table():
    """Establishes connection to the local 45M track vector vault."""
    db = lancedb.connect(VECTOR_DB_PATH)
    return db.open_table("tracks")

def get_spotify_client():
    """Initializes Spotify API client for user profile retrieval."""
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope="user-top-read playlist-read-private"
    )
    return spotipy.Spotify(auth_manager=auth_manager)

# --- TRACK DATA ACQUISITION ---

def fetch_top_tracks(sp, time_range="medium_term", limit=20):
    """Retrieves user listening history for the selected time horizon."""
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    return [
        {"id": t["id"], "name": t["name"], "artist": t["artists"][0]["name"]} 
        for t in results["items"]
    ]

def fetch_playlist_tracks(sp, playlist_url, limit=20):
    """Parses playlist URLs or IDs to retrieve seed track metadata."""
    if "playlist/" in playlist_url:
        playlist_id = playlist_url.split("playlist/")[1].split("?")[0]
    else:
        playlist_id = playlist_url
        
    results = sp.playlist_tracks(playlist_id, limit=limit)
    return [
        {"id": t["track"]["id"], "name": t["track"]["name"], "artist": t["track"]["artists"][0]["name"]} 
        for t in results["items"] if t["track"]
    ]

# --- RECOMMENDATION ENGINE ---

def get_recommendations(seed_ids, diversity_level, pop_threshold, limit=20):
    """
    Performs vector similarity search against local features.
    Bypasses the deprecated Spotify audio-features API by using a local LanceDB vault.
    """
    table = get_vector_table()
    formatted_ids = ", ".join([f"'{tid}'" for tid in seed_ids])
    
    # Retrieve local vectors for seed tracks
    seed_data = table.search().where(f"id IN ({formatted_ids})").to_pandas()
    
    if seed_data.empty:
        st.error("No seed tracks found in the local archive. Try different seeds.")
        return pd.DataFrame()
        
    # Build user vibe centroid
    seed_vectors = np.stack(seed_data["vector"].values)
    profile_vector = np.mean(seed_vectors, axis=0).astype(np.float32)
    
    # Discovery variance
    if diversity_level > 0:
        noise = np.random.normal(0, diversity_level, profile_vector.shape)
        profile_vector = profile_vector + noise
    
    # Search local feature vault
    where_clause = f"popularity > {pop_threshold / 100} AND id NOT IN ({formatted_ids})"
    results = (
        table.search(profile_vector)
        .where(where_clause)
        .limit(limit * 10) 
        .to_pandas()
    )

    if results.empty:
        return pd.DataFrame()

    # Two-stage deduplication logic
    id_grouped = (
        results.groupby("id")
        .agg({
            "name": "first",
            "artist_name": lambda x: " & ".join(sorted(x.unique())),
            "popularity": "first",
            "_distance": "first"
        })
        .reset_index()
    )

    final_df = (
        id_grouped.groupby(["name", "artist_name"])
        .agg({
            "id": "first",
            "popularity": "max", 
            "_distance": "min"
        })
        .reset_index()
        .sort_values("_distance")
        .head(limit)
    )

    return final_df

# --- STREAMLIT DASHBOARD UI ---

st.title("🎵 Signal to Song")
st.write("Navigating the post-API landscape with local-first discovery across 45 million tracks.")

with st.sidebar:
    st.header("1. Input Data")
    source_type = st.radio("Seed Source", ["User Top Tracks", "Spotify Playlist"])
    
    if source_type == "User Top Tracks":
        time_period = st.selectbox("Horizon", ["short_term", "medium_term", "long_term"], index=1)
    else:
        playlist_url = st.text_input("Playlist URL or ID")
        
    st.divider()
    st.header("2. Search Parameters")
    min_pop = st.slider("Min Popularity", 0, 100, 70)
    discovery = st.slider("Discovery / Randomness", 0.0, 0.5, 0.1)
    
    st.divider()
    generate_clicked = st.button("Generate Recommendations", type="primary")

# --- MAIN LOGIC EXECUTION ---

if generate_clicked:
    try:
        spotify = get_spotify_client()
        with st.spinner("Executing local inference on your musical DNA..."):
            if source_type == "User Top Tracks":
                seeds = fetch_top_tracks(spotify, time_range=time_period)
            else:
                if not playlist_url:
                    st.error("Please enter a playlist URL.")
                    st.stop()
                seeds = fetch_playlist_tracks(spotify, playlist_url)
                
            seed_ids = [s["id"] for s in seeds]
            
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("Seed Profile")
            for s in seeds[:10]:
                st.write(f"* {s['name']} - {s['artist']}")
            if len(seeds) > 10:
                st.caption("... and more")
            
        with col_right:
            st.subheader("Recommended Matches")
            recommendations = get_recommendations(seed_ids, discovery, min_pop)
            
            if not recommendations.empty:
                output_df = recommendations[["name", "artist_name", "popularity"]].copy()
                output_df["popularity"] = (output_df["popularity"] * 100).astype(int)
                output_df.columns = ["Track Name", "Artist", "Popularity"]
                st.dataframe(output_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No matches found within selected constraints.")
                
    except Exception as error:
        st.error(f"Search failed: {error}")
else:
    st.info("Tune your parameters in the sidebar to search the local 45M track vault.")