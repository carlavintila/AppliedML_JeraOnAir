import os
import csv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build

OUTPUT_FILE = "../datasets/artists_2022.csv"

def get_spotify_client():
    """Create and return an authenticated Spotify API client."""
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="", client_secret="",))

def get_youtube_client():
    """Create and return an authenticated YouTube Data API client."""
    return build("youtube", "v3", developerKey="")

def get_artist_data(sp, artist_id: str, fallback_name=None):
    """
    Fetch Spotify metadata for an artist.

    Args:
        sp: Authenticated Spotify API client.
        artist_id: Spotify artist ID.
        fallback_name: Name to use if the Spotify response does not include one.

    Returns:
        A dictionary containing artist name, popularity score, followers, and genres,
        or None if the artist data could not be fetched.
    """
    try:
        artist = sp.artist(artist_id)

        return {
            "name": artist.get("name") or fallback_name,
            "score": artist.get("popularity"),
            "followers": artist.get("followers", {}).get("total"),
            "genre": ", ".join(artist.get("genres", [])),}

    except Exception as e:
        print(f"Error fetching artist {artist_id}: {e}")
        return None


def get_youtube_channel_data(youtube, artist_name: str):
    """
    Fetch YouTube channel statistics for an artist.

    Args:
        youtube: Authenticated YouTube Data API client.
        artist_name: Artist name used to search for a YouTube channel.

    Returns:
        A dictionary containing YouTube subscriber and view counts. Empty strings are
        returned when no channel is found or when the data cannot be fetched.
    """
    try:
        search_response = youtube.search().list(q=artist_name, part="snippet", type="channel", maxResults=1).execute()

        items = search_response.get("items", [])
        if not items:
            return {"youtube_subscribers": "", "youtube_views": ""}

        channel_id = items[0]["snippet"]["channelId"]
        channel_response = youtube.channels().list(part="statistics", id=channel_id).execute()
        stats = channel_response["items"][0]["statistics"]

        return {"youtube_subscribers": stats.get("subscriberCount", ""), "youtube_views": stats.get("viewCount", "")}

    except Exception as e:
        print(f"YouTube error for {artist_name}: {e}")
        return {"youtube_subscribers": "", "youtube_views": ""}

def save_to_csv(rows):
    """
    Save artist data rows to the output CSV file.

    Args:
        rows: Iterable of dictionaries containing artist and YouTube data.
    """
    headers = ["name", "score", "followers", "genre", "youtube_subscribers", "youtube_views"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def get_artists(sp, playlist_url: str):
    """
    Extract unique artists from a Spotify playlist.

    Args:
        sp: Authenticated Spotify API client.
        playlist_url: Spotify playlist URL.

    Returns:
        A list of dictionaries containing unique Spotify artist IDs and names.
    """
    artists = []
    seen = set()

    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    response = sp.playlist_items(playlist_id)

    while response:
        for item in response.get("items", []):
            track = item.get("track")
            if not track:
                continue

            for artist in track.get("artists", []):
                artist_id = artist.get("id")

                if artist_id not in seen:
                    seen.add(artist_id)
                    artists.append({"id": artist_id, "name": artist.get("name")})

        if response.get("next"):
            response = sp.next(response)
        else:
            response = None

    return artists

if __name__ == "__main__":
    PLAYLIST_URL = "https://open.spotify.com/playlist/7hgVGY52fDPoJrvVPDUWSv?si=bc7fc74c89a24b0e"

    sp = get_spotify_client()
    youtube = get_youtube_client()

    artists = get_artists(sp, PLAYLIST_URL)
    artists = sorted(artists, key=lambda a: a["name"].lower())
    print(f"Found {len(artists)} unique artists")

    results = []

    for artist in artists:
        data = get_artist_data(sp, artist["id"], artist["name"])
        if not data:
            continue

        yt_data = get_youtube_channel_data(youtube, artist["name"])
        data.update(yt_data)

        results.append(data)

    save_to_csv(results)
    print(f"Saved {len(results)} artists to {OUTPUT_FILE}")