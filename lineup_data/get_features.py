import pandas as pd
from collections import Counter
import numpy as np
import re
from difflib import get_close_matches

df = pd.read_csv("../datasets/artists_2022.csv")

def normalize(text):
    """Normalize genre text for consistent matching."""
    text = str(text).lower().strip()
    text = text.replace("r&b", "rnb")
    text = text.replace("r & b", "rnb")
    text = text.replace("rnb", "rnb")
    text = text.replace("rhythm and blues", "rnb")
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ORDER MATTERS
MACRO_RULES = [
    # SPECIFIC RULES FIRST 
    ("metalcore", "metal"),
    ("mathcore", "metal"),
    ("deathcore", "metal"),

    ("metallic hardcore", "punk_hardcore"),
    ("post hardcore", "punk_hardcore"),
    ("hardcore", "punk_hardcore"),

    # METAL 
    ("alternative metal", "metal"),
    ("black metal", "metal"),
    ("death metal", "metal"),
    ("thrash metal", "metal"),
    ("doom metal", "metal"),
    ("sludge metal", "metal"),
    ("progressive metal", "metal"),
    ("nu metal", "metal"),
    ("industrial metal", "industrial"),  # override
    ("metal", "metal"),
    ("djent", "metal"),
    ("grindcore", "metal"),
    ("drone metal", "metal"),
    ("heavy metal", "metal"),

    # PUNK / HARDCORE 
    ("punk", "punk_hardcore"),
    ("emo", "punk_hardcore"),
    ("screamo", "punk_hardcore"),
    ("riot grrrl", "punk_hardcore"),
    ("queercore", "punk_hardcore"),

    # ELECTRONIC 
    ("hardcore techno", "electronic"),
    ("gabber", "electronic"),
    ("hardstyle", "electronic"),
    ("frenchcore", "electronic"),
    ("happy hardcore", "electronic"),
    ("techno", "electronic"),
    ("house", "electronic"),
    ("trance", "electronic"),
    ("drum and bass", "electronic"),
    ("dnb", "electronic"),
    ("dubstep", "electronic"),
    ("riddim", "electronic"),
    ("electronic", "electronic"),
    ("big beat", "electronic"),
    ("darkwave", "electronic"),

    # HIP HOP 
    ("hip hop", "hip_hop"),
    ("rap", "hip_hop"),
    ("trap", "hip_hop"),
    ("drill", "hip_hop"),
    ("rnb", "hip_hop"),

    # INDUSTRIAL 
    ("industrial rock", "industrial"),
    ("industrial", "industrial"),
    ("ebm", "industrial"),

    # ROCK / ALT
    ("post rock", "rock_alt"),
    ("noise rock", "rock_alt"),
    ("garage rock", "rock_alt"),
    ("shoegaze", "rock_alt"),
    ("indie rock", "rock_alt"),
    ("alternative rock", "rock_alt"),
    ("rock", "rock_alt"),
    ("grunge", "rock_alt"),
    ("rock 'n roll", "rock_alt"),

    # POP 
    ("synthpop", "electronic"),
    ("electropop", "electronic"),
    ("indie pop", "pop"),
    ("bedroom pop", "pop"),
    ("pop", "pop"),
    ("nederpop", "pop"),

    # FOLK 
    ("folk punk", "punk_hardcore"),
    ("anti folk", "folk"),
    ("indie folk", "folk"),
    ("folk rock", "rock_alt"),
    ("americana", "folk"),
    ("bluegrass", "folk"),
    ("folk", "folk"),

    # REGGAE / SKA 
    ("ska punk", "punk_hardcore"),
    ("ska", "punk_hardcore"),
    ("reggae", "reggae"),

    # GENERIC (LOW PRIORITY!)
    ("wave", "electronic"),
    ("core", "punk_hardcore"),  
    ("organ music", "classical"),
]

KEYWORDS = [k for k, _ in MACRO_RULES]

def fuzzy_match(token):
    """
    Find the closest genre keyword match for a token.

    Args:
        token: Genre token to compare against known keywords.

    Returns:
        The closest matching keyword if one meets the cutoff threshold, otherwise None.
    """
    matches = get_close_matches(token, KEYWORDS, n=1, cutoff=0.8)
    return matches[0] if matches else None

def map_to_macro(genre):
    """
    Map a raw genre string to a macro genre category.

    Args:
        genre: Raw genre value to classify.

    Returns:
        A macro genre label, or "other" if no matching rule is found.
    """
    if not genre or str(genre) == "nan":
        return "other"

    genre = normalize(genre)

    # 1. direct substring matching (priority order)
    for key, macro in MACRO_RULES:
        if key in genre:
            return macro

    # 2. fuzzy fallback 
    tokens = genre.split()
    for token in tokens:
        match = fuzzy_match(token)
        if match:
            for key, macro in MACRO_RULES:
                if key == match:
                    return macro

    return "other"

def compute_headliner_stats(df, col):
    """
    Compute count and average popularity metrics for a headliner indicator column.

    Args:
        df: DataFrame containing artist data.
        col: Name of the binary headliner indicator column.

    Returns:
        A dictionary containing the number of matching artists and their average
        popularity metrics.
    """
    subset = df[df[col] == 1]
    return {f"num_{col}": subset.shape[0], f"{col}_popularity (0-100)": subset["popularity_score"].mean(), f"{col}_popularity_spotify (0-100)": subset["score"].mean(),}

if __name__ == "__main__":
    df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in str(x).split(",")])
    df["macro_genres"] = df["genre_list"].apply(lambda genres: list(set(map(map_to_macro, genres))))

    ALL_MACRO_GENRES = ["punk_hardcore", "metal", "industrial", "electronic", "hip_hop", "rock_alt", "pop", "folk", "reggae", "classical", "other",]

    avg_popularity = df["popularity_score"].mean()
    
    df["headliner_combined"] = ((df["headliner"] == 1) | (df["headliner_2"] == 1)).astype(int)

    combined_stats = compute_headliner_stats(df, "headliner_combined")
    headliner_stats = compute_headliner_stats(df, "headliner")
    sub_headliner_stats = compute_headliner_stats(df, "headliner_2")
    avg_popularity_sp = df["score"].mean()

    genre_weights = Counter()

    for i, genres in enumerate(df["macro_genres"]):
        if len(genres) == 0:
            continue

        artist_weight = df["popularity_score"].iloc[i]
        per_genre_weight = artist_weight / len(genres)

        for g in genres:
            genre_weights[g] += per_genre_weight

    # reduce "other" noise
    if "other" in genre_weights:
        genre_weights["other"] *= 0.5

    total_weight = sum(genre_weights.values())

    probs = np.array([genre_weights[g] / total_weight if total_weight > 0 else 0 for g in ALL_MACRO_GENRES])

    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    max_entropy = np.log2(len(ALL_MACRO_GENRES))
    normalized_entropy = entropy / max_entropy

    effective_genres = 2 ** entropy
    normalized_effective_genres = effective_genres / len(ALL_MACRO_GENRES)

    dominant_genre_share = probs.max()
    dominant_genre = ALL_MACRO_GENRES[np.argmax(probs)]

    features = {
        "avg_popularity (0-100)": avg_popularity,
        "avg_popularity_spotify (0-100)": avg_popularity_sp,
        **headliner_stats,
        **sub_headliner_stats,
        **combined_stats,
        "genre_entropy (0-1)": normalized_entropy,
        "num_genres": effective_genres,
        "dominant_genre_ratio (0-1)": dominant_genre_share,
        "dominant_genre": dominant_genre,}

    features_df = pd.DataFrame([features])
    features_df.to_csv("features_2022.csv", index=False)

    print(features_df)