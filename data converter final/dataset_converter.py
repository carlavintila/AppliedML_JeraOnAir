from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


BASE_DIR = Path(__file__).resolve().parent
FINAL_SCHEMA_PATH = BASE_DIR / "Model prediction (final)" / "final.csv"
LINEUP_PATH = BASE_DIR / "Lineup features + code" / "data" / "lineup_features.csv"

TICKET_TYPES = ["Early Bird", "Single Day", "Full Weekend", "Ambassador", "Camping", "Other"]
MODEL_FEATURES = [
    "days_since_sales_open",
    "sales_lag_1",
    "sales_roll_3_prior",
    "sales_roll_7_prior",
    "sales_roll_14_prior",
    "is_announcement",
    "sales_lag_14",
    "Full Weekend_roll_3d",
    "marketing_total_engagement_roll_7_prior",
    "Full Weekend_lag_1d",
    "fb_total_engagement_sum_roll_7_prior",
    "fb_total_reach_sum_roll_7_prior",
    "days_until_ambassador_expires",
]


@dataclass
class ConversionSummary:
    rows: int
    start_date: str
    end_date: str
    total_tickets: int
    marketing_days: int
    missing_model_features: list[str]
    warnings: list[str]


def load_final_schema() -> list[str]:
    return pd.read_csv(FINAL_SCHEMA_PATH, nrows=0).columns.tolist()


def read_table(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    name = filename or str(source)
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".csv":
        if hasattr(source, "read"):
            raw = source.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            source = io.BytesIO(raw)
        return pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    raise ValueError(f"Unsupported file type for {name}. Use CSV or Excel.")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _to_number(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None).dt.normalize()
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed.dt.normalize()


def _classify_ticket_type(name: object) -> str:
    text = str(name).lower()
    if "ambassador" in text:
        return "Ambassador"
    if "camping" in text:
        return "Camping"
    if "early bird" in text or "earlybird" in text:
        return "Early Bird"
    if any(word in text for word in ["friday", "saturday", "sunday", "thursday", "single day"]):
        return "Single Day"
    if any(word in text for word in ["3-day", "3 day", "combi", "weekend", "pass"]):
        return "Full Weekend"
    return "Other"


def standardize_tickets(
    ticket_df: pd.DataFrame,
    festival_year: int,
    festival_date: str | pd.Timestamp,
    include_days_until_event: bool = True,
) -> pd.DataFrame:
    df = _clean_columns(ticket_df)
    festival_date = pd.Timestamp(festival_date).normalize()
    warnings: list[str] = []

    if {"sale_date", "tickets_sold"}.issubset({c.lower() for c in df.columns}):
        sale_col = _first_existing(df, ["sale_date"])
        sold_col = _first_existing(df, ["tickets_sold"])
        year_col = _first_existing(df, ["festival_year"])
        days_col = _first_existing(df, ["days_to_event"])
        event_col = _first_existing(df, ["is_event_day"])
        out = pd.DataFrame()
        out["sale_date"] = _parse_dates(df[sale_col])
        out["festival_year"] = pd.to_numeric(df[year_col], errors="coerce").fillna(festival_year).astype(int) if year_col else festival_year
        out["tickets_sold"] = _to_number(df[sold_col])
        out["days_to_event"] = pd.to_numeric(df[days_col], errors="coerce") if days_col else (festival_date - out["sale_date"]).dt.days
        out["is_event_day"] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int) if event_col else (out["sale_date"] == festival_date).astype(int)
        for ticket_type in TICKET_TYPES:
            col = _first_existing(df, [ticket_type])
            out[ticket_type] = _to_number(df[col]) if col else 0.0
        out["# created"] = out["sale_date"]
    else:
        created_col = _first_existing(df, ["# created", "created", "created_at", "date"])
        count_col = _first_existing(df, ["count(1)", "count", "quantity", "tickets_sold"])
        name_col = _first_existing(df, ["name", "product_name", "ticket_type"])
        if not created_col or not count_col:
            raise ValueError("Ticket file needs '# created' and 'count(1)' columns, or standardized 'sale_date' and 'tickets_sold'.")

        work = pd.DataFrame()
        work["sale_date"] = _parse_dates(df[created_col])
        work["tickets_sold"] = _to_number(df[count_col])
        work["ticket_type"] = df[name_col].map(_classify_ticket_type) if name_col else "Other"
        work = work.dropna(subset=["sale_date"])

        daily = work.groupby("sale_date", as_index=False)["tickets_sold"].sum()
        type_daily = (
            work.pivot_table(index="sale_date", columns="ticket_type", values="tickets_sold", aggfunc="sum", fill_value=0)
            .reset_index()
        )
        out = daily.merge(type_daily, on="sale_date", how="left")
        out["festival_year"] = int(festival_year)
        out["days_to_event"] = (festival_date - out["sale_date"]).dt.days
        out["is_event_day"] = (out["sale_date"] == festival_date).astype(int)
        out["# created"] = out["sale_date"]

    out = out.dropna(subset=["sale_date"]).copy()
    out = out[out["sale_date"] <= festival_date].copy()
    if include_days_until_event and not out.empty:
        full_dates = pd.date_range(out["sale_date"].min(), festival_date, freq="D")
        out = out.set_index("sale_date").reindex(full_dates).rename_axis("sale_date").reset_index()
    out["festival_year"] = int(festival_year)
    out["days_to_event"] = (festival_date - out["sale_date"]).dt.days
    out["is_event_day"] = (out["sale_date"] == festival_date).astype(int)
    out["tickets_sold"] = pd.to_numeric(out.get("tickets_sold", 0), errors="coerce").fillna(0)
    out["# created"] = out["sale_date"]
    for ticket_type in TICKET_TYPES:
        if ticket_type not in out:
            out[ticket_type] = 0
        out[ticket_type] = pd.to_numeric(out[ticket_type], errors="coerce").fillna(0)
    return out.sort_values("sale_date").reset_index(drop=True)


def _post_type_counts(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    type_col = _first_existing(df, ["Berichttype", "post_type", "type"])
    if not type_col:
        return pd.DataFrame({"sale_date": df[date_col].dropna().unique()})
    work = df[[date_col, type_col]].dropna(subset=[date_col]).copy()
    labels = work[type_col].astype(str).str.lower()
    mapping = {
        "reel": labels.str.contains("reel", na=False),
        "image": labels.str.contains("foto|photo|image|afbeeld", na=False),
        "carousel": labels.str.contains("carousel|carrousel", na=False),
        "video": labels.str.contains("video", na=False) & ~labels.str.contains("reel", na=False),
    }
    if prefix == "fb":
        mapping["link"] = labels.str.contains("link", na=False)
        mapping["text"] = labels.str.contains("status|text", na=False)
    result = pd.DataFrame({"sale_date": sorted(work[date_col].dropna().unique())})
    for label, mask in mapping.items():
        col = f"{prefix}_{label}_posts_count"
        tmp = work.loc[mask].groupby(date_col).size().rename(col).reset_index().rename(columns={date_col: "sale_date"})
        result = result.merge(tmp, on="sale_date", how="left")
    return result.fillna(0)


def standardize_instagram(marketing_df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(marketing_df)
    if "sale_date" in df.columns and any(c.startswith("ig_") for c in df.columns):
        out = df.copy()
        out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce").dt.normalize()
        return out

    date_col = _first_existing(df, ["Publicatietijdstip", "date", "Datum"])
    if not date_col:
        raise ValueError("Instagram file needs 'Publicatietijdstip' or 'sale_date'.")

    df["_date"] = _parse_dates(df[date_col])
    df = df.dropna(subset=["_date"])
    specs = {
        "ig_views_sum": ["Weergaven", "views"],
        "ig_reach_sum": ["Bereik", "reach"],
        "ig_likes_sum": ["Vind-ik-leuks", "likes"],
        "ig_shares_sum": ["Aantal keer gedeeld", "shares"],
        "ig_comments_sum": ["Opmerkingen", "comments"],
        "ig_saves_sum": ["Aantal keer opgeslagen", "saves"],
        "ig_follows_sum": ["Volgt", "follows"],
    }
    out = df.groupby("_date").size().rename("ig_posts_count").reset_index().rename(columns={"_date": "sale_date"})
    for target, candidates in specs.items():
        col = _first_existing(df, candidates)
        values = _to_number(df[col]) if col else 0
        tmp = pd.DataFrame({"sale_date": df["_date"], target: values}).groupby("sale_date", as_index=False)[target].sum()
        out = out.merge(tmp, on="sale_date", how="left")
    counts = _post_type_counts(df, "_date", "ig")
    out = out.merge(counts, on="sale_date", how="left")
    for col in ["ig_reel_posts_count", "ig_image_posts_count", "ig_carousel_posts_count", "ig_video_posts_count", "ig_other_posts_count"]:
        if col not in out:
            out[col] = 0
    known = out[["ig_reel_posts_count", "ig_image_posts_count", "ig_carousel_posts_count", "ig_video_posts_count"]].sum(axis=1)
    out["ig_other_posts_count"] = np.maximum(out["ig_posts_count"] - known, out.get("ig_other_posts_count", 0))
    out["ig_has_post"] = (out["ig_posts_count"] > 0).astype(int)
    for metric in ["views", "reach", "likes", "comments", "shares", "saves"]:
        sum_col = f"ig_{metric}_sum"
        avg_col = f"ig_avg_{metric}_per_post"
        out[avg_col] = np.where(out["ig_posts_count"] > 0, out.get(sum_col, 0) / out["ig_posts_count"], 0)
    return out


def standardize_facebook(marketing_df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(marketing_df)
    date_col_existing = _first_existing(df, ["sale_date", "date"])
    if date_col_existing and any(c.startswith("fb_") for c in df.columns):
        out = df.copy().rename(columns={date_col_existing: "sale_date"})
        out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce").dt.normalize()
        return out

    date_col = _first_existing(df, ["Publicatietijdstip", "date", "Datum"])
    if not date_col:
        raise ValueError("Facebook file needs 'Publicatietijdstip' or 'date'.")
    df["_date"] = _parse_dates(df[date_col])
    df = df.dropna(subset=["_date"])
    specs = {
        "fb_total_engagement_sum": ["Reacties, opmerkingen en deelacties"],
        "fb_reactions_sum": ["Reacties"],
        "fb_comments_sum": ["Opmerkingen"],
        "fb_shares_sum": ["Deelacties"],
        "fb_total_clicks_sum": ["Totaal aantal klikken"],
        "fb_link_clicks_sum": ["Klikken op link"],
        "fb_other_clicks_sum": ["Overige klikken"],
        "fb_video_clicks_sum": ["Consumptie overeenkomend met doelgroeptargeting (Video Click)"],
        "fb_photo_clicks_sum": ["Consumptie overeenkomend met doelgroeptargeting (Photo Click)"],
        "fb_organic_reach_sum": ["Bereik via Organische berichten"],
        "fb_promoted_reach_sum": ["Bereik via Gepromote berichten"],
        "fb_total_reach_sum": ["Bereik"],
        "fb_video_views_3s_sum": ["Videoweergaven van 3 seconden"],
        "fb_video_views_1min_sum": ["Videoweergaven van 1 minuut"],
        "fb_organic_video_views_3s_sum": ["Videoweergaven van 3 seconden van Organische berichten"],
        "fb_promoted_video_views_3s_sum": ["Videoweergaven van 3 seconden van Gepromote berichten"],
        "fb_seconds_watched_sum": ["Seconden bekeken"],
        "fb_duration_sec_sum": ["Duur (sec.)"],
    }
    out = df.groupby("_date").size().rename("fb_posts_count").reset_index().rename(columns={"_date": "sale_date"})
    for target, candidates in specs.items():
        col = _first_existing(df, candidates)
        values = _to_number(df[col]) if col else 0
        tmp = pd.DataFrame({"sale_date": df["_date"], target: values}).groupby("sale_date", as_index=False)[target].sum()
        out = out.merge(tmp, on="sale_date", how="left")
    out["fb_duration_sec_mean"] = np.where(out["fb_posts_count"] > 0, out["fb_duration_sec_sum"] / out["fb_posts_count"], 0)
    out["fb_has_post"] = (out["fb_posts_count"] > 0).astype(int)
    shared_col = _first_existing(df, ["Is een gedeeld bericht"])
    promoted_col = _first_existing(df, ["Bereik via Gepromote berichten"])
    out["fb_shared_posts_count"] = 0
    out["fb_promoted_reach_posts_count"] = 0
    if shared_col:
        tmp = pd.DataFrame({"sale_date": df["_date"], "v": _to_number(df[shared_col])}).groupby("sale_date", as_index=False)["v"].sum()
        out = out.drop(columns=["fb_shared_posts_count"]).merge(tmp.rename(columns={"v": "fb_shared_posts_count"}), on="sale_date", how="left")
    if promoted_col:
        tmp = pd.DataFrame({"sale_date": df["_date"], "v": (_to_number(df[promoted_col]) > 0).astype(int)}).groupby("sale_date", as_index=False)["v"].sum()
        out = out.drop(columns=["fb_promoted_reach_posts_count"]).merge(tmp.rename(columns={"v": "fb_promoted_reach_posts_count"}), on="sale_date", how="left")
    counts = _post_type_counts(df, "_date", "fb")
    out = out.merge(counts, on="sale_date", how="left")
    for col in ["fb_link_posts_count", "fb_other_posts_count", "fb_photo_posts_count", "fb_reel_posts_count", "fb_text_posts_count", "fb_video_posts_count"]:
        if col not in out:
            out[col] = 0
    for metric, source in {
        "engagement": "fb_total_engagement_sum",
        "reactions": "fb_reactions_sum",
        "comments": "fb_comments_sum",
        "shares": "fb_shares_sum",
        "clicks": "fb_total_clicks_sum",
        "reach": "fb_total_reach_sum",
        "seconds_watched": "fb_seconds_watched_sum",
    }.items():
        out[f"fb_avg_{metric}_per_post"] = np.where(out["fb_posts_count"] > 0, out.get(source, 0) / out["fb_posts_count"], 0)
    out["fb_crossposted_posts_count"] = 0
    out["fb_crossposted_data_available_posts_count"] = 0
    out["fb_crossposted_data_available"] = 0
    out["fb_avg_seconds_viewed_mean"] = out.get("fb_avg_seconds_watched_per_post", 0)
    out["fb_source_files_count"] = 1
    out["fb_total_views_data_available_posts"] = out["fb_posts_count"]
    out["fb_views_data_available"] = (out["fb_posts_count"] > 0).astype(int)
    return out


def standardize_marketing_files(marketing_files: Iterable[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list[str]]:
    ig_frames: list[pd.DataFrame] = []
    fb_frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for filename, df in marketing_files:
        cols = {str(c).lower() for c in df.columns}
        try:
            if any(c.startswith("ig_") for c in df.columns) or "vind-ik-leuks" in cols or "account-id" in cols:
                ig_frames.append(standardize_instagram(df))
            elif any(c.startswith("fb_") for c in df.columns) or "pagina-id" in cols or "reacties" in cols:
                fb_frames.append(standardize_facebook(df))
            else:
                warnings.append(f"Skipped {filename}: could not identify it as Instagram or Facebook data.")
        except Exception as exc:
            warnings.append(f"Skipped {filename}: {exc}")

    merged = pd.DataFrame({"sale_date": pd.Series(dtype="datetime64[ns]")})
    if ig_frames:
        ig = pd.concat(ig_frames, ignore_index=True)
        ig = ig.groupby("sale_date", as_index=False).sum(numeric_only=True)
        merged = merged.merge(ig, on="sale_date", how="outer")
    if fb_frames:
        fb = pd.concat(fb_frames, ignore_index=True)
        fb = fb.groupby("sale_date", as_index=False).sum(numeric_only=True)
        merged = merged.merge(fb, on="sale_date", how="outer")
    return merged, warnings


def add_feature_engineering(
    tickets: pd.DataFrame,
    marketing: pd.DataFrame,
    festival_year: int,
    ambassador_expires: str | None = None,
) -> pd.DataFrame:
    df = tickets.merge(marketing, on="sale_date", how="left") if not marketing.empty else tickets.copy()
    df = df.sort_values(["festival_year", "sale_date"]).reset_index(drop=True)

    df["month"] = df["sale_date"].dt.month
    df["weekday"] = df["sale_date"].dt.weekday
    df["weekofyear"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    sales = df.groupby("festival_year")["tickets_sold"]
    df["sales_lag_1"] = sales.shift(1)
    df["sales_lag_7"] = sales.shift(7)
    df["sales_lag_14"] = sales.shift(14)
    for window in [3, 7, 14, 30]:
        df[f"sales_roll_{window}_prior"] = sales.transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    df["cumulative_sales_prior"] = sales.transform(lambda s: s.shift(1).cumsum())

    sales_open = df.groupby("festival_year")["sale_date"].transform("min")
    df["sales_open_date"] = sales_open
    df["days_since_sales_open"] = (df["sale_date"] - sales_open).dt.days

    ig_roll_cols = [
        "ig_posts_count", "ig_likes_sum", "ig_comments_sum", "ig_saves_sum", "ig_reach_sum",
        "ig_views_sum", "ig_follows_sum", "ig_reel_posts_count", "ig_carousel_posts_count",
        "ig_image_posts_count",
    ]
    for col in ig_roll_cols:
        if col in df:
            grouped = df.groupby("festival_year")[col]
            for window in [3, 7, 14]:
                df[f"{col}_{window}d_prior"] = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    fb_lag_cols = ["fb_posts_count", "fb_total_engagement_sum", "fb_total_reach_sum", "fb_total_clicks_sum", "fb_video_views_3s_sum", "fb_comments_sum", "fb_shares_sum"]
    marketing_lag_cols = ["marketing_total_posts", "marketing_total_reach", "marketing_total_engagement", "marketing_total_views_core", "marketing_total_comments", "marketing_total_shares"]

    _add_marketing_combined(df)
    for col in fb_lag_cols + marketing_lag_cols:
        if col in df:
            grouped = df.groupby("festival_year")[col]
            for lag in [1, 3, 7]:
                df[f"{col}_lag_{lag}"] = grouped.shift(lag)
            for window in [3, 7]:
                df[f"{col}_roll_{window}_prior"] = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    for ticket_type in TICKET_TYPES:
        grouped = df.groupby("festival_year")[ticket_type]
        df[f"{ticket_type}_lag_1d"] = grouped.shift(1)
        df[f"{ticket_type}_roll_3d"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        df[f"{ticket_type}_roll_7d"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())

    df["copywriting_impact_score"] = df.get("copywriting_impact_score", 1.0)
    text_activity = df.get("marketing_total_posts", 0).fillna(0) if isinstance(df.get("marketing_total_posts", 0), pd.Series) else 0
    df["is_announcement"] = (text_activity > 0).astype(int) if isinstance(text_activity, pd.Series) else 0
    if ambassador_expires:
        expiry = pd.Timestamp(ambassador_expires).normalize()
        df["days_until_ambassador_expires"] = (expiry - df["sale_date"]).dt.days.clip(lower=0)
    else:
        first_amb = df.loc[df["Ambassador"] > 0, "sale_date"].min()
        if pd.isna(first_amb):
            df["days_until_ambassador_expires"] = 0
        else:
            df["days_until_ambassador_expires"] = (first_amb + pd.Timedelta(days=30) - df["sale_date"]).dt.days.clip(lower=0)
    df["is_ambassador_expired"] = (df["days_until_ambassador_expires"] <= 0).astype(int)

    _add_lineup_features(df, festival_year)
    return df


def _add_marketing_combined(df: pd.DataFrame) -> None:
    for col in ["ig_posts_count", "ig_reach_sum", "ig_likes_sum", "ig_comments_sum", "ig_shares_sum", "ig_views_sum", "fb_posts_count", "fb_total_reach_sum", "fb_total_engagement_sum", "fb_video_views_3s_sum", "fb_comments_sum", "fb_shares_sum", "fb_total_clicks_sum", "fb_link_clicks_sum"]:
        if col not in df:
            df[col] = 0
    df["marketing_total_posts"] = df["ig_posts_count"].fillna(0) + df["fb_posts_count"].fillna(0)
    df["marketing_total_reach"] = df["ig_reach_sum"].fillna(0) + df["fb_total_reach_sum"].fillna(0)
    df["marketing_total_engagement"] = (
        df["ig_likes_sum"].fillna(0) + df["ig_comments_sum"].fillna(0) + df["ig_shares_sum"].fillna(0) + df["fb_total_engagement_sum"].fillna(0)
    )
    df["marketing_total_views_core"] = df["ig_views_sum"].fillna(0) + df["fb_video_views_3s_sum"].fillna(0)
    df["marketing_total_comments"] = df["ig_comments_sum"].fillna(0) + df["fb_comments_sum"].fillna(0)
    df["marketing_total_shares"] = df["ig_shares_sum"].fillna(0) + df["fb_shares_sum"].fillna(0)
    df["marketing_total_clicks_known"] = df["fb_total_clicks_sum"].fillna(0)
    df["marketing_link_clicks_known"] = df["fb_link_clicks_sum"].fillna(0)
    df["marketing_activity_day"] = (df["marketing_total_posts"] > 0).astype(int)
    df["marketing_both_platforms_active"] = ((df["ig_posts_count"] > 0) & (df["fb_posts_count"] > 0)).astype(int)
    df["marketing_only_instagram_active"] = ((df["ig_posts_count"] > 0) & (df["fb_posts_count"] <= 0)).astype(int)
    df["marketing_only_facebook_active"] = ((df["fb_posts_count"] > 0) & (df["ig_posts_count"] <= 0)).astype(int)
    df["marketing_platforms_active_count"] = (df["ig_posts_count"] > 0).astype(int) + (df["fb_posts_count"] > 0).astype(int)
    df["marketing_avg_reach_per_post"] = np.where(df["marketing_total_posts"] > 0, df["marketing_total_reach"] / df["marketing_total_posts"], 0)
    df["marketing_avg_engagement_per_post"] = np.where(df["marketing_total_posts"] > 0, df["marketing_total_engagement"] / df["marketing_total_posts"], 0)
    df["marketing_avg_views_per_post_core"] = np.where(df["marketing_total_posts"] > 0, df["marketing_total_views_core"] / df["marketing_total_posts"], 0)


def _add_lineup_features(df: pd.DataFrame, festival_year: int) -> None:
    if not LINEUP_PATH.exists():
        return
    lineup = pd.read_csv(LINEUP_PATH)
    year_col = _first_existing(lineup, ["year"])
    if not year_col:
        return
    numeric_cols = [c for c in lineup.columns if c != year_col and pd.api.types.is_numeric_dtype(lineup[c])]
    selected = lineup[lineup[year_col] == festival_year]
    if selected.empty:
        selected = lineup.sort_values(year_col).tail(1)
    if selected.empty:
        return
    for col in numeric_cols:
        df[col] = selected.iloc[0][col]


def finalize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    schema = load_final_schema()
    out = df.copy()
    for col in schema:
        if col not in out:
            out[col] = 0
    out = out[schema]
    date_cols = ["sale_date", "sales_open_date", "# created"]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in out.columns.difference(date_cols):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    missing_model_features = [col for col in MODEL_FEATURES if col not in out.columns]
    return out, missing_model_features


def convert_files(
    ticket_file: tuple[str, pd.DataFrame],
    marketing_files: Iterable[tuple[str, pd.DataFrame]],
    festival_year: int,
    festival_date: str,
    ambassador_expires: str | None = None,
) -> tuple[pd.DataFrame, ConversionSummary]:
    _, ticket_df = ticket_file
    tickets = standardize_tickets(ticket_df, festival_year=festival_year, festival_date=festival_date)
    marketing, warnings = standardize_marketing_files(marketing_files)
    engineered = add_feature_engineering(tickets, marketing, festival_year=festival_year, ambassador_expires=ambassador_expires)
    final, missing_model_features = finalize_schema(engineered)
    marketing_days = int((final["marketing_activity_day"] > 0).sum()) if "marketing_activity_day" in final else 0
    summary = ConversionSummary(
        rows=len(final),
        start_date=str(final["sale_date"].min()),
        end_date=str(final["sale_date"].max()),
        total_tickets=int(final["tickets_sold"].sum()),
        marketing_days=marketing_days,
        missing_model_features=missing_model_features,
        warnings=warnings,
    )
    return final, summary
