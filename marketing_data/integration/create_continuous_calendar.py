import pandas as pd
import numpy as np

# =========================================================
# STEP 1B:
# BUILD TRUE CONTINUOUS MASTER OPERATIONAL CALENDAR
# =========================================================

input_file = "master_ticket_operational_calendar_2022_2026.csv"
output_file = "master_operational_calendar_continuous_2022_2026.csv"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(input_file)

print("=" * 80)
print("STEP 1B: BUILD CONTINUOUS MASTER OPERATIONAL CALENDAR")
print("=" * 80)

print("\nOriginal ticket calendar shape:")
print(df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
    "sale_date",
    "festival_year",
    "days_to_event",
    "tickets_sold",
    "is_event_day",
    "actual_calendar_year",
    "source_ticket_file"
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError("Missing required columns: " + str(missing_cols))

print("\nRequired column check passed.")

# =========================================================
# PARSE AND CLEAN BASIC COLUMNS
# =========================================================

df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

bad_dates = df["sale_date"].isna().sum()

print("\nBad sale_date values:", bad_dates)

if bad_dates > 0:
    raise ValueError("Bad sale_date values found. Stop and inspect.")

df["festival_year"] = pd.to_numeric(df["festival_year"], errors="coerce").astype(int)
df["days_to_event"] = pd.to_numeric(df["days_to_event"], errors="coerce")
df["tickets_sold"] = pd.to_numeric(df["tickets_sold"], errors="coerce").fillna(0)
df["is_event_day"] = pd.to_numeric(df["is_event_day"], errors="coerce").fillna(0).astype(int)

# Mark rows that came from the original ticket files
df["was_original_ticket_row"] = 1

# =========================================================
# BUILD CONTINUOUS CALENDAR PER FESTIVAL YEAR
# =========================================================

continuous_parts = []

print("\n" + "=" * 80)
print("BUILDING CONTINUOUS CALENDAR PER FESTIVAL YEAR")
print("=" * 80)

for festival_year, group in df.groupby("festival_year"):

    group = group.sort_values("sale_date").copy()

    start_date = group["sale_date"].min()
    end_date = group["sale_date"].max()

    print("\nFestival year:", festival_year)
    print("Original start date:", start_date.date())
    print("Original end date:", end_date.date())
    print("Original rows:", len(group))

    # Create full continuous date range
    full_dates = pd.DataFrame({
        "sale_date": pd.date_range(start=start_date, end=end_date, freq="D")
    })

    full_dates["festival_year"] = festival_year

    # Merge original ticket data onto full date range
    merged = full_dates.merge(
        group,
        on=["sale_date", "festival_year"],
        how="left"
    )

    # Mark missing operational days
    merged["was_original_ticket_row"] = merged["was_original_ticket_row"].fillna(0).astype(int)

    # Fill ticket sales for missing operational dates
    # Inside active operational cycle, missing ticket rows mean zero observed sales.
    merged["tickets_sold"] = merged["tickets_sold"].fillna(0)

    # Fill source file for added rows
    source_file_value = group["source_ticket_file"].dropna().iloc[0]
    merged["source_ticket_file"] = merged["source_ticket_file"].fillna(source_file_value)

    # Recalculate actual calendar year
    merged["actual_calendar_year"] = merged["sale_date"].dt.year

    # =====================================================
    # Recalculate days_to_event
    # =====================================================
    # Event start date is the first date where is_event_day == 1.
    # For incomplete 2026, there is no event day yet.
    # In that case, infer event start date from existing days_to_event:
    # event_start_date = sale_date + days_to_event

    event_days = group[group["is_event_day"] == 1]

    if len(event_days) > 0:
        event_start_date = event_days["sale_date"].min()
        print("Event start date from is_event_day:", event_start_date.date())
    else:
        inferred_event_dates = group["sale_date"] + pd.to_timedelta(group["days_to_event"], unit="D")
        event_start_date = inferred_event_dates.mode().iloc[0]
        print("Event start date inferred from days_to_event:", event_start_date.date())

    # Days to event cannot be negative during event period.
    recalculated_days = (event_start_date - merged["sale_date"]).dt.days
    merged["days_to_event"] = recalculated_days.clip(lower=0)

    # Recreate event-day flag
    # For completed years, preserve known event days.
    # For 2026, there are no event days in the current partial data.
    if len(event_days) > 0:
        known_event_dates = set(event_days["sale_date"])
        merged["is_event_day"] = merged["sale_date"].isin(known_event_dates).astype(int)
    else:
        merged["is_event_day"] = 0

    # Add helpful flags
    merged["is_added_calendar_day"] = np.where(
        merged["was_original_ticket_row"] == 0,
        1,
        0
    )

    merged["covid_cycle_flag"] = np.where(
        merged["festival_year"] == 2022,
        1,
        0
    )

    # Validation per year
    print("Continuous rows:", len(merged))
    print("Added calendar days:", merged["is_added_calendar_day"].sum())
    print("Total tickets_sold after filling:", merged["tickets_sold"].sum())

    # Check continuity
    expected_days = (end_date - start_date).days + 1

    if len(merged) != expected_days:
        raise ValueError(f"Calendar continuity failed for festival_year {festival_year}")

    if merged["sale_date"].duplicated().sum() > 0:
        raise ValueError(f"Duplicate dates inside festival_year {festival_year}")

    continuous_parts.append(merged)

# =========================================================
# COMBINE ALL FESTIVAL YEARS
# =========================================================

master_continuous = pd.concat(continuous_parts, ignore_index=True)

master_continuous = master_continuous.sort_values(
    ["festival_year", "sale_date"]
).reset_index(drop=True)

# =========================================================
# FINAL VALIDATION
# =========================================================

print("\n" + "=" * 80)
print("FINAL CONTINUOUS MASTER CALENDAR VALIDATION")
print("=" * 80)

print("Final shape:", master_continuous.shape)

summary = master_continuous.groupby("festival_year").agg(
    start_date=("sale_date", "min"),
    end_date=("sale_date", "max"),
    rows=("sale_date", "count"),
    original_ticket_rows=("was_original_ticket_row", "sum"),
    added_calendar_days=("is_added_calendar_day", "sum"),
    total_tickets_sold=("tickets_sold", "sum"),
    event_days=("is_event_day", "sum"),
    covid_cycle_flag=("covid_cycle_flag", "max")
).reset_index()

print("\nSummary by festival_year:")
print(summary)

print("\nDuplicate sale_date across all festival years:")
print(master_continuous["sale_date"].duplicated().sum())

if master_continuous["sale_date"].duplicated().sum() > 0:
    print("\nWARNING: Duplicate calendar dates across different festival years found.")
    print("This may be acceptable only if festival cycles overlap.")
    print(master_continuous[master_continuous["sale_date"].duplicated(keep=False)])

print("\nMissing values:")
print(master_continuous.isna().sum())

# Validate ticket totals preserved from original input
original_total_tickets = df["tickets_sold"].sum()
continuous_total_tickets = master_continuous["tickets_sold"].sum()

print("\nOriginal total tickets_sold:", original_total_tickets)
print("Continuous calendar total tickets_sold:", continuous_total_tickets)

if original_total_tickets == continuous_total_tickets:
    print("Ticket sales total preservation: PASSED")
else:
    raise ValueError("Ticket sales totals changed after calendar expansion.")

# Validate original rows count preserved
original_rows = len(df)
preserved_rows = master_continuous["was_original_ticket_row"].sum()

print("\nOriginal ticket rows:", original_rows)
print("Preserved original ticket rows:", preserved_rows)

if original_rows == preserved_rows:
    print("Original row preservation: PASSED")
else:
    raise ValueError("Original ticket row count changed after calendar expansion.")

# =========================================================
# SAVE FILE
# =========================================================

master_continuous["sale_date"] = master_continuous["sale_date"].dt.strftime("%Y-%m-%d")

master_continuous.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("CONTINUOUS MASTER OPERATIONAL CALENDAR SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 1B complete.")
print("Next step: merge Instagram daily features onto this continuous master calendar.")