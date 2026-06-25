import pandas as pd
import numpy as np


# STEP 1A:
# BUILD MASTER TICKET OPERATIONAL CALENDAR


ticket_files = {
    2022: "Ticket sales standard template 2022.xlsx",
    2023: "Ticket sales standard template 2023.xlsx",
    2024: "Ticket sales standard template 2024.xlsx",
    2025: "Ticket sales standard template 2025.xlsx",
    2026: "Ticket sales standard template 2026.xlsx",
}

output_file = "master_ticket_operational_calendar_2022_2026.csv"

required_columns = [
    "sale_date",
    "festival_year",
    "days_to_event",
    "tickets_sold",
    "is_event_day"
]

all_dfs = []

print("=" * 80)
print("STEP 1A: BUILD MASTER TICKET OPERATIONAL CALENDAR")
print("=" * 80)

# LOAD AND VALIDATE EACH YEAR

for expected_year, file_name in ticket_files.items():

    print("\n" + "=" * 80)
    print(f"LOADING FESTIVAL YEAR {expected_year}")
    print("=" * 80)

    df = pd.read_excel(file_name)

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{file_name} is missing required columns: {missing_cols}"
        )

    # Keep only required columns for master calendar
    df = df[required_columns].copy()

    # Parse dates
    df["sale_date"] = pd.to_datetime(
        df["sale_date"],
        errors="coerce"
    )

    bad_dates = df["sale_date"].isna().sum()

    print("Bad sale_date values:", bad_dates)

    if bad_dates > 0:
        raise ValueError(
            f"{file_name} contains bad sale_date values."
        )

    # Convert numeric columns safely
    df["festival_year"] = pd.to_numeric(
        df["festival_year"],
        errors="coerce"
    ).astype("Int64")

    df["days_to_event"] = pd.to_numeric(
        df["days_to_event"],
        errors="coerce"
    )

    df["tickets_sold"] = pd.to_numeric(
        df["tickets_sold"],
        errors="coerce"
    ).fillna(0)

    df["is_event_day"] = pd.to_numeric(
        df["is_event_day"],
        errors="coerce"
    ).fillna(0).astype(int)

    # Validate festival year
    unique_years = df["festival_year"].dropna().unique()

    print("Festival years found:", unique_years)

    if len(unique_years) != 1 or unique_years[0] != expected_year:
        raise ValueError(
            f"{file_name} has unexpected festival_year values: {unique_years}"
        )

    # Add useful helper columns
    df["actual_calendar_year"] = df["sale_date"].dt.year
    df["source_ticket_file"] = file_name

    # Sort
    df = df.sort_values("sale_date").reset_index(drop=True)

    # Year-level validation
    print("Date range:", df["sale_date"].min().date(), "to", df["sale_date"].max().date())
    print("Rows:", len(df))
    print("Duplicate sale_date within file:", df["sale_date"].duplicated().sum())
    print("Total tickets_sold:", df["tickets_sold"].sum())
    print("Event days:", df["is_event_day"].sum())

    # Check possible missing dates inside min-max range
    full_range = pd.date_range(
        start=df["sale_date"].min(),
        end=df["sale_date"].max(),
        freq="D"
    )

    missing_dates = sorted(
        set(full_range.date) - set(df["sale_date"].dt.date)
    )

    print("Calendar days between start and end:", len(full_range))
    print("Observed rows:", len(df))
    print("Missing dates inside this date range:", len(missing_dates))

    if len(missing_dates) > 0:
        print("First 10 missing dates:")
        print(missing_dates[:10])

    all_dfs.append(df)

# =========================================================
# COMBINE ALL YEARS
# =========================================================

master = pd.concat(all_dfs, ignore_index=True)

master = master.sort_values(["festival_year", "sale_date"]).reset_index(drop=True)

# FINAL VALIDATION

print("\n" + "=" * 80)
print("FINAL MASTER TICKET CALENDAR VALIDATION")
print("=" * 80)

print("Final shape:", master.shape)

print("\nRows per festival_year:")
print(master["festival_year"].value_counts().sort_index())

print("\nDate range per festival_year:")
summary = master.groupby("festival_year").agg(
    start_date=("sale_date", "min"),
    end_date=("sale_date", "max"),
    rows=("sale_date", "count"),
    total_tickets_sold=("tickets_sold", "sum"),
    event_days=("is_event_day", "sum")
).reset_index()

print(summary)

print("\nDuplicate sale_date across all years:", master["sale_date"].duplicated().sum())

if master["sale_date"].duplicated().sum() > 0:
    print("\nWARNING: Duplicate sale_date values found across festival years.")
    print("Inspect these rows:")
    duplicate_dates = master[master["sale_date"].duplicated(keep=False)]
    print(duplicate_dates.sort_values("sale_date"))

print("\nMissing values:")
print(master.isna().sum())

# COVID-cycle validation
covid_cycle = master[master["festival_year"] == 2022]

print("\n" + "=" * 80)
print("COVID-CYCLE VALIDATION FOR FESTIVAL YEAR 2022")
print("=" * 80)

print("2022 festival cycle actual calendar years:")
print(covid_cycle["actual_calendar_year"].value_counts().sort_index())

print("\nThis confirms that 2019–2021 dates belong to festival_year 2022.")

# SAVE FILE
# Save date as YYYY-MM-DD for clean merging later
master["sale_date"] = master["sale_date"].dt.strftime("%Y-%m-%d")

master.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("MASTER TICKET OPERATIONAL CALENDAR SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 1A complete.")
print("Next step will be merging Instagram daily features onto this master calendar.")
