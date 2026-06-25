import pandas as pd

# -----------------------------
# 1. Load the raw Excel file
# -----------------------------
file_path = "Ticket sale data 2026.xlsx"
df = pd.read_excel(file_path)

print("Raw data loaded.")
print(df.head())
print(df.columns)

# -----------------------------
# 2. Keep only needed columns
# -----------------------------
df = df[["# created", "count(1)"]].copy()

# rename columns to simpler names
df = df.rename(columns={
    "# created": "created_at",
    "count(1)": "ticket_count"
})

print("\nAfter keeping only needed columns:")
print(df.head())

# -----------------------------
# 3. Convert timestamp to proper datetime
#    Handle mixed timezone offsets safely
# -----------------------------
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

# convert UTC back to Europe/Amsterdam local time
df["created_at"] = df["created_at"].dt.tz_convert("Europe/Amsterdam")

# keep only the date part
df["sale_date"] = df["created_at"].dt.date

print("\nAfter converting to date:")
print(df.head())

# -----------------------------
# 4. Remove rows with missing values
# -----------------------------
df = df.dropna(subset=["sale_date", "ticket_count"])

# make sure ticket_count is numeric
df["ticket_count"] = pd.to_numeric(df["ticket_count"], errors="coerce")
df = df.dropna(subset=["ticket_count"])

# -----------------------------
# 5. Group by date and sum tickets sold
# -----------------------------
daily_sales = df.groupby("sale_date", as_index=False)["ticket_count"].sum()

# rename to match template
daily_sales = daily_sales.rename(columns={"ticket_count": "tickets_sold"})

print("\nDaily aggregated sales:")
print(daily_sales.head())

# -----------------------------
# 6. Add festival_year
# -----------------------------
daily_sales["festival_year"] = 2026

# -----------------------------
# 7. Add event-day information
# -----------------------------
event_days = [
    pd.to_datetime("2026-06-25").date(),
    pd.to_datetime("2026-06-26").date(),
    pd.to_datetime("2026-06-27").date()
]

daily_sales["is_event_day"] = daily_sales["sale_date"].isin(event_days).astype(int)

# -----------------------------
# 8. Add days_to_event
#    festival starts on 2022-06-23
# -----------------------------
festival_start = pd.to_datetime("2026-06-25").date()

daily_sales["days_to_event"] = daily_sales["sale_date"].apply(
    lambda d: max(0, (festival_start - d).days)
)

# -----------------------------
# 9. Reorder columns to match template
# -----------------------------
daily_sales = daily_sales[
    ["sale_date", "festival_year", "days_to_event", "tickets_sold", "is_event_day"]
]

# -----------------------------
# 10. Sort by date
# -----------------------------
daily_sales = daily_sales.sort_values("sale_date").reset_index(drop=True)

print("\nFinal template data:")
print(daily_sales.head(10))
print(daily_sales.tail(10))

# -----------------------------
# 11. Save to Excel
# -----------------------------
output_file = "Ticket sales standard template 2026.xlsx"
daily_sales.to_excel(output_file, index=False)

print(f"\nDone. File saved as: {output_file}")




# -----------------------------
# 12. VALIDATION CHECKS
# -----------------------------

# CHECK 1: Total tickets match
raw_total = df["ticket_count"].sum()
processed_total = daily_sales["tickets_sold"].sum()

print("\nCHECK 1: TOTAL TICKETS")
print("Raw total:", raw_total)
print("Processed total:", processed_total)

if raw_total == processed_total:
    print("Totals match")
else:
    print("Totals DO NOT match")

# CHECK 2: Random date check
sample_date = daily_sales["sale_date"].iloc[10]  # pick any index

raw_check = df[df["sale_date"] == sample_date]["ticket_count"].sum()
processed_check = daily_sales[daily_sales["sale_date"] == sample_date]["tickets_sold"].values[0]

print("\nCHECK 2: RANDOM DATE CHECK")
print("Date:", sample_date)
print("Raw:", raw_check)
print("Processed:", processed_check)

if raw_check == processed_check:
    print("Date check passed")
else:
    print("Date check failed")




daily_sales["sale_date"].diff().value_counts()