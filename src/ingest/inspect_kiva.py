# Code written with the assistance of Claude (Anthropic).
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "kiva_raw.sqlite"

con = sqlite3.connect(db)
con.execute("PRAGMA query_only = ON")


def show(title, query):
    print(f"\n--- {title} ---")
    cur = con.execute(query)
    columns = [x[0] for x in cur.description]
    print(" | ".join(columns))
    for row in cur.fetchall():
        print(" | ".join("" if x is None else str(x) for x in row))


show("Dataset size", """
SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT loan_id) AS unique_loans,
    MIN(loan_id) AS minimum_id,
    MAX(loan_id) AS maximum_id
FROM loans
""")

show("Date coverage", """
SELECT
    MIN(fundraising_date) AS earliest,
    MAX(fundraising_date) AS latest
FROM loans
""")

show("Loans by status", """
SELECT status, COUNT(*) AS loans
FROM loans
GROUP BY status
ORDER BY loans DESC
""")

show("Numeric summary", """
SELECT
    ROUND(AVG(loan_amount), 2) AS mean_loan_amount,
    MIN(loan_amount) AS minimum_loan_amount,
    MAX(loan_amount) AS maximum_loan_amount,
    ROUND(AVG(share_raised), 4) AS mean_share_raised,
    ROUND(AVG(borrower_count), 2) AS mean_borrowers
FROM loans
""")

show("Missing key fields", """
SELECT
    SUM(status IS NULL) AS missing_status,
    SUM(loan_amount IS NULL) AS missing_loan_amount,
    SUM(share_raised IS NULL) AS missing_share_raised,
    SUM(gender IS NULL OR TRIM(gender) = '') AS missing_gender,
    SUM(country_iso IS NULL OR TRIM(country_iso) = '') AS missing_country,
    SUM(sector IS NULL OR TRIM(sector) = '') AS missing_sector,
    SUM(fundraising_date IS NULL) AS missing_date
FROM loans
""")

show("Top 15 countries", """
SELECT country_name, COUNT(*) AS loans
FROM loans
GROUP BY country_name
ORDER BY loans DESC
LIMIT 15
""")

show("Loans by sector", """
SELECT sector, COUNT(*) AS loans
FROM loans
GROUP BY sector
ORDER BY loans DESC
""")

show("Outcomes by gender", """
SELECT
    COALESCE(NULLIF(TRIM(gender), ''), 'missing') AS gender_group,
    COUNT(*) AS loans,
    ROUND(AVG(share_raised), 4) AS mean_share_raised,
    ROUND(
        100.0 * AVG(CASE WHEN status = 'expired' THEN 1.0 ELSE 0.0 END),
        2
    ) AS expired_percent
FROM loans
GROUP BY gender_group
ORDER BY loans DESC
""")

show("Latest five loan records", """
SELECT
    loan_id, status, loan_amount, share_raised,
    gender, sector, country_name, fundraising_date
FROM loans
ORDER BY loan_id DESC
LIMIT 5
""")

con.close()