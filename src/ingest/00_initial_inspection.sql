-- Code written with the assistance of Claude (Anthropic).
-- Specification, review and validation by Fazal Ur Rehman Cheema.
-- who-gets-funded — initial post-harvest inspection
-- Run 17 August 2026 by Fazal Ur Rehman Cheema. Read-only; no writes to the database.
-- Database: ~/Downloads/kiva_raw.sqlite  (12 GB, 7 Aug 2026 02:25)
-- Note: first invocation targeted who-gets-funded/data/ and failed — the database
-- lives in ~/Downloads and is deliberately left there.
-- Usage: sqlite3 ~/Downloads/kiva_raw.sqlite < 00_initial_inspection.sql
-- Output of this run: 00_initial_inspection_output_2026-08-17.txt

.mode column
.headers on

-- Row count and coverage against the platform totalCount at harvest
SELECT COUNT(*) loans, ROUND(100.0*COUNT(*)/3152795,2) pct_of_platform,
       (SELECT COUNT(*) FROM harvest_progress) batches FROM loans;

-- Status distribution and mean share raised
SELECT status, COUNT(*) n, ROUND(AVG(share_raised),4) mean_share FROM loans GROUP BY 1 ORDER BY 2 DESC;

-- Posting range
SELECT MIN(fundraising_date) first_posted, MAX(fundraising_date) last_posted FROM loans;

-- Integrity: bounds, overfunding, nulls in the analysis-critical columns
SELECT SUM(share_raised>1.0) share_gt_1, SUM(funded_amount>loan_amount) overfunded,
       SUM(loan_amount IS NULL OR loan_amount<=0) bad_amount,
       SUM(share_raised IS NULL) null_share, SUM(fundraising_date IS NULL) null_date,
       SUM(gender IS NULL) null_gender, SUM(sector IS NULL) null_sector,
       SUM(country_iso IS NULL) null_country FROM loans;

-- Expiry rate by gender, population figures
SELECT gender, COUNT(*) n, SUM(status='expired') expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),2) expiry_pct FROM loans
WHERE gender IS NOT NULL GROUP BY 1;

-- Product split: LoanPartner vs LoanDirect
SELECT typename, COUNT(*) n FROM loans GROUP BY 1;

-- plannedExpirationDate availability by posting year
SELECT SUBSTR(fundraising_date,1,4) yr, COUNT(*) n, SUM(planned_expiration_date IS NULL) no_expiry_date
FROM loans WHERE fundraising_date IS NOT NULL GROUP BY 1 ORDER BY 1;
