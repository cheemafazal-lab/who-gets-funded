-- Code written with the assistance of Claude (Anthropic).
-- Specification, review and validation by Fazal Ur Rehman Cheema.
-- =============================================================================
-- who-gets-funded — feat_v1 feature matrix build
-- Read-only SELECT. No writes to the database, no views, no temp tables.
-- Writes ONE csv to outputs/tables/feat_v1_full.csv, then prints a summary.
--
-- COHORT (locked 17 Aug 2026, RESULTS_LOG §2):
--     typename='LoanPartner' AND status IN ('funded','expired')
--     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'
--   => 1,344,542 rows, 79,993 expired (5.9495%)
--
-- DESIGN DECISIONS BAKED IN (RESULTS_LOG §13):
--   * partner_id IS a feature (contextual) — never offerable as recourse.
--   * NO historical/as-of performance features (partner or country past expiry
--     rates). Decided 17 Aug: leakage risk not worth the accuracy.
--   * user_favorite / volunteer_pick are emitted with an x_leak_ prefix ONLY so
--     the leakage hypothesis can be tested. They are lender behaviour recorded
--     after posting. DO NOT put them in a model.
--   * Zero-variance-in-cohort columns are not emitted at all: typename,
--     distribution_model, is_matchable, match_ratio, matcher_name, in_pfp.
--   * Outcome/realised fields are never emitted except the two targets below.
--
-- TARGETS
--   share_raised     — primary fractional outcome, [0,1]
--   is_fully_funded  — auxiliary binary for the counterfactual tooling
--
-- Usage:
--   cd path/to/who-gets-funded
--   sqlite3 ~/Downloads/kiva_raw.sqlite < src/features/03_build_features.sql
--   # then: python3 src/features/03b_make_dev_sample.py
--
-- Expect ~2-4 min and a 200-350 MB csv.
-- =============================================================================

.headers on
.mode csv
.once outputs/tables/feat_v1_full.csv

WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner'
     AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01'
     AND fundraising_date < '2020-01-01'
),
t AS (SELECT *, ','||COALESCE(tags,'')||','   AS tg,
                ','||COALESCE(themes,'')||',' AS th FROM cohort)
SELECT
  -- ---- identifiers and split keys (NOT predictors) ------------------------
  loan_id,
  fundraising_date,
  CAST(SUBSTR(fundraising_date,1,4) AS INTEGER)            AS posting_year,

  -- ---- targets -----------------------------------------------------------
  share_raised,
  CASE WHEN status='funded' THEN 1 ELSE 0 END              AS is_fully_funded,

  -- ---- protected / audit-only (never offerable as recourse) --------------
  gender,
  country_iso,

  -- ---- actionable: amount and structure ----------------------------------
  loan_amount,
  ROUND(LN(loan_amount), 6)                                AS log_loan_amount,
  ROUND(loan_amount * 1.0 / NULLIF(borrower_count,0), 4)   AS amount_per_borrower,

  -- ---- contextual: group structure ---------------------------------------
  borrower_count,
  is_group,

  -- ---- actionable: narrative ---------------------------------------------
  use_text_length,
  description_length,
  CASE WHEN use_text_length    IS NULL THEN 0 ELSE 1 END   AS has_use_text,
  CASE WHEN description_length IS NULL THEN 0 ELSE 1 END   AS has_description,

  -- ---- actionable: timing ------------------------------------------------
  CAST(SUBSTR(fundraising_date,6,2) AS INTEGER)            AS posting_month,
  CAST(strftime('%w', SUBSTR(fundraising_date,1,10)) AS INTEGER) AS posting_dow,
  CAST(strftime('%j', SUBSTR(fundraising_date,1,10)) AS INTEGER) AS posting_doy,

  -- ---- semi-actionable: partner-set loan terms ---------------------------
  fundraising_window_days,
  repayment_interval,
  lender_repayment_term,
  anonymization_level,
  min_note_size,

  -- ---- contextual: business and geography --------------------------------
  sector,
  activity,
  region,
  partner_id,
  original_language,
  is_repeat_borrower,
  CASE WHEN previous_loan_id IS NULL THEN 0 ELSE 1 END     AS has_previous_loan,

  -- ---- actionable: presentation tags (descriptive only) ------------------
  CASE WHEN TRIM(COALESCE(tags,''))='' THEN 0
       ELSE LENGTH(tags) - LENGTH(REPLACE(tags,',','')) + 1 END
     - (CASE WHEN INSTR(tg, ',user_favorite,')  > 0 THEN 1 ELSE 0 END)
     - (CASE WHEN INSTR(tg, ',volunteer_pick,') > 0 THEN 1 ELSE 0 END)
                                                            AS n_tags_desc,
  CASE WHEN INSTR(tg, ',#Woman-Owned Business,')  > 0 THEN 1 ELSE 0 END AS tag_woman_owned,
  CASE WHEN INSTR(tg, ',#Parent,')                > 0 THEN 1 ELSE 0 END AS tag_parent,
  CASE WHEN INSTR(tg, ',#Repeat Borrower,')       > 0 THEN 1 ELSE 0 END AS tag_repeat_borrower,
  CASE WHEN INSTR(tg, ',#Elderly,')               > 0 THEN 1 ELSE 0 END AS tag_elderly,
  CASE WHEN INSTR(tg, ',#Animals,')               > 0 THEN 1 ELSE 0 END AS tag_animals,
  CASE WHEN INSTR(tg, ',#Eco-friendly,')          > 0 THEN 1 ELSE 0 END AS tag_eco_friendly,
  CASE WHEN INSTR(tg, ',#Health and Sanitation,') > 0 THEN 1 ELSE 0 END AS tag_health_sanitation,
  CASE WHEN INSTR(tg, ',#Technology,')            > 0 THEN 1 ELSE 0 END AS tag_technology,

  -- ---- contextual: programme themes (partner/platform assigned) ----------
  CASE WHEN TRIM(COALESCE(themes,''))='' THEN 0
       ELSE LENGTH(themes) - LENGTH(REPLACE(themes,',','')) + 1 END      AS n_themes,
  CASE WHEN INSTR(th, ',Underfunded Areas,')      > 0 THEN 1 ELSE 0 END AS theme_underfunded,
  CASE WHEN INSTR(th, ',Rural Exclusion,')        > 0 THEN 1 ELSE 0 END AS theme_rural_exclusion,
  CASE WHEN INSTR(th, ',Start-Up,')               > 0 THEN 1 ELSE 0 END AS theme_startup,
  CASE WHEN INSTR(th, ',Crop Insurance,')         > 0 THEN 1 ELSE 0 END AS theme_crop_insurance,
  CASE WHEN INSTR(th, ',Vulnerable Groups,')      > 0 THEN 1 ELSE 0 END AS theme_vulnerable,
  CASE WHEN INSTR(th, ',Conflict Zones,')         > 0 THEN 1 ELSE 0 END AS theme_conflict_zones,
  CASE WHEN INSTR(th, ',Water and Sanitation,')   > 0 THEN 1 ELSE 0 END AS theme_water_sanitation,
  CASE WHEN INSTR(th, ',Higher Education,')       > 0 THEN 1 ELSE 0 END AS theme_higher_education,
  CASE WHEN INSTR(th, ',Refugees/Displaced,')     > 0 THEN 1 ELSE 0 END AS theme_refugees,
  CASE WHEN INSTR(th, ',Clean Energy,')           > 0 THEN 1 ELSE 0 END AS theme_clean_energy,

  -- ---- retained but EXCLUDED from feat_v1 (sensitivity checks only) ------
  research_score                                           AS x_excl_research_score,
  CASE WHEN INSTR(tg, ',user_favorite,')  > 0 THEN 1 ELSE 0 END AS x_leak_user_favorite,
  CASE WHEN INSTR(tg, ',volunteer_pick,') > 0 THEN 1 ELSE 0 END AS x_leak_volunteer_pick
FROM t
ORDER BY fundraising_date, loan_id;

-- ---------------------------------------------------------------------------
.mode column
.headers on
.timer on

.print ''
.print '== FB01  export reconciliation — must match 1,344,542 / 79,993 =============='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT COUNT(*) rows_exported,
       SUM(status='expired')                        expired,
       SUM(status='funded')                         funded,
       ROUND(AVG(share_raised),6)                   share_mean,
       ROUND(1-AVG(share_raised),6)                 naive_mae,
       SUM(loan_amount IS NULL OR loan_amount<=0)   bad_amount_ln_risk,
       SUM(borrower_count IS NULL OR borrower_count=0) bad_borrower_count,
       COUNT(DISTINCT partner_id)                   partners,
       COUNT(DISTINCT activity)                     activities,
       COUNT(DISTINCT country_iso)                  countries
FROM cohort;

.print ''
.print '== FB02  tag and theme flag coverage (are the flags actually firing?) ======='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
t AS (SELECT ','||COALESCE(tags,'')||',' tg, ','||COALESCE(themes,'')||',' th FROM cohort)
SELECT SUM(INSTR(tg,',#Woman-Owned Business,')>0)  woman_owned,
       SUM(INSTR(tg,',#Parent,')>0)                parent,
       SUM(INSTR(tg,',#Repeat Borrower,')>0)       repeat_borrower,
       SUM(INSTR(tg,',#Elderly,')>0)               elderly,
       SUM(INSTR(tg,',#Animals,')>0)               animals,
       SUM(INSTR(tg,',#Eco-friendly,')>0)          eco_friendly,
       SUM(INSTR(tg,',#Health and Sanitation,')>0) health_sanitation,
       SUM(INSTR(tg,',#Technology,')>0)            technology,
       SUM(INSTR(tg,',user_favorite,')>0)          x_user_favorite,
       SUM(INSTR(tg,',volunteer_pick,')>0)         x_volunteer_pick,
       SUM(INSTR(th,',Underfunded Areas,')>0)      th_underfunded,
       SUM(INSTR(th,',Rural Exclusion,')>0)        th_rural,
       SUM(INSTR(th,',Start-Up,')>0)               th_startup,
       SUM(INSTR(th,',Clean Energy,')>0)           th_clean_energy
FROM t;

.print ''
.print '== FB03  leakage probe — do the two x_leak_ tags predict the outcome? ======='
.print '-- if expiry differs sharply by these flags, the leakage call was right --'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT CASE WHEN INSTR(','||COALESCE(tags,'')||',', ',user_favorite,')>0
            THEN 'user_favorite' ELSE 'no user_favorite' END flag,
       COUNT(*) n, ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct,
       ROUND(AVG(share_raised),6) share_mean
FROM cohort GROUP BY 1;
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT CASE WHEN INSTR(','||COALESCE(tags,'')||',', ',volunteer_pick,')>0
            THEN 'volunteer_pick' ELSE 'no volunteer_pick' END flag,
       COUNT(*) n, ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct,
       ROUND(AVG(share_raised),6) share_mean
FROM cohort GROUP BY 1;

.print ''
.print '== FB04  exporting the COMPLETE tag and theme vocabularies =================='
.print '-- D20 was truncated at 15; these are every distinct combination with counts,'
.print '   so the flag list above can be checked for anything high-frequency missed --'

.mode csv
.headers on
.once outputs/tables/feat_v1_tag_combinations.csv
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT COALESCE(tags,'(null)') tags, COUNT(*) n,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct
FROM cohort GROUP BY 1 ORDER BY 2 DESC;

.once outputs/tables/feat_v1_theme_combinations.csv
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename='LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT COALESCE(themes,'(null)') themes, COUNT(*) n,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct
FROM cohort GROUP BY 1 ORDER BY 2 DESC;

.mode column
.print '   -> outputs/tables/feat_v1_tag_combinations.csv'
.print '   -> outputs/tables/feat_v1_theme_combinations.csv'

.print ''
.print '== end of feature build ====================================================='
