-- Code written with the assistance of Claude (Anthropic).
-- Specification, review and validation by Fazal Ur Rehman Cheema.
-- =============================================================================
-- who-gets-funded — feat_v1 feature definitions (single source of truth)
--
-- A single read-only SELECT. No dot-commands, no writes, no temp tables, so it
-- can be executed either by the Python runner (which adds a progress bar and the
-- reconciliation checks) or piped straight into the sqlite3 CLI.
--
--   python3 src/features/03_build_features.py          <- preferred, has a progress bar
--   sqlite3 ~/Downloads/kiva_raw.sqlite < this_file     <- needs native LN()
--
-- COHORT (locked 17 Aug 2026, RESULTS_LOG section 2):
--   typename='LoanPartner' AND status IN ('funded','expired')
--   AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'
--   => 1,344,542 rows, 79,993 expired (5.9495%)
--
-- DESIGN DECISIONS (RESULTS_LOG section 13):
--   * partner_id IS a feature (contextual) — never offerable as recourse.
--   * NO historical/as-of performance features. Decided 17 Aug: leakage risk.
--   * user_favorite / volunteer_pick carry an x_leak_ prefix — they are lender
--     behaviour recorded AFTER posting. Emitted only so the leakage hypothesis
--     can be tested. DO NOT put them in a model.
--   * Zero-variance-in-cohort columns are not emitted: typename,
--     distribution_model, is_matchable, match_ratio, matcher_name, in_pfp.
--   * No correlation- or importance-based selection of any kind. Columns qualify
--     on two rules only: available at posting, and non-constant in the cohort.
--   * Realised-outcome fields are never emitted except the two targets.
--
-- TARGETS
--   share_raised     primary fractional outcome, [0,1]
--   is_fully_funded  auxiliary binary, for the counterfactual tooling only
-- =============================================================================

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
