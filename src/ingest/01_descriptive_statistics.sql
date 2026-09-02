-- Code written with the assistance of Claude (Anthropic).
-- Specification, review and validation by Fazal Ur Rehman Cheema.
-- =============================================================================
-- who-gets-funded — summary and descriptive statistics
-- Read-only. No writes, no schema changes, no temp tables.
--
-- Usage:
--   cd path/to/who-gets-funded
--   sqlite3 ~/Downloads/kiva_raw.sqlite < src/ingest/01_descriptive_statistics.sql
--
-- Runtime: expect 25-40 min total. The `loans` table carries the raw JSON in
-- `raw`, so every full scan is slow (80-100s measured). Sections marked [SORT]
-- also sort 3.1m rows and are the slowest; skip them by deleting the block if
-- you want a fast first pass.
--
-- Standard deviations are NOT computed here: SQLite has no STDDEV, and SQRT()
-- is only present in builds with math functions enabled. Each numeric section
-- therefore reports mean and mean_sq (= AVG(x*x)); variance is mean_sq - mean^2.
-- =============================================================================

.mode column
.headers on
.timer on
.nullvalue NULL

.print ''
.print '== D01  numeric summary, all 3,132,059 rows ================================='
SELECT
  COUNT(*)                                        n,
  ROUND(AVG(loan_amount),2)                       loan_amt_mean,
  ROUND(AVG(loan_amount*loan_amount),2)           loan_amt_mean_sq,
  MIN(loan_amount)                                loan_amt_min,
  MAX(loan_amount)                                loan_amt_max,
  ROUND(AVG(funded_amount),2)                     funded_mean,
  ROUND(AVG(share_raised),6)                      share_mean,
  ROUND(AVG(share_raised*share_raised),6)         share_mean_sq,
  ROUND(AVG(borrower_count),4)                    borrowers_mean,
  MAX(borrower_count)                             borrowers_max,
  ROUND(AVG(use_text_length),2)                   use_len_mean,
  MAX(use_text_length)                            use_len_max,
  ROUND(AVG(description_length),2)                desc_len_mean,
  MAX(description_length)                         desc_len_max,
  ROUND(AVG(fundraising_window_days),4)           window_days_mean,
  MIN(fundraising_window_days)                    window_days_min,
  MAX(fundraising_window_days)                    window_days_max,
  ROUND(AVG(lender_repayment_term),4)             repay_term_mean,
  ROUND(AVG(research_score),4)                    research_score_mean
FROM loans;

.print ''
.print '== D02  missingness across every analysis-relevant column ==================='
SELECT
  SUM(loan_amount             IS NULL) loan_amount,
  SUM(funded_amount           IS NULL) funded_amount,
  SUM(share_raised            IS NULL) share_raised,
  SUM(borrower_count          IS NULL) borrower_count,
  SUM(is_group                IS NULL) is_group,
  SUM(gender                  IS NULL) gender,
  SUM(borrower_genders        IS NULL) borrower_genders,
  SUM(sector                  IS NULL) sector,
  SUM(activity                IS NULL) activity,
  SUM(country_iso             IS NULL) country_iso,
  SUM(region                  IS NULL) region,
  SUM(fundraising_date        IS NULL) fundraising_date,
  SUM(planned_expiration_date IS NULL) planned_expiration_date,
  SUM(fundraising_window_days IS NULL) fundraising_window_days,
  SUM(repayment_interval      IS NULL) repayment_interval,
  SUM(lender_repayment_term   IS NULL) lender_repayment_term,
  SUM(distribution_model      IS NULL) distribution_model,
  SUM(anonymization_level     IS NULL) anonymization_level,
  SUM(tags                    IS NULL) tags,
  SUM(themes                  IS NULL) themes,
  SUM(use_text_length         IS NULL) use_text_length,
  SUM(description_length      IS NULL) description_length,
  SUM(original_language       IS NULL) original_language,
  SUM(is_repeat_borrower      IS NULL) is_repeat_borrower,
  SUM(is_matchable            IS NULL) is_matchable,
  SUM(match_ratio             IS NULL) match_ratio,
  SUM(partner_id              IS NULL) partner_id,
  SUM(research_score          IS NULL) research_score
FROM loans;

.print ''
.print '== D03  binary flags and empty-string counts, single pass ==================='
SELECT
  SUM(is_group=1)                     is_group_1,
  SUM(is_matchable=1)                 is_matchable_1,
  SUM(is_repeat_borrower=1)           repeat_borrower_1,
  SUM(in_pfp=1)                       in_pfp_1,
  SUM(delinquent=1)                   delinquent_1,
  SUM(previous_loan_id IS NOT NULL)   has_previous_loan,
  SUM(TRIM(COALESCE(tags,''))='')     tags_blank,
  SUM(TRIM(COALESCE(themes,''))='')   themes_blank,
  SUM(COALESCE(use_text_length,0)=0)  use_text_empty,
  SUM(COALESCE(description_length,0)=0) desc_empty
FROM loans;

.print ''
.print '== D04  by status ==========================================================='
SELECT status, COUNT(*) n,
       ROUND(100.0*COUNT(*)/3132059,3)  pct,
       ROUND(AVG(share_raised),6)       share_mean,
       ROUND(AVG(loan_amount),2)        loan_amt_mean,
       MIN(loan_amount)                 loan_amt_min,
       MAX(loan_amount)                 loan_amt_max,
       ROUND(AVG(borrower_count),3)     borrowers_mean,
       ROUND(AVG(use_text_length),1)    use_len_mean,
       ROUND(AVG(fundraising_window_days),2) window_days_mean
FROM loans GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== D05  by gender ==========================================================='
SELECT gender, COUNT(*) n,
       ROUND(100.0*COUNT(*)/3132059,3)  pct,
       SUM(status='expired')            expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(share_raised),6)       share_mean,
       ROUND(AVG(loan_amount),2)        loan_amt_mean,
       ROUND(AVG(borrower_count),3)     borrowers_mean,
       SUM(is_group=1)                  group_loans,
       ROUND(AVG(use_text_length),1)    use_len_mean,
       SUM(is_matchable=1)              matchable
FROM loans GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== D06  by sector =========================================================='
SELECT sector, COUNT(*) n,
       ROUND(100.0*COUNT(*)/3132059,3)  pct,
       SUM(status='expired')            expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(share_raised),6)       share_mean,
       ROUND(AVG(loan_amount),2)        loan_amt_mean,
       SUM(gender='female')             female,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== D07  by country, top 30 of all countries ================================'
SELECT country_iso, country_name, COUNT(*) n,
       SUM(status='expired')            expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(share_raised),6)       share_mean,
       ROUND(AVG(loan_amount),2)        loan_amt_mean,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct
FROM loans GROUP BY 1,2 ORDER BY 3 DESC LIMIT 30;

.print '-- country cardinality --'
SELECT COUNT(DISTINCT country_iso) countries, COUNT(DISTINCT region) regions,
       COUNT(DISTINCT activity) activities, COUNT(DISTINCT sector) sectors,
       COUNT(DISTINCT partner_id) partners FROM loans;

.print ''
.print '== D08  by posting year ===================================================='
SELECT SUBSTR(fundraising_date,1,4) yr, COUNT(*) n,
       SUM(status='expired')            expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       SUM(status='refunded')           refunded,
       SUM(status='fundraising')        still_live,
       ROUND(AVG(share_raised),6)       share_mean,
       ROUND(AVG(loan_amount),2)        loan_amt_mean,
       ROUND(AVG(fundraising_window_days),2) window_days_mean,
       SUM(planned_expiration_date IS NULL) no_expiry_date,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct,
       SUM(typename='LoanDirect')       loan_direct
FROM loans WHERE fundraising_date IS NOT NULL GROUP BY 1 ORDER BY 1;

.print ''
.print '== D09  group size bands ==================================================='
SELECT CASE WHEN borrower_count=1 THEN '1'
            WHEN borrower_count BETWEEN 2 AND 4  THEN '2-4'
            WHEN borrower_count BETWEEN 5 AND 9  THEN '5-9'
            WHEN borrower_count BETWEEN 10 AND 19 THEN '10-19'
            WHEN borrower_count >= 20 THEN '20+' ELSE 'other' END band,
       COUNT(*) n,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(loan_amount),2) loan_amt_mean
FROM loans GROUP BY 1 ORDER BY MIN(borrower_count);

.print ''
.print '== D10  categorical distributions =========================================='
.print '-- repayment_interval --'
SELECT repayment_interval v, COUNT(*) n, ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC;
.print '-- distribution_model --'
SELECT distribution_model v, COUNT(*) n, ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC;
.print '-- anonymization_level --'
SELECT anonymization_level v, COUNT(*) n, ROUND(AVG(share_raised),6) share_mean
FROM loans GROUP BY 1 ORDER BY 2 DESC;
.print '-- original_language, top 15 --'
SELECT original_language v, COUNT(*) n, ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
.print '-- typename --'
SELECT typename v, COUNT(*) n, ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(loan_amount),2) loan_amt_mean
FROM loans GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== D11  match funding ======================================================'
SELECT is_matchable, match_ratio, COUNT(*) n,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(loan_amount),2) loan_amt_mean
FROM loans GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20;

.print '-- matcher_name, top 15 --'
SELECT COALESCE(matcher_name,'(null)') matcher, COUNT(*) n,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

.print ''
.print '== D12  lender_repayment_term bands ========================================'
SELECT CASE WHEN lender_repayment_term IS NULL THEN '(null)'
            WHEN lender_repayment_term <= 6   THEN '<=6'
            WHEN lender_repayment_term <= 12  THEN '7-12'
            WHEN lender_repayment_term <= 24  THEN '13-24'
            WHEN lender_repayment_term <= 36  THEN '25-36'
            ELSE '37+' END band,
       COUNT(*) n, ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== D13  partner concentration, top 20 ======================================'
SELECT partner_id, partner_name, COUNT(*) n,
       ROUND(100.0*COUNT(*)/3132059,3) pct,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(AVG(loan_amount),2) loan_amt_mean
FROM loans GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20;

.print ''
.print '== D14  narrative length by status ========================================='
SELECT status,
       ROUND(AVG(use_text_length),1)     use_len_mean,
       MIN(use_text_length)              use_len_min,
       MAX(use_text_length)              use_len_max,
       ROUND(AVG(description_length),1)  desc_len_mean,
       MIN(description_length)           desc_len_min,
       MAX(description_length)           desc_len_max
FROM loans GROUP BY 1 ORDER BY 1;

.print ''
.print '== D15  loan_amount bands vs outcome ======================================='
SELECT CASE WHEN loan_amount <   250 THEN 'a <250'
            WHEN loan_amount <   500 THEN 'b 250-499'
            WHEN loan_amount <  1000 THEN 'c 500-999'
            WHEN loan_amount <  2500 THEN 'd 1000-2499'
            WHEN loan_amount <  5000 THEN 'e 2500-4999'
            WHEN loan_amount < 10000 THEN 'f 5000-9999'
            ELSE 'g 10000+' END band,
       COUNT(*) n,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct,
       ROUND(AVG(borrower_count),3) borrowers_mean
FROM loans GROUP BY 1 ORDER BY 1;

.print ''
.print '== D16  gender x sector, mean share and expiry ============================='
SELECT sector, gender, COUNT(*) n,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(100.0*SUM(status='expired')/COUNT(*),3) expiry_pct
FROM loans WHERE gender IN ('female','male')
GROUP BY 1,2 ORDER BY 1,2;

.print ''
.print '== D17  expired loans only: how far short did they fall? ==================='
SELECT CASE WHEN share_raised =  0    THEN 'a raised nothing'
            WHEN share_raised <  0.25 THEN 'b <25%'
            WHEN share_raised <  0.50 THEN 'c 25-49%'
            WHEN share_raised <  0.75 THEN 'd 50-74%'
            WHEN share_raised <  0.95 THEN 'e 75-94%'
            ELSE 'f 95%+' END band,
       COUNT(*) n,
       ROUND(100.0*COUNT(*)/93321,3) pct_of_expired,
       ROUND(AVG(loan_amount),2) loan_amt_mean,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct
FROM loans WHERE status='expired' GROUP BY 1 ORDER BY 1;

.print ''
.print '== D18  harvest_progress table ============================================='
SELECT COUNT(*) batches, MIN(rowid) min_rowid, MAX(rowid) max_rowid FROM harvest_progress;
SELECT * FROM harvest_progress ORDER BY rowid LIMIT 3;
SELECT * FROM harvest_progress ORDER BY rowid DESC LIMIT 3;

.print ''
.print '== D19  indexes present on loans ==========================================='
SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='loans';

.print ''
.print '== D20  sample of tags and themes values, for parsing later ================'
SELECT tags, COUNT(*) n FROM loans WHERE TRIM(COALESCE(tags,''))<>'' GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
SELECT themes, COUNT(*) n FROM loans WHERE TRIM(COALESCE(themes,''))<>'' GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

-- =============================================================================
-- [SORT] quantiles. These sort 3.1m rows and are the slowest blocks in the file.
-- Delete from here down for a faster first pass.
-- =============================================================================

.print ''
.print '== D21  [SORT] quantiles, loan_amount ======================================'
WITH t AS (SELECT loan_amount v, ROW_NUMBER() OVER (ORDER BY loan_amount) rn,
                  COUNT(*) OVER () n FROM loans WHERE loan_amount IS NOT NULL)
SELECT MAX(CASE WHEN rn=CAST(n*0.01 AS INT) THEN v END) p01,
       MAX(CASE WHEN rn=CAST(n*0.05 AS INT) THEN v END) p05,
       MAX(CASE WHEN rn=CAST(n*0.25 AS INT) THEN v END) p25,
       MAX(CASE WHEN rn=CAST(n*0.50 AS INT) THEN v END) median,
       MAX(CASE WHEN rn=CAST(n*0.75 AS INT) THEN v END) p75,
       MAX(CASE WHEN rn=CAST(n*0.95 AS INT) THEN v END) p95,
       MAX(CASE WHEN rn=CAST(n*0.99 AS INT) THEN v END) p99
FROM t;

.print ''
.print '== D22  [SORT] quantiles, use_text_length and description_length ==========='
WITH t AS (SELECT use_text_length v, ROW_NUMBER() OVER (ORDER BY use_text_length) rn,
                  COUNT(*) OVER () n FROM loans WHERE use_text_length IS NOT NULL)
SELECT 'use_text_length' col,
       MAX(CASE WHEN rn=CAST(n*0.05 AS INT) THEN v END) p05,
       MAX(CASE WHEN rn=CAST(n*0.25 AS INT) THEN v END) p25,
       MAX(CASE WHEN rn=CAST(n*0.50 AS INT) THEN v END) median,
       MAX(CASE WHEN rn=CAST(n*0.75 AS INT) THEN v END) p75,
       MAX(CASE WHEN rn=CAST(n*0.95 AS INT) THEN v END) p95
FROM t;
WITH t AS (SELECT description_length v, ROW_NUMBER() OVER (ORDER BY description_length) rn,
                  COUNT(*) OVER () n FROM loans WHERE description_length IS NOT NULL)
SELECT 'description_length' col,
       MAX(CASE WHEN rn=CAST(n*0.05 AS INT) THEN v END) p05,
       MAX(CASE WHEN rn=CAST(n*0.25 AS INT) THEN v END) p25,
       MAX(CASE WHEN rn=CAST(n*0.50 AS INT) THEN v END) median,
       MAX(CASE WHEN rn=CAST(n*0.75 AS INT) THEN v END) p75,
       MAX(CASE WHEN rn=CAST(n*0.95 AS INT) THEN v END) p95
FROM t;

.print ''
.print '== D23  [SORT] quantiles of share_raised among expired loans only =========='
WITH t AS (SELECT share_raised v, ROW_NUMBER() OVER (ORDER BY share_raised) rn,
                  COUNT(*) OVER () n FROM loans WHERE status='expired')
SELECT MAX(CASE WHEN rn=CAST(n*0.05 AS INT) THEN v END) p05,
       MAX(CASE WHEN rn=CAST(n*0.25 AS INT) THEN v END) p25,
       MAX(CASE WHEN rn=CAST(n*0.50 AS INT) THEN v END) median,
       MAX(CASE WHEN rn=CAST(n*0.75 AS INT) THEN v END) p75,
       MAX(CASE WHEN rn=CAST(n*0.95 AS INT) THEN v END) p95
FROM t;

.print ''
.print '== end of descriptive statistics ==========================================='
