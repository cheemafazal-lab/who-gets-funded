-- Code written with the assistance of Claude (Anthropic).
-- Specification, review and validation by Fazal Ur Rehman Cheema.
-- =============================================================================
-- who-gets-funded — composition check: is the gender gap real, or a country/partner artefact?
-- Read-only. No writes, no views, no temp tables, no schema changes.
--
-- LOCKED COHORT (decided 17 August 2026):
--     typename = 'LoanPartner'          -- excludes the 24,116 US LoanDirect loans
--     status IN ('funded','expired')    -- excludes refunded (10,480) and fundraising (3,905)
--     posted 2013-01-01 .. 2019-12-31   -- complete plannedExpirationDate, and expiry
--                                          is a live phenomenon (5.94% vs 0.41% after 2019)
-- Expected size ~1,344,125 rows. C01 reports the exact figure — my estimate subtracted
-- the three exclusions independently, so treat C01 as the authoritative count.
--
-- Date filter uses a range comparison, NOT SUBSTR(), so the ix_fund index on
-- fundraising_date can be used. Expect this file to run much faster than
-- 01_descriptive_statistics.sql.
--
-- Usage:
--   cd path/to/who-gets-funded
--   sqlite3 ~/Downloads/kiva_raw.sqlite < src/ingest/02_composition_check.sql 2>&1 \
--     | tee outputs/tables/02_composition_check_output.txt
--
-- THE QUESTION. Crude female/male expiry rates differ 4.09x platform-wide. But
-- corr(female share, expiry rate) across the 30 largest markets is -0.52, and 28.2%
-- of all female-tagged loans sit in the Philippines (0.66% expiry). C05-C08 apply
-- direct standardisation: they re-weight each group's country-specific (or
-- partner-specific) expiry rate by ONE common volume distribution, so composition is
-- held constant and only the within-stratum difference survives.
--
-- Read C05 as: if women and men were distributed identically across countries,
-- what would each group's expiry rate be? If the standardised gap is close to the
-- crude gap, the disparity is real. If it collapses, the crude gap was composition.
-- =============================================================================

.mode column
.headers on
.timer on
.nullvalue NULL

.print ''
.print '== C01  locked cohort — reconciliation ======================================'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT COUNT(*) n_cohort,
       SUM(status='expired')                       expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct,
       ROUND(AVG(share_raised),6)                  share_mean,
       ROUND(1-AVG(share_raised),6)                naive_predict_1_mae,
       MIN(fundraising_date)                       first_posted,
       MAX(fundraising_date)                       last_posted,
       COUNT(DISTINCT country_iso)                 countries,
       COUNT(DISTINCT partner_id)                  partners,
       COUNT(DISTINCT sector)                      sectors,
       SUM(planned_expiration_date IS NULL)        no_expiry_date
FROM cohort;

.print ''
.print '== C02  cohort gender split =================================================='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT gender, COUNT(*) n,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM cohort),3) pct_of_cohort,
       SUM(status='expired') expired,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct,
       ROUND(AVG(share_raised),6) share_mean
FROM cohort GROUP BY 1 ORDER BY 2 DESC;

.print ''
.print '== C03  CRUDE gap, female vs male, whole cohort =============================='
.print '-- this is the uncontrolled number; C05 is the controlled one --'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT ROUND(100.0*SUM(gender='female' AND status='expired')/SUM(gender='female'),4) expiry_female,
       ROUND(100.0*SUM(gender='male'   AND status='expired')/SUM(gender='male'),4)   expiry_male,
       ROUND(100.0*SUM(gender='male'   AND status='expired')/SUM(gender='male')
           - 100.0*SUM(gender='female' AND status='expired')/SUM(gender='female'),4) gap_pp,
       ROUND((1.0*SUM(gender='male'   AND status='expired')/SUM(gender='male'))
           / (1.0*SUM(gender='female' AND status='expired')/SUM(gender='female')),4) ratio_male_over_female
FROM cohort WHERE gender IN ('female','male');

.print ''
.print '== C04  where the two groups actually sit — country composition =============='
.print '-- countries with >=500 of each gender, ordered by cohort volume --'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
bycty AS (
  SELECT country_iso, country_name, COUNT(*) n_total,
         SUM(gender='female') nf, SUM(gender='male') nm,
         SUM(gender='female' AND status='expired') ef,
         SUM(gender='male'   AND status='expired') em
    FROM cohort WHERE gender IN ('female','male') GROUP BY 1,2)
SELECT country_iso, country_name, n_total,
       nf, nm,
       ROUND(100.0*nf/(nf+nm),2)   female_pct,
       ROUND(100.0*ef/nf,4)        expiry_female,
       ROUND(100.0*em/nm,4)        expiry_male,
       ROUND(100.0*em/nm - 100.0*ef/nf,4) gap_pp,
       ROUND(100.0*n_total/(SELECT COUNT(*) FROM cohort),3) pct_of_cohort
FROM bycty WHERE nf >= 500 AND nm >= 500
ORDER BY n_total DESC;

.print ''
.print '== C05  DIRECTLY STANDARDISED BY COUNTRY — the headline test ================='
.print '-- crude vs standardised on the SAME country subset, so the only difference'
.print '   between the two rows is composition, not which countries are included --'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
bycty AS (
  SELECT country_iso, COUNT(*) n_total,
         SUM(gender='female') nf, SUM(gender='male') nm,
         SUM(gender='female' AND status='expired') ef,
         SUM(gender='male'   AND status='expired') em
    FROM cohort WHERE gender IN ('female','male') GROUP BY 1),
k AS (SELECT * FROM bycty WHERE nf >= 500 AND nm >= 500),
tot AS (SELECT SUM(n_total) N, SUM(nf) NF, SUM(nm) NM, SUM(ef) EF, SUM(em) EM FROM k)
SELECT 'crude (same subset)' method,
       (SELECT COUNT(*) FROM k) countries_used,
       (SELECT N FROM tot) loans_used,
       ROUND(100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) expiry_female,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot),4) expiry_male,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot)
           - 100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) gap_pp
UNION ALL
SELECT 'standardised by country',
       (SELECT COUNT(*) FROM k),
       (SELECT N FROM tot),
       ROUND(100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot)
           - 100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4);

.print ''
.print '== C06  DIRECTLY STANDARDISED BY FIELD PARTNER =============================='
.print '-- partner is finer than country: it absorbs country plus lender-facing'
.print '   presentation, pricing and promotion practice --'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
byp AS (
  SELECT partner_id, COUNT(*) n_total,
         SUM(gender='female') nf, SUM(gender='male') nm,
         SUM(gender='female' AND status='expired') ef,
         SUM(gender='male'   AND status='expired') em
    FROM cohort WHERE gender IN ('female','male') AND partner_id IS NOT NULL GROUP BY 1),
k AS (SELECT * FROM byp WHERE nf >= 500 AND nm >= 500),
tot AS (SELECT SUM(n_total) N, SUM(nf) NF, SUM(nm) NM, SUM(ef) EF, SUM(em) EM FROM k)
SELECT 'crude (same subset)' method,
       (SELECT COUNT(*) FROM k) partners_used,
       (SELECT N FROM tot) loans_used,
       ROUND(100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) expiry_female,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot),4) expiry_male,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot)
           - 100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) gap_pp
UNION ALL
SELECT 'standardised by partner',
       (SELECT COUNT(*) FROM k),
       (SELECT N FROM tot),
       ROUND(100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot)
           - 100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4);

.print ''
.print '== C07  DIRECTLY STANDARDISED BY COUNTRY x SECTOR ==========================='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
cs AS (
  SELECT country_iso, sector, COUNT(*) n_total,
         SUM(gender='female') nf, SUM(gender='male') nm,
         SUM(gender='female' AND status='expired') ef,
         SUM(gender='male'   AND status='expired') em
    FROM cohort WHERE gender IN ('female','male') GROUP BY 1,2),
k AS (SELECT * FROM cs WHERE nf >= 200 AND nm >= 200),
tot AS (SELECT SUM(n_total) N, SUM(nf) NF, SUM(nm) NM, SUM(ef) EF, SUM(em) EM FROM k)
SELECT 'crude (same subset)' method,
       (SELECT COUNT(*) FROM k) strata_used,
       (SELECT N FROM tot) loans_used,
       ROUND(100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) expiry_female,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot),4) expiry_male,
       ROUND(100.0*(SELECT EM FROM tot)/(SELECT NM FROM tot)
           - 100.0*(SELECT EF FROM tot)/(SELECT NF FROM tot),4) gap_pp
UNION ALL
SELECT 'standardised country x sector',
       (SELECT COUNT(*) FROM k),
       (SELECT N FROM tot),
       ROUND(100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot),4),
       ROUND(100.0*(SELECT SUM(n_total*1.0*em/nm) FROM k)/(SELECT N FROM tot)
           - 100.0*(SELECT SUM(n_total*1.0*ef/nf) FROM k)/(SELECT N FROM tot),4);

.print ''
.print '== C08  does the gap ever reverse? per-partner detail, top 25 by volume ======'
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01'),
byp AS (
  SELECT partner_id, partner_name, country_iso, COUNT(*) n_total,
         SUM(gender='female') nf, SUM(gender='male') nm,
         SUM(gender='female' AND status='expired') ef,
         SUM(gender='male'   AND status='expired') em
    FROM cohort WHERE gender IN ('female','male') AND partner_id IS NOT NULL
    GROUP BY 1,2,3)
SELECT partner_id, SUBSTR(partner_name,1,34) partner, country_iso, n_total,
       ROUND(100.0*nf/(nf+nm),1)   female_pct,
       ROUND(100.0*ef/nf,3)        expiry_female,
       ROUND(100.0*em/nm,3)        expiry_male,
       ROUND(100.0*em/nm - 100.0*ef/nf,3) gap_pp
FROM byp WHERE nf >= 500 AND nm >= 500
ORDER BY n_total DESC LIMIT 25;

.print ''
.print '== C09  cohort data quality — does excluding refunded clear the text nulls? ==='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT COUNT(*) n_cohort,
       SUM(use_text_length    IS NULL) null_use_len,
       SUM(description_length IS NULL) null_desc_len,
       SUM(planned_expiration_date IS NULL) null_expiry_date,
       SUM(fundraising_window_days IS NULL) null_window,
       SUM(lender_repayment_term IS NULL)   null_term,
       SUM(research_score IS NULL)          null_research_score,
       SUM(partner_id IS NULL)              null_partner,
       SUM(TRIM(COALESCE(tags,''))='')      tags_blank,
       SUM(TRIM(COALESCE(themes,''))='')    themes_blank,
       SUM(is_matchable=1)                  matchable
FROM cohort;

.print ''
.print '== C10  cohort by year — confirm within-cohort stability ====================='
WITH cohort AS (
  SELECT * FROM loans
   WHERE typename = 'LoanPartner' AND status IN ('funded','expired')
     AND fundraising_date >= '2013-01-01' AND fundraising_date < '2020-01-01')
SELECT SUBSTR(fundraising_date,1,4) yr, COUNT(*) n,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4) expiry_pct,
       ROUND(AVG(share_raised),6) share_mean,
       ROUND(1-AVG(share_raised),6) naive_mae,
       ROUND(100.0*SUM(gender='female')/COUNT(*),2) female_pct,
       ROUND(100.0*SUM(gender='female' AND status='expired')/SUM(gender='female'),4) expiry_female,
       ROUND(100.0*SUM(gender='male'   AND status='expired')/SUM(gender='male'),4)   expiry_male
FROM cohort GROUP BY 1 ORDER BY 1;

.print ''
.print '== end of composition check =================================================='
