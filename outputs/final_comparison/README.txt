who-gets-funded — final comparison folder (assembled by src/models/16_compare_final.py)

TABLES
  C1_fractional_by_fold.csv      MAE / RMSE / shortfall-MAE, every model, every fold
  C2_headline.csv                fold means and % against the B0 floor
  C3_binary_by_fold.csv          binary aux, untuned vs tuned
  C4_tuning_summary.csv          search budget, winner, gain over the untuned default
  C5_counterfactual_summary.csv  recourse feasibility, both readings
  C6_recourse_by_gender.csv      the fairness-of-recourse result
  C7_calibration_by_group.csv    within-group calibration
  C8_error_rates_by_group.csv    equalised-odds style error rates
  C9_fractional_error_by_group.csv  signed error by group

FIGURES  F35-F36 built here; F25-F34 copied from the step scripts.

SAMPLE NOTE
  R002/R003/R004 were fitted on the 200,003-row development sample.
  R012/R013 are fitted on the full 1,344,542-row cohort. The B0 floors
  come from the full cohort throughout, so the tuned rows are measured
  against their own denominator and the untuned rows against one that
  differs by 0.1-1.2%.

  All numbers are raw model output. Nothing here is interpreted.
