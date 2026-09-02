who-gets-funded — model comparison folder (assembled by src/models/12_compare_models.py)

TABLES
  T1_fractional_by_fold.csv   MAE / RMSE / shortfall-MAE per fold per model
  T2_headline.csv             fold means, % vs the B0 naive floor
  T3_binary.csv               binary aux vs AUC/PR/accuracy floors
  T4_oot_by_year.csv          out-of-regime (if run)
  T5_cf_summary.csv           counterfactual feasibility (if run)
  T6_recourse_by_gender.csv   recourse audit (if run)

FIGURES  F21-F23 built here; F13-F20 copied from the step scripts
  F13_gbm_gain_importance.png, F14_binary_roc_pr.png, F15_oot_degradation.png, F16_shap_global_bar.png, F17_shap_beeswarm.png, F18_shap_gender_compare.png, F19_cf_actions.png, F20_recourse_gap.png

MISSING INPUTS AT ASSEMBLY TIME: none

All numbers are raw model outputs. Nothing here is tuned and nothing is
interpreted; analysis happens in the report stage.
