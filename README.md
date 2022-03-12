# Predicting Credit Risk For Loan Applications

Several machines learning models have been used in this exercise to be trained on mortgage loan data. The data was captured from the Lending Club as shown below with two tables utilized for the traning and testing of the models.

- Lending Data Table
  | loan_size | interest_rate | homeowner | borrower_income | debt_to_income | num_of_accounts | derogatory_marks | total_debt | loan_status |
  | --------- | ------------- | --------- | --------------- | -------------- | --------------- | ---------------- | ---------- | ----------- |
  | 10700.0 | 7.672 | own | 52800 | 0.431818 | 5 | 1 | 22800 | low_risk |
  | 8400.0 | 6.692 | own | 43600 | 0.311927 | 3 | 0 | 13600 | low_risk |
  | 9000.0 | 6.963 | rent | 46100 | 0.349241 | 3 | 0 | 16100 | low_risk |
  | 10700.0 | 7.664 | own | 52700 | 0.430740 | 5 | 1 | 22700 | low_risk |
  | 10800.0 | 7.698 | mortgage | 53000 | 0.433962 | 5 | 1 | 23000 | low_risk |

- Lending Statistics for Q1 2019

| loan_amnt | int_rate | installment | home_ownership | annual_inc | verification_status |  issue_d | loan_status | pymnt_plan |   dti | ... | pct_tl_nvr_dlq | percent_bc_gt_75 | pub_rec_bankruptcies | tax_liens | tot_hi_cred_lim | total_bal_ex_mort | total_bc_limit | total_il_high_credit_limit | hardship_flag | debt_settlement_flag |     |
| --------: | -------: | ----------: | -------------: | ---------: | ------------------: | -------: | ----------: | ---------: | ----: | --: | -------------: | ---------------: | -------------------: | --------: | --------------: | ----------------: | -------------: | -------------------------: | ------------: | -------------------: | --- |
|   10500.0 |   0.1719 |      375.35 |           RENT |    66000.0 |     Source Verified | Mar-2019 |    low_risk |          n | 27.24 | ... |           85.7 |            100.0 |                  0.0 |       0.0 |         65687.0 |           38199.0 |         2000.0 |                    61987.0 |             N |                    N |     |
|   25000.0 |   0.2000 |      929.09 |       MORTGAGE |   105000.0 |            Verified | Mar-2019 |    low_risk |          n | 20.23 | ... |           91.2 |             50.0 |                  1.0 |       0.0 |        271427.0 |           60641.0 |        41200.0 |                    49197.0 |             N |                    N |     |
|   20000.0 |   0.2000 |      529.88 |       MORTGAGE |    56000.0 |            Verified | Mar-2019 |    low_risk |          n | 24.26 | ... |           66.7 |             50.0 |                  0.0 |       0.0 |         60644.0 |           45684.0 |         7500.0 |                    43144.0 |             N |                    N |     |
|   10000.0 |   0.1640 |      353.55 |           RENT |    92000.0 |            Verified | Mar-2019 |    low_risk |          n | 31.44 | ... |          100.0 |             50.0 |                  1.0 |       0.0 |         99506.0 |           68784.0 |        19700.0 |                    76506.0 |             N |                    N |     |
|   22000.0 |   0.1474 |      520.39 |       MORTGAGE |    52000.0 |        Not Verified | Mar-2019 |    low_risk |          n | 18.76 | ... |          100.0 |              0.0 |                  0.0 |       0.0 |        219750.0 |           25919.0 |        27600.0 |                    20000.0 |             N |                    N |     |

## Two approaches were utilized and are stored in two different notebooks

- Resampling the data and Running Logistic Regression learning by :

  1. Running the logistic regression without and resampling
  1. Oversampling the data using the Naive Random Oversampler and SMOTE algorithms.
  1. Undersampling the data using the Cluster Centroids algorithm.
  1. Combination Over and Undesampling using the SMOTEENN algorithm.

- Running the Ensabmle learning classifiers :
  1. Balanced Random Forest learning
  2. Easy Ensabmle learning

For both of the exercises above the imbalanced classification reports were generated to compare the performance of the models and preprocessing methods used.

## Observations and Summary

In general the resampling did improve the final results and the SMOTE and SMOTEENN models were the best overall balanced accuracy scores of 0.99 however the unbalanced results were 0.98.

Summary Table of the results of the resampling exercises

|        Unbalanced - Balanced Accuracy: 0.9892813049736127 |      |      |      |      |      |      |       |
| --------------------------------------------------------: | ---: | ---: | ---: | ---: | ---: | ---: | ----: |
|                                                           |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                 high_risk | 0.86 | 0.98 | 0.99 | 0.92 | 0.99 | 0.98 |   625 |
|                                                  low_risk | 1.00 | 0.99 | 0.98 | 1.00 | 0.99 | 0.98 | 18759 |
|                                               avg / total | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.98 | 19384 |
| Naive_OverSampler - Balanced Accuracy: 0.9946414201183431 |      |      |      |      |      |      |       |
|                                                           |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                 high_risk | 0.85 | 1.00 | 0.99 | 0.92 | 0.99 | 0.99 |   625 |
|                                                  low_risk | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 18759 |
|                                               avg / total | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 19384 |
|             SMOTE - Balanced Accuracy: 0.9946680739911509 |      |      |      |      |      |      |       |
|                                                           |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                 high_risk | 0.85 | 1.00 | 0.99 | 0.92 | 0.99 | 0.99 |   625 |
|                                                  low_risk | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 18759 |
|                                               avg / total | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 19384 |
|   ClusterCentroid - Balanced Accuracy: 0.9932813049736127 |      |      |      |      |      |      |       |
|                                                           |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                 high_risk | 0.86 | 0.99 | 0.99 | 0.92 | 0.99 | 0.99 |   625 |
|                                                  low_risk | 1.00 | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 18759 |
|                                               avg / total | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 19384 |
|          SMOTEENN - Balanced Accuracy: 0.9946680739911509 |      |      |      |      |      |      |       |
|                                                           |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                 high_risk | 0.85 | 1.00 | 0.99 | 0.92 | 0.99 | 0.99 |   625 |
|                                                  low_risk | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 18759 |
|                                               avg / total | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 19384 |

The Easy Ensamble Classifier had the best balanced score of 0.93 compared to the Balanced Random Forest model which was 0.81

Summary Table for the Ensamble Learning techniques

| BalancedRandom Forest - Balanced Accuracy: 0.8196061012606211 |      |      |      |      |      |      |       |
| ------------------------------------------------------------: | ---: | ---: | ---: | ---: | ---: | ---: | ----: |
|                                                               |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                     high_risk | 0.05 | 0.71 | 0.93 | 0.09 | 0.81 | 0.65 |    87 |
|                                                      low_risk | 1.00 | 0.93 | 0.71 | 0.96 | 0.81 | 0.67 | 17118 |
|                                                   avg / total | 0.99 | 0.93 | 0.71 | 0.96 | 0.81 | 0.67 | 17205 |
| EasyEnsambleClassifier - Balanced Accuracy: 0.927457888651188 |      |      |      |      |      |      |       |
|                                                               |  pre |  rec |  spe |   f1 |  geo |  iba |   sup |
|                                                     high_risk | 0.06 | 0.93 | 0.92 | 0.11 | 0.93 | 0.86 |    87 |
|                                                      low_risk | 1.00 | 0.92 | 0.93 | 0.96 | 0.93 | 0.86 | 17118 |
|                                                   avg / total | 0.99 | 0.92 | 0.93 | 0.96 | 0.93 | 0.86 | 17205 |

Based on the esamble learning techniques it was found that the top 3 features used in the learning model were

- total_rec_prncp : Principal received to date
- total_rec_int: Interest received to date
- total_pymnt : Payments received to date for total amount funded
