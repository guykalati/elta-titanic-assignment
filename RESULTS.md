# Evaluation results

## Design

Only Kaggle train.csv (891 passengers) is used. A fixed stratified seed-42 split
reserves 179 passengers and uses 712 for development. EDA uses development labels.
All 18 candidates use the same five stratified development folds. Imputation,
category vocabulary, scaling, age-group medians and ticket dictionaries are fitted
within each fold. Selection uses out-of-fold accuracy, then log loss. The final
PyTorch configuration and weights are saved before holdout metrics are computed.
The probability threshold is fixed at 0.5. Three fixed training seeds are averaged.

## Final PyTorch classifier

A linear layer with engineered features and sigmoid probabilities was selected.

| Evaluation | Accuracy | F1 | ROC AUC |
| --- | ---: | ---: | ---: |
| Development OOF (selection) | 82.9% | 76.5% | 0.875 |
| Development grouped sensitivity | 82.4% | 75.8% | 0.872 |
| Reserved validation | 83.2% | 76.6% | 0.862 |
| Majority reference on validation | 61.5% | 0.0% | 0.500 |

Validation: 149/179 correct. Precision 83.1%, recall 71.0%, Brier score 0.129.
Approximate 95% Wilson accuracy interval: 77.1%–88.0%. Confusion matrix (actual rows,
predicted columns; outcome order 0,1): `[[100,10],[20,49]]`.

## Feature and model comparison

| Configuration | OOF accuracy | OOF F1 | ROC AUC |
| --- | ---: | ---: | ---: |
| extended_torch_0 | 82.87% | 76.54% | 0.875 |
| eda_torch_16 | 82.72% | 75.64% | 0.868 |
| eda_torch_32 | 82.72% | 75.74% | 0.863 |
| eda_torch_0 | 82.44% | 75.73% | 0.877 |
| eda_forest | 82.44% | 75.92% | 0.881 |
| extended_forest | 82.44% | 75.82% | 0.881 |
| eda_xgb_3 | 82.30% | 75.58% | 0.884 |
| basic_torch_16 | 82.30% | 74.60% | 0.856 |
| extended_xgb_2 | 82.16% | 76.44% | 0.880 |
| basic_torch_32 | 82.16% | 74.55% | 0.855 |
| extended_xgb_3 | 81.88% | 75.52% | 0.884 |
| basic_xgb_2 | 81.88% | 74.76% | 0.872 |
| eda_xgb_2 | 81.32% | 74.67% | 0.880 |
| basic_xgb_3 | 81.32% | 73.87% | 0.874 |
| extended_torch_16 | 81.18% | 74.43% | 0.866 |
| extended_torch_32 | 81.18% | 74.62% | 0.863 |
| basic_forest | 80.62% | 72.40% | 0.869 |
| basic_torch_0 | 79.92% | 72.34% | 0.852 |

Basic uses age, fare, family counts, sex, class and port. EDA adds title,
missingness, log fare, family group and sex/class interaction. Extended adds
training-group age estimates, life-stage/sex/class and sex/family interactions,
deck, cabin token count, ticket prefix, training-reference ticket frequency and a
fare-per-observed-ticket-member proxy. Unknown categories are handled by the
saved encoder; rare categories use a training-frequency threshold.

For the same linear architecture, accuracy increased from 79.92% to 82.44% with
EDA features and 82.87% with the extended block. These blocks are bundled ablations;
individual feature contributions are not established. The final extension adds
only three correct OOF predictions over the EDA linear model, so its gain is small.
The best XGBoost accuracy was 82.30%. Deeper networks did not improve the selected
linear model; the extended MLPs were worse. These results apply to the configurations tested.

## Limits

The holdout was evaluated after selection and was not used to retune the model.
Selection among 18 candidates makes the winning CV score optimistic. A 179-row
holdout gives a wide interval; one error changes accuracy by about 0.56 percentage
points. Wilson intervals assume independent observations and may be optimistic
with related passengers.

Some random-split passengers share tickets or approximate families across
partitions. Development grouped sensitivity keeps connected ticket/family
components together (564 components), giving 82.44% accuracy. This is a sensitivity
analysis for new travel groups, not a second holdout or nested model-selection
estimate. Family keys are approximate.

Ticket counts are observed training-reference counts, including the training row
when fitting and falling back to one for unseen tickets. Fare divided by that
count is a proxy; it is not proven actual per-person payment. Missing age is
retained as an indicator; extended life stage is based on imputed age when absent.
Raw PassengerId, raw name and raw ticket identity are not model predictors.

Accuracy guided selection. Other objectives could favor a different model.
The final model misses 20 of 69 survivors in validation.

`CabinCount` is the number of whitespace-separated cabin tokens. A token can be a
deck notation rather than a distinct cabin; this feature is a coarse text proxy.

## Post-selection development diagnostics

The optional `python ablation.py` diagnoses the already selected specification.
It uses the original development folds and fixed hyperparameters. It does not
change selection, weights, or final validation results. Baseline OOF probabilities
are checked against the saved selection run.

| Diagnostic | OOF accuracy | Difference from selected baseline |
| --- | ---: | ---: |
| Selected baseline | 82.87% | — |
| Remove Title indicator | 82.30% | −0.56 points |
| Remove engineered family block | 82.58% | −0.28 points |
| Remove demographic interactions | 82.16% | −0.70 points |
| Remove cabin block | 82.87% | 0.00 points |
| Remove ticket block | 82.44% | −0.42 points |
| Global replacement for numeric Age only | 82.44% | −0.42 points |
| Raw numeric Fare only | 82.87% | 0.00 points |

Effects are conditional on the remaining correlated predictors and involve only
2–5 passengers. Title-based age estimates remain when the title indicator is
removed. Numeric-age/fare replacements do not change their related interaction
or ticket-adjusted features. Cabin removal ties accuracy but slightly lowers F1/AUC.
The selected model was kept fixed during these diagnostics. Full values are in
`artifacts/ablation.csv`.
