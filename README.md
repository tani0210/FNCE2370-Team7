# FNCE 2370 | Team 7 — Predicting Corporate Distress Risk

Fall 2026 · Topic 5 · Parts I–III: Data preparation, feature construction and chronological research design.

We prepare a common U.S. security-month panel for predicting one-month-ahead corporate failure, following the economic logic of Campbell, Hilscher and Szilagyi (2008; CHS). This code supports the eight-predictor dynamic logit benchmark and expanded models with 37 additional Global Characteristics. Model estimation, evaluation and robustness analyses in Parts IV–VII are separate group contributions. **The Parts I–III pipeline has been run successfully on the supplied full dataset; upstream source construction clarifications are pending.**

## Setup and replication

1. Obtain the instructor-provided `Project_5_data_chs_main_global.parquet` through the course and place it in the repository root. The raw and derived datasets are not distributed in this repository.
2. Open the repository root in VS Code (with Python and Jupyter extensions) or use another Jupyter-compatible environment. The verified local environment was Python **3.11.5**, NumPy 1.24.3, pandas 2.0.3, PyArrow 11.0.0, Matplotlib 3.7.2 and ipykernel 6.25.0. Install the direct dependencies if needed:
   ```bash
   python -m pip install -r requirements.txt
   ```
   These are the tested direct versions, not a full environment lockfile. Use Python 3.11. Select that environment as the notebook kernel. The compressed raw file is about 2.44 GB; allow several GB of working RAM and disk space for intermediate outputs.
3. Run the notebooks in the following order from the repository root. Shut down the previous kernel between notebooks to free memory. Do not bypass failed assertions.

| Order | Notebook | Main purpose / reproducible outputs |
|---|---|---|
| 00 | `00_source_provenance_checks.ipynb` | Source metadata, candidate merge keys, event-history and company-level sensitivity diagnostics (`provenance_*.csv`). Standalone. |
| 01 | `01_data_exploration.ipynb` | Raw schema, coverage, target comparison and feature distributions (displayed in notebook). Standalone. |
| 02 | `02_part_I_distress_definition.ipynb` | Target and at-risk validation, merge diagnostics, sample attrition, censoring and lifecycle tables; `part_I_panel.parquet`. |
| 03 | `03_part_II_feature_engineering.ipynb` | CHS construction checks, feature coverage and selection, preprocessing parameters, manifest, missingness and economic diagnostic tables; `part_II_features.parquet`; `global_feature_coverage.png`. |
| 04 | `04_part_III_chronological_design.ipynb` | Chronological splits, common modeling panel, split statistics, environment and preprocessing verification; `part_III_model_panel.parquet`. |

The notebooks automatically create `data/processed/`, `outputs/tables/` and `outputs/figures/` as needed. Re-run the relevant notebook to reproduce its tables and figures. Saved notebook outputs are cleared in the repository copies to avoid exposing individual financial records or local paths.

Run the synthetic-data unit tests from the repository root:
```bash
python -m unittest discover -s tests -v
```
Three tests cover fixed transformations, missing/infinite values, and rejection of invalid training/required features. All passed in the local replication.

## Research design and modeling choices

| Choice | Rationale |
|---|---|
| One-month conditional failure probability | Required CHS-style discrete-time hazard. Use supplied `failure_t_plus_1` (exported as `target`), validated against the first recorded failure date; no additional shifting. |
| Observation unit and dates | Unique `permno` + `month_end` security-month key; retain PERMCO and GVKEY. `feature_month` is calendar month-end and `forecast_month` is the next calendar month-end. |
| Common universe | Matched U.S. common primary main observations on main exchanges, at risk at the feature date. Use the same eligible sample for benchmark and expanded models. |
| Censoring | Preserve missing outcomes for forecasts, never replace them with zeros, and exclude them from supervised fitting/evaluation. |
| CHS predictors | `nimtaavg`, `tlmta`, `exretavg`, `sigma`, `rsize`, `cashmta`, `mb`, `price_log`; preserve interpretable profitability, leverage, market performance, volatility, size, liquidity, valuation and price signals. |
| Baseline missingness | Require finite values for seven CHS predictors; impute only SIGMA using the clipped training mean. Relative to complete-case deletion, this retained 432 additional positive initial-training observations. |
| Expanded predictors | 39 economically motivated candidate Global Characteristics, screened by at least 60% finite initial-training coverage. Retain 37; exclude `rd_sale` and `rd5_at`. Correlated features remain available for regularized or other expanded models. |
| Outliers and imputation | Pooled 5th/95th percentile winsorization, estimated on the eligible labeled estimation sample only; expanded predictors use clipped training medians. Infinite values are missing. No shared scaling; fit model-specific scaling within the relevant training sample. |
| Chronological design | Split by **forecast month**, not feature month: initial training through 1995, validation 1996–2003, untouched-for-selection test 2004–2024. No random train/test split or random cross-validation. |
| Raw feature export | Preserve untransformed values and metadata to support expanding/rolling-window estimation and alternative preprocessing. Natural class proportions retained. |

The benchmark models $P(Y_{i,t+1}=1\mid\mathcal I_t,A_{i,t}=1)=\Lambda(\alpha+X_{i,t}'\beta)$, a conditional monthly hazard rather than cumulative multimonth risk. Score-only models require development-based calibration before interpretation as probabilities. The notebook Markdown cells contain additional justifications and feature definitions.

## Verified sample and replication results

The supplied panel has 3,502,382 rows and 223 columns, of which 3,442,786 (98.2984%) are matched. Part I retains 3,416,761 forecast rows (3,415,782 labeled). The common Part III panel has 2,623,210 rows, 55 columns (45 predictors + 10 metadata fields), and 2,623,095 labeled rows. All 252 test forecast months are present.

| Common sample (forecast month) | Training | Validation | Test |
|---|---:|---:|---:|
| Period | May 1966–Dec 1995 | Jan 1996–Dec 2003 | Jan 2004–Dec 2024 |
| Forecast rows | 1,042,167 | 594,123 | 986,920 |
| Labeled rows | 1,042,054 | 594,121 | 986,920 |
| Unique labeled securities | 12,847 | 10,624 | 9,620 |
| Unique labeled companies (PERMCO) | 12,723 | 10,512 | 9,511 |
| Positive security-months | 2,207 | 1,925 | 1,335 |
| Distinct PERMCO-month failure events | 2,205 | 1,925 | 1,331 |
| Failure prevalence (%) | 0.211793 | 0.324008 | 0.135269 |

These are descriptive replication checks, not test-performance-based model selection criteria. Detailed statistics are reproduced in `outputs/tables/chronological_split_summary.csv`.

## Model handoff (Parts IV onward)

Use `data/processed/part_III_model_panel.parquet`, the ordered `outputs/tables/feature_manifest.csv`, `outputs/tables/preprocessing_params_initial.csv`, and `src/preprocessing.py`. The manifest is the authoritative predictor allowlist; **do not use all numeric columns as predictors**. The broader Part II panel supports alternative sample restrictions.

Example for fixed initial training/validation (run from the repository root):
```python
import pandas as pd
from src.preprocessing import transform_features

manifest = pd.read_csv('outputs/tables/feature_manifest.csv')
selected = manifest.loc[manifest['selected'].astype(str).str.lower().eq('true')]
chs = selected.loc[selected['source'].eq('CHS'), 'feature'].tolist()
global_features = selected.loc[selected['source'].eq('Global Characteristics'), 'feature'].tolist()
features = chs  # or chs + global_features for an expanded model
params = pd.read_csv('outputs/tables/preprocessing_params_initial.csv', index_col='feature').loc[features]
panel = pd.read_parquet('data/processed/part_III_model_panel.parquet',
                        columns=['split', 'target', 'labeled'] + features)

def development_xy(split):
    assert split in ('train', 'validation')
    rows = panel.loc[panel['split'].eq(split) & panel['labeled']]
    return transform_features(rows[features], params), rows['target'].astype(int)

X_train, y_train = development_xy('train')
X_val, y_val = development_xy('validation')
```

- During development, choose features, hyperparameters, thresholds and calibration using only pre-2004 data. Refit preprocessing and any scaling within each training fold for rolling-origin validation. Do not tune on test outcomes.
- After the design is frozen, refit preprocessing and the selected model on the 1,636,175 eligible labeled development rows through December 2003. Part III's `DESIGN_FROZEN` is deliberately `False`; set it to `True` only when the team is ready to generate final preprocessing parameters. This does not fit model coefficients. Fit/choose calibration and thresholds using pre-2004 out-of-sample development predictions; document their transfer to the refitted model. Do not calibrate on final-test labels or in-sample fitted probabilities.
- Generate probabilities for all eligible test rows; evaluate competing models on the same labeled rows. Preserve PERMNO, PERMCO, feature month, forecast month, target, split and each model's probability for the final prediction file. Account for within-company dependence when conducting inference. The representative-security diagnostic is available as a basis for robustness testing.

## Source provenance and limitations (TA clarification pending)

- **Merge:** the instructor's file is premerged. On 1,785,767 matched initial-training-window rows, numeric Global Characteristics `id` equals PERMNO and both calendar month-end and trading dates agree; 1,771,231 comparable market-equity pairs agree within tolerance. This supports a candidate security-month upstream merge (`permno`, `month_end`) to (`id`, `eom`), not proof of its original join code or source-level uniqueness. Subsequent joins explicitly use PERMNO and month-end. Confirmation that using the supplied merge satisfies the course requirement is pending.
- **Security versus company:** selecting the largest contemporaneous-market-equity representative per PERMCO-month retained 99.76% of labeled training rows and 99.91% of positives, with no conflicting observed labels in repeated company-month groups. Retain security-months and report company-level event counts separately; this sensitivity check establishes composition, not equivalent model performance.
- **Event definition:** all 2,983 recorded failure security-months through November 1995 coincide with exits, but no event-type flags establish whether failure is bankruptcy-only or the broader CHS bankruptcy/financial-delisting/D-rating definition. Use the supplied corporate failure indicator pending confirmation.
- **Information timing:** same-month alignment and complete-history rolling checks support the supplied timing, but accounting publication dates, data age, restatement treatment, exact CHS/Global Characteristics release and incomplete-history imputation are not independently verifiable. The final refit assumes December 2003 outcomes are available for January 2004 forecasting. No arbitrary second accounting lag is imposed. Our own fitted transformations are estimation-sample-only; upstream point-in-time availability remains an assumption, not a certified fact.
- **CHS deviations:** supplied NIMTAAVG matches monthly geometric weighting on 869,404 complete histories, rather than CHS quarterly-block weighting; EXRETAVG matches the monthly three-month-half-life convention on complete histories. Daily SIGMA construction is not independently reproduced. We use pooled training-based winsorization and training-mean SIGMA imputation rather than the original cross-sectional imputation. Results should be described as a CHS-inspired replication and extension, not an exact reproduction.

The pending clarifications can be incorporated without changing the design unless they reveal a substantive construction issue. Do not commit the instructor's data, derived panels or generated outputs without redistribution permission; `.gitignore` excludes these.

## Reference

Campbell, J. Y., Hilscher, J., & Szilagyi, J. (2008). In Search of Distress Risk. *Journal of Finance*, 63(6), 2899–2939. https://doi.org/10.1111/j.1540-6261.2008.01416.x
