# An Intelligent Multi-Layered Phishing Detection Framework using Machine Learning and Behavioral Analysis

## Project Context

This repository documents the machine learning component of a multi-layered phishing detection system. It covers dataset finalization, preprocessing, feature engineering, model training, hyperparameter tuning, stacking ensemble construction, explainability, and evaluation — developed end-to-end in Google Colab.

**Assigned responsibilities (from project task card):**
- Finalize dataset
- Handle preprocessing
- Feature engineering
- Train ML models — Random Forest, XGBoost, LightGBM, CatBoost
- Hyperparameter tuning
- Generate evaluation metrics

**Required deliverables:**
- Clean dataset
- Feature-engineered dataset
- Trained models
- Accuracy comparison table
- Confusion matrix
- ROC curve
- Precision, Recall, F1 score

**Additional goal set by professor:** develop a methodology that is *not* already present in the reviewed literature, so the work is publishable rather than a straightforward reproduction of existing approaches.

---

## 1. Background Research

Two supporting documents informed this project:

1. **Data Analysis and Feature Engineering doc** — described the two source datasets:
   - **PhiUSIIL (Phishing URL Dataset)**: 235,795 samples (134,850 legitimate, 100,945 phishing), sourced from the UCI Repository.
   - **Phishing Email Dataset**: reported as 5,695 samples (4,327 legitimate/ham, 1,368 phishing), sourced from "Kaggle/Enron."

2. **Literature Review** — synthesized recent (2023–2026) peer-reviewed research on multi-layered phishing detection using ML and behavioral analysis. Key gaps identified in the literature that this project explicitly targets:
   - Reactive rather than proactive detection
   - High computational overhead limiting real-time deployment
   - Fragmented, single-layer approaches (URL-only or email-only, never fused)
   - Weak or absent explainability (XAI) in complex hybrid models
   - Data imbalance issues

These gaps directly shaped the **novelty strategy** for this project (see Section 8).

---

## 2. Novelty / Methodology Strategy

Since a plain "train 4 algorithms and compare accuracy" approach is already common in the cited literature, three specific novelty angles were adopted:

1. **Stacked ensemble fusion** — Random Forest, XGBoost, LightGBM, and CatBoost combined via a logistic regression meta-learner, rather than simply reporting which single algorithm performs best.
2. **Multi-layer feature fusion** — URL lexical/structural features **and** email structural + TF-IDF lexical features, explicitly separated into a "real-time-deployable" (lexical-only) layer vs. a "full-page" (content/crawl-based) layer.
3. **Explainability (SHAP)** — feature-level attribution added on top of the final models, directly addressing the literature's identified XAI gap.

---

## 3. Pipeline Overview (Piece-by-Piece)

The pipeline was built incrementally in a single Google Colab notebook, piece by piece.

### Piece 1 — Setup and Data Loading
- Installed `catboost`, `lightgbm`, `optuna`, `shap`, `imbalanced-learn`.
- Mounted Google Drive for persistent storage.
- Loaded the URL dataset and an initial email dataset.

**Issue found:** The email dataset originally loaded was **the raw Enron corpus** (517,401 rows, columns `file` and `message`) — not a labeled phishing/legitimate dataset. It has no label column and is entirely legitimate business email, so it cannot be used alone to train a binary classifier.

**Resolution:** Switched to the labeled Kaggle dataset **"Phishing Email Detection" by Subhadeep Chakraborty** (`subhajournal/phishingemails`), which provides `Email Text` and `Email Type` (`Safe Email` / `Phishing Email`) columns suitable for supervised classification.

### Piece 2 — Data Cleaning
- Downloaded the correct labeled email dataset via `kagglehub`.
- Renamed columns to a consistent schema (`text`, `label`).
- Dropped missing/empty rows and duplicate emails.
- Encoded labels: `Phishing Email → 1`, `Safe Email → 0`.
- Cleaned the URL dataset: dropped duplicate URLs, confirmed no missing values.

### Piece 3 — Feature Engineering
**URL side:** Extended PhiUSIIL's ~50 existing engineered columns with:
- `URLEntropy` (Shannon entropy of the URL string)
- `HasSuspiciousKeyword` (login/verify/secure/etc. keyword flag)
- `DigitToLetterRatio`

**Email side (raw text, no existing structure):** Built handcrafted features from scratch:
- `text_length`, `word_count`, `num_links`, `num_exclamations`, `num_digits`
- `urgency_keyword_count` (based on a curated list of manipulation/urgency phrases)
- `num_html_tags`, `uppercase_word_ratio`, `avg_word_length`, `url_density`

### Piece 3b — TF-IDF / Bag-of-Words (added on request)
- Added a TF-IDF vectorizer (unigrams + bigrams, 500 max features, `min_df=5`, `max_df=0.9`) on cleaned email text.
- Combined handcrafted features + TF-IDF into a single sparse matrix (`X_email_combined`) using `scipy.sparse.hstack`.
- Saved as `.npz` (sparse) rather than CSV for storage efficiency.

### Piece 4 — Train Base Models (first pass)
Trained Random Forest, XGBoost, LightGBM, CatBoost on both datasets with baseline hyperparameters.

**Initial results:**

| Dataset | Model | Accuracy |
|---|---|---|
| URL | RandomForest / XGBoost / LightGBM / CatBoost | ~0.9999–1.0000 |
| Email | XGBoost | 0.9661 |
| Email | LightGBM | 0.9652 |
| Email | CatBoost | 0.9632 |
| Email | RandomForest | 0.9507 |

**Critical issue found:** A **perfect 1.0000 accuracy across all four independent algorithms on the URL dataset** is a strong indicator of data leakage, not genuine model skill.

### Piece 4b/4c/4d — Diagnosing and Fixing URL Data Leakage
Investigated via feature importance and label-correlation analysis. Two separate leakage issues were found and addressed in stages:

1. **Similarity/probability columns** (`URLSimilarityIndex`, `TLDLegitimateProb`, `DomainTitleMatchScore`, `URLTitleMatchScore`) — these are pre-computed by the dataset creators by comparing each URL against known legitimate sites, effectively encoding the label. Dropped these columns. Accuracy barely moved (still ~0.9999), showing this wasn't the full story.

2. **Page-content / crawled-HTML features** (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, `NoOfJS`, `NoOfCSS`, `HasSocialNet`, `HasCopyrightInfo`, `IsHTTPS`, `HasSubmitButton`, `IsResponsive`, `HasFavicon`, `HasHiddenFields`, etc.) — these describe the crawled webpage, not the URL string itself. Legitimate sites in PhiUSIIL tend to have rich, fully-built pages; phishing pages tend to be bare/stripped clones — making the classes trivially separable on "how much of a real website is this," which is **not something available at real-time, pre-click/pre-crawl decision time.**

**Resolution — deliberate feature-set split (this became a core methodology decision, not just a bug fix):**
- **`URL_Lexical`** — 20 features computable from the URL string alone, before ever visiting the site (URL length, digit/letter ratios, entropy, obfuscation indicators, subdomain count, etc.). Represents a realistic, real-time-deployable filter.
- **`URL_Full`** — all 49 features (lexical + page-content), kept as an upper-bound reference requiring page crawling.

**Result after the split:**
- `URL_Lexical`: ~99.5–99.7% accuracy — high, but honest and real-time-realistic.
- `URL_Full`: ~99.98–100% accuracy — expected ceiling, requires crawl access.

This became a reportable finding: *"content-based features achieve near-perfect but unrealistic accuracy; a lightweight lexical-only model trades a small amount of accuracy for real-time, pre-crawl deployability."*

### Piece 4 (final) — Three-Feature-Set Training
Retrained all four models across three feature sets: `URL_Lexical`, `URL_Full`, `Email`.

| Feature Set | Best Model | Accuracy |
|---|---|---|
| URL_Lexical | LightGBM | 0.9967 |
| URL_Full | XGBoost / LightGBM | 1.0000 |
| Email | XGBoost | 0.9661 |

### Piece 5 — Hyperparameter Tuning (Optuna)
- Used Optuna with 3-fold stratified CV, optimizing for **F1 score** (not accuracy — more appropriate for phishing detection since it balances precision/recall).
- **`URL_Full` was excluded from tuning** — already saturated near 100%, no meaningful headroom.
- Defined per-model search spaces for all four algorithms (RandomForest, XGBoost, LightGBM, CatBoost).

**Practical issues encountered:**
- Full tuning (30 trials × 4 models × 2 feature sets = 240 studies) was too slow on Colab CPU.
- **Mitigation attempted 1:** Reduced trial count and scoped down models/feature sets for a quick test run (`N_TRIALS=2`, only LightGBM/XGBoost, only `URL_Lexical`) to confirm the pipeline worked before committing to a long run.
- **GPU setup:** Switched Colab runtime to a T4 GPU (`Runtime → Change runtime type → T4 GPU`). Added `device='cuda'` (XGBoost), `device='gpu'` (LightGBM), `task_type='GPU'` (CatBoost) configs, with automatic CPU fallback via a `torch.cuda.is_available()` check. Note: scikit-learn's RandomForest has no GPU support and was unaffected. `n_jobs=1` was required during GPU cross-validation to avoid multiple processes contending for one GPU.
- **Result:** Full tuning (all 4 models, both feature sets, 15 trials) was **not completed** by the end of this project phase due to time constraints — this remains an open item (see Section 9, Next Steps).
- A `KeyError: 'RandomForest'` was encountered when downstream code (Piece 6) assumed `best_params` was fully populated but only contained entries from the earlier quick-test scope. Diagnosed via inspecting `OBJECTIVES.keys()` / `tuning_targets.keys()` / `N_TRIALS`, which confirmed the trimmed test scope had never been reset. A fallback patch was applied (see Piece 6) rather than blocking on a full tuning rerun.

### Piece 6 — Stacking Ensemble
- Built a `StackingClassifier` (scikit-learn) combining all four base models with a logistic regression meta-learner.
- Used `cv=3` internally so the meta-learner trains on genuine out-of-fold base-model predictions (avoids leakage from base models "memorizing" their own training folds).
- `passthrough=False` — meta-learner sees only the four base models' probability outputs, not raw features.
- **Fallback logic added:** since full tuning (Piece 5) was incomplete, a patched `get_stacking_estimators()` function falls back to baseline default hyperparameters for any model missing from `best_params`, instead of raising a `KeyError`. All four base models in the reported results below used baseline (untuned) parameters.

**Stacking vs. best individual model (pre-email-cleanup results):**

| Feature Set | Best Individual | Individual Acc | Stacking Acc | Gain |
|---|---|---|---|---|
| URL_Lexical | LightGBM | 0.9967 | 0.9966 | ~0 (negligible/negative) |
| URL_Full | XGBoost | 0.9999 | 1.0000 | 0 (saturated) |
| Email | XGBoost | 0.9661 | 0.9684 | +0.0023 |

### Piece 7 — SHAP Explainability
- Applied `shap.TreeExplainer` to the best individual model per feature set (LightGBM for `URL_Lexical`, XGBoost for `Email`) rather than the stacking ensemble itself, since `TreeExplainer` is built for single tree-based models and gives more interpretable feature-level output than explaining a meta-learner over four model outputs.
- Sampled 1,000 rows per feature set for computational efficiency (SHAP on the full test sets would be very slow).

**Top URL_Lexical SHAP features:** `LetterRatioInURL`, `SpacialCharRatioInURL`, `DomainLength`, `URLLength`, `NoOfDegitsInURL` — all genuine lexical/structural signals, no leakage concerns.

**Critical issue found:** Top Email SHAP features included `tfidf_enron`, `tfidf_wrote`, `tfidf_university`, `tfidf_linguistics`, `tfidf_thanks` — these are **corpus-provenance artifacts**, not phishing-language signals. Because the "legitimate" class in the email dataset draws heavily from Enron-style corporate correspondence while the "phishing" class comes from unrelated sources, the model was partly learning "mentions Enron/university" as a proxy for "legitimate," rather than learning generalizable deceptive-intent language. This is the email-domain equivalent of the URL page-content leakage found in Piece 4b.

### Piece 7b — Corpus-Artifact Mitigation
- Refit TF-IDF with an extended stopword list excluding `enron`, `wrote`, `university`, `linguistics`, `thanks`.
- Retrained XGBoost on the cleaned feature set as a quick check.

**Result:** Accuracy dropped only modestly, from 0.9661 → 0.9609. This confirmed most of the model's original signal was genuine (urgency language, link density, exclamations), with only a small fraction attributable to corpus artifacts.

**Decision:** Adopted the clean feature set as the official version going forward. All four base models were retrained on it (`splits['Email']` and `all_results['Email']` updated accordingly):

| Model | Accuracy (clean features) |
|---|---|
| XGBoost | 0.9609 |
| LightGBM | 0.9572 |
| CatBoost | 0.9544 |
| RandomForest | 0.9421 |

The `Email` stacking ensemble was also rerun on the clean features: **Accuracy 0.9609, F1 0.9476** — essentially tied with the best individual model (XGBoost), rather than exceeding it.

### Piece 8 — Final Evaluation
Produced all remaining task-card deliverables in one pass:
- **Full metrics table** (accuracy, precision, recall, F1) for every model × every feature set, including the stacking ensemble — saved as `final_metrics_table.csv`.
- **Confusion matrices** — best individual model and stacking ensemble, per feature set (6 total) — saved as `confusion_matrices.png`.
- **ROC curves** — all models overlaid per feature set, with AUC values — saved as `roc_curves.png`.
- **Classification reports** (precision/recall/F1 per class) for the stacking ensemble on each feature set.

**Final reported results (stacking ensemble, clean data):**

| Feature Set | Accuracy | Precision (Phishing) | Recall (Phishing) | F1 (Phishing) |
|---|---|---|---|---|
| URL_Lexical | ~0.9966 | 1.00 | 1.00 | 1.00 |
| URL_Full | ~1.0000 | 1.00 | 1.00 | 1.00 |
| Email | 0.96 | 0.95 | 0.94 | 0.95 |

Email phishing recall of 0.94 means roughly 6% of actual phishing emails are still missed — flagged as the most operationally important number in the whole project, since missed phishing is more costly than a false alarm on a legitimate email.

---

## 4. Full List of Errors Encountered and Fixes

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | Email dataset was raw Enron corpus with no labels | Wrong Kaggle dataset originally used | Switched to `subhajournal/phishingemails` labeled dataset |
| 2 | URL models scored a perfect 1.0000 accuracy | `URLSimilarityIndex`, `TLDLegitimateProb`, `DomainTitleMatchScore`, `URLTitleMatchScore` pre-encode the label | Dropped these 4 columns |
| 3 | Accuracy stayed ~0.9999 even after fix #2 | Page-content/crawled-HTML features (JS/CSS/image counts, social links, etc.) trivially separate classes | Split into `URL_Lexical` (real-time-safe) vs `URL_Full` (upper bound) feature sets |
| 4 | `KeyError: 'RandomForest'` in Piece 6 | `best_params` only partially populated from an earlier quick-test tuning scope that was never reset to the full model/feature-set list | Added fallback logic (`get_stacking_estimators`) defaulting to baseline hyperparameters for any untuned model |
| 5 | `NameError: name 'tuned_results' is not defined` in Piece 6 comparison table | Full Piece 5 tuning was never completed, so `tuned_results` was never created | Added guard clause defaulting `tuned_results = all_results` when `tuned_results` doesn't exist |
| 6 | Top email SHAP features included `tfidf_enron`, `tfidf_university`, etc. | Corpus-provenance artifact — legitimate/phishing classes partly separable by dataset source, not phishing intent | Extended TF-IDF stopword list to exclude artifact tokens; retrained all four models on the cleaned feature set |
| 7 | Optuna tuning too slow on Colab CPU | 240+ studies (4 models × 2 feature sets × 30 trials) | Reduced trial count for quick tests; switched to T4 GPU runtime with GPU-enabled configs for XGBoost/LightGBM/CatBoost |

---

## 5. Datasets

| Dataset | Source | Samples | Notes |
|---|---|---|---|
| PhiUSIIL Phishing URL Dataset | UCI ML Repository | 235,795 (134,850 legit / 100,945 phishing) | 55 original columns; 4 dropped for leakage; split into lexical (20 features) vs full (49 features) |
| Phishing Email Detection | Kaggle (`subhajournal/phishingemails`) | ~18,000+ (exact split differs from originally documented 5,695/4,327/1,368 — that figure could not be matched to a specific public dataset and was likely from a different/unconfirmed source) | Combined handcrafted structural features + TF-IDF (500 features, unigrams+bigrams) |
| Enron Email Dataset (Kaggle) | `wcukierski/enron-email-dataset` (via user upload) | 517,401 | **Not used for training** — raw, unlabeled, all-legitimate corpus; originally mistaken for the labeled dataset above |

---

## 6. Feature Engineering Summary

**URL — Lexical (real-time-deployable) — 20 features:**
`URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `URLEntropy` *(custom)*, `HasSuspiciousKeyword` *(custom)*, `DigitToLetterRatio` *(custom)*

**URL — Full (requires crawl) — adds 29 page-content features:**
`CharContinuationRate`, `URLCharProb`, `IsHTTPS`, `LineOfCode`, `LargestLineLength`, `HasTitle`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`

**Email — Handcrafted (10 features):**
`text_length`, `word_count`, `num_links`, `num_exclamations`, `num_digits`, `urgency_keyword_count`, `num_html_tags`, `uppercase_word_ratio`, `avg_word_length`, `url_density`

**Email — TF-IDF:** 500 max features, unigrams + bigrams, `min_df=5`, `max_df=0.9`, English stopwords + corpus-artifact words (`enron`, `wrote`, `university`, `linguistics`, `thanks`) excluded.

---

## 7. Models Trained

- Random Forest (`sklearn.ensemble.RandomForestClassifier`)
- XGBoost (`xgboost.XGBClassifier`)
- LightGBM (`lightgbm.LGBMClassifier`)
- CatBoost (`catboost.CatBoostClassifier`)
- Stacking Ensemble (`sklearn.ensemble.StackingClassifier`, logistic regression meta-learner over the four base models above)

All trained independently across three feature sets: `URL_Lexical`, `URL_Full`, `Email` (clean).

---

## 8. Novelty Claims for the Paper

1. **Stacked multi-model ensemble** (RF + XGBoost + LightGBM + CatBoost fused via meta-learner) rather than single-algorithm comparison, as is common in the reviewed literature.
2. **Deliberate real-time-vs-full-page feature separation** for URLs — a lightweight, pre-crawl-deployable lexical model reported alongside a page-content upper bound, directly responding to the literature's "computational overhead / real-time deployment" gap.
3. **Multi-layer feature fusion** on the email side — structural/behavioral-proxy features combined with TF-IDF lexical features.
4. **SHAP-based explainability layer**, addressing the literature's identified XAI gap.
5. **Two independent data-leakage/artifact investigations performed and documented** (URL page-content leakage, email corpus-provenance artifacts) — presented as a methodological strength (rigor in avoiding inflated, unrealistic accuracy) rather than hidden or glossed over.

**Honest caveat to include in the paper:** with current (mostly untuned) base models, the stacking ensemble shows only a **modest gain on Email** (+0.002 F1 in the intermediate run) and is **flat/tied** with the best individual model on `URL_Lexical` and `URL_Full`. Full hyperparameter tuning (Piece 5) was not completed and is the most likely lever to make the stacking ensemble outperform single models more convincingly.

---

## 9. Next Steps / Open Items

1. **Complete full Optuna hyperparameter tuning** (all 4 models × `URL_Lexical` + `Email`, ideally 15+ trials each) — currently the single highest-value remaining task, since it's most likely to improve the stacking ensemble's advantage over individual models.
2. **Re-run the stacking ensemble** on fully tuned base models once (1) is complete, and update the final comparison table accordingly.
3. **Verify the email dataset's true source/size** — the originally documented split (5,695 / 4,327 / 1,368) never matched a specific dataset found; the project proceeded with `subhajournal/phishingemails` instead. Confirm with the team whether the original figures came from a different specific file, and reconcile the written data-analysis report with the dataset actually used.
4. **Write the final paper**, structured as: dataset & multi-layer feature engineering → leakage/artifact discoveries as methodology strength → baseline model comparison → stacking ensemble results (with honest framing on modest gains) → SHAP explainability findings → limitations and future work.
5. **Deploy to the team website** (Flask/Streamlit) — see Section 10.

---

## 10. Deployment Handoff (for the Website/Frontend Teammate)

The trained models are not meant to be used from inside the Colab notebook. A `deploy_artifacts/` folder was exported from Google Drive containing:

```
deploy_artifacts/
├── URL_Lexical_StackingEnsemble.pkl
├── Email_StackingEnsemble.pkl
├── tfidf_vectorizer_clean.pkl
├── url_feature_columns.pkl
├── email_feature_columns.pkl
└── version_info.json
```

Alongside this, a standalone `feature_engineering.py` module was extracted from the notebook (via Colab's `%%writefile` magic), containing the exact preprocessing/feature-extraction functions (`extract_url_features`, `extract_email_features`, `clean_for_tfidf`) needed to convert a raw URL or email into the same feature format the models were trained on.

**Recommended website project structure:**

```
phishing-detector-website/
│
├── app/
│   ├── app.py                      # Flask entrypoint (or streamlit_app.py)
│   ├── feature_engineering.py      # must match training exactly
│   ├── model_loader.py
│   ├── models/                     # deploy_artifacts/ contents go here
│   ├── templates/                  # Flask HTML pages
│   ├── static/                     # Flask CSS/JS
│   └── utils/
│       └── explain.py              # optional SHAP-based per-prediction explanation
│
├── notebooks/
│   └── training_pipeline.ipynb     # original Colab notebook, kept for reference
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Critical rule for any future modification:** the `.pkl` model files cannot be edited directly — they are frozen trained snapshots. To change the model itself, retrain from a **copy** of the original Colab notebook (never edit the original directly), then re-run the export cell to produce updated artifacts. If the feature set changes in any way (added/removed/reordered columns), `feature_engineering.py` in the website **must** be updated to match exactly, or the app will silently produce incorrect predictions without throwing an error.

---

## 11. Environment / Libraries Used

- Python (Google Colab)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (RandomForest, StackingClassifier, TfidfVectorizer, train_test_split, metrics)
- `xgboost`, `lightgbm`, `catboost`
- `optuna` (hyperparameter tuning)
- `shap` (explainability)
- `imbalanced-learn` (installed for SMOTE; class imbalance handling)
- `kagglehub` (dataset download)
- `scipy.sparse` (sparse matrix handling for TF-IDF + handcrafted feature fusion)

---

## 12. Team

**Shankha Suvro Dutta** — Dataset finalization, preprocessing, feature engineering, model training (Random Forest, XGBoost, LightGBM, CatBoost), hyperparameter tuning, evaluation metrics generation.

*(Other team members' sections — behavioral analysis layer, website/frontend, etc. — to be added by respective owners.)*
