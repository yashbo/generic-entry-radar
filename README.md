# Generic Entry Radar

**Live demo:** https://generic-entry-radar.streamlit.app

Estimates when generic competition can enter for every NDA drug in the FDA Orange Book.
It uses listed patents and exclusivities, and a local LLM explains each drug's outlook.

**Hosted demo** (Streamlit Community Cloud): the hosted copy has no GPU, so instead of AI-written text it shows the verified brief the model would be given. Run it locally with Ollama for full explanations.

## What it computes (per NDA)
| Field | Meaning |
|---|---|
| `entry_all_listed` | Latest expiry of every listed patent and exclusivity. This is the conservative date. |
| `entry_with_carveout` | Same, but ignores method-of-use patents and indication-specific exclusivities (I-, ODE, D-), which a generic can often carve out of its label. |
| `main_barrier` | Which kind of protection expires last: compound, formulation, method-of-use or exclusivity. |
| `status` | One of: Protected / Carve-out possible now / Open: no listed barrier, no generic / Generic approved (a matching ANDA exists for the same ingredient and dosage form/route). |

Handled details: pediatric `*PED` extensions, delisted patents, discontinued products, and multiple strengths per NDA.

**Limits.** The estimates ignore Paragraph IV litigation, settlements, 30-month stays, 180-day exclusivity, and unlisted (e.g. process) patents. Biologics are in the Purple Book, not here. Use the output as a screening signal, not legal advice.

## Setup
```powershell
pip install -r generic_entry_radar/requirements.txt
```
1. Download **Orange Book Data Files (compressed)** in your browser from
   https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
   (FDA blocks scripted downloads). Save the zip into `generic_entry_radar/data/`.
2. `python generic_entry_radar/fetch_data.py` extracts it to `data/orange_book/`.
3. Optional LLM: install [Ollama](https://ollama.com) (`winget install Ollama.Ollama`), then `ollama pull llama3.2:3b` (default, fits a 4 GB GPU). With more memory, `ollama pull llama3.1:8b` and set `OLLAMA_MODEL=llama3.1:8b`.
   You can choose another model with the `OLLAMA_MODEL` env var.
4. `python -m streamlit run generic_entry_radar/app.py`

Without real data the app runs on a synthetic demo fixture.

## Backtest
```powershell
python generic_entry_radar/fetch_history.py   # 7 archived editions (2010-2022) from web.archive.org, ~4.5 MB
python generic_entry_radar/backtest.py        # writes data/backtest/*.csv
```
The dashboard also has a **Backtest** page.

**Method.** Each archived edition predicts entry for drugs that had no generic at that date, using only that edition's data. The truth is the first ANDA approval on or after that date, from the current Orange Book. Timing is scored only for predictions at least 2 years old, which avoids censoring bias. Three predictors are compared:
- `all_listed`: every listed patent and exclusivity
- `carveout`: drops method-of-use patents and label-specific exclusivities
- `compound`: compound patent plus exclusivity only

**Findings (Oct 2026 data), for matured `all_listed` predictions:**
- **No generic arrives** for 43–63% of drugs, even 2+ years after listed protection ends. Patents are not the binding constraint for most drugs; the market isn't attractive enough to generic makers.
- **When a generic does come**, the median error is between −1.3 and +0.1 years, and the error histogram has a sharp spike at 0. Many blockbusters were predicted almost to the day from the 2010 edition (Lexapro, Seroquel, Singulair).
- **Generics arrive more than a year early** in 12–30% of cases in the older editions. Causes include Paragraph IV wins, design-arounds of polymorph or formulation patents (e.g. Lipitor), and settlements.
- **3-year "generic within the window" prediction:** recall is highest for `compound` (0.63–0.84), and precision is 0.17–0.33 for every predictor. Precision is low because of the no-show drugs.

**Caveats.** ANDA approval is not the same as launch. Ingredient names that change between editions cause 1.5–4.3% of mismatches, which are excluded from the error stats.

## Paragraph IV patent challenges
FDA's [Paragraph IV list](https://www.fda.gov/drugs/abbreviated-new-drug-application-anda/patent-certifications-and-suitability-petitions) records every drug where a generic maker has certified that a listed patent is invalid or not infringed. It has 1,650 entries going back to 2004, with the challenged NDA number, first filing date, number of challengers, 180-day exclusivity status and the first challenger's approval and launch dates. fda.gov blocks scripts, so download the Excel file ("Download New Paragraph IV List Excel Spreadsheet") in a browser and save it into `generic_entry_radar/data/piv/`. `piv.py` picks up the newest file.

- **In the dashboard:** a "PIV filed" and "PIV challengers" column, a filter for challenged drugs, a challenge panel and timeline marker in the drug deep dive, and the challenge facts passed to the AI explainer.
- **Current picture:** 246 of the 771 protected drugs already face a challenge (Oct 2026 list). 76% of drugs with "Generic approved, patents still listed" have one, which is how those approvals happen.
- **Parsing:** "Pre-MMA" filings (before Dec 2003) get 8 Dec 2003 as an upper-bound date. 66 old entries with no NDA number (e.g. Pepcid, Tequin) can't be matched.

```powershell
python generic_entry_radar/fetch_partd.py      # Medicare Part D spending 2011-2024, ~17 MB (archived CMS files)
python generic_entry_radar/attractiveness.py   # about 1.5 minutes; writes data/model/
```
The model answers the question the backtest raised: **will a generic come at all?** Its label is whether a generic was approved no later than 2 years after a drug's listed protection ended. It's trained on 2,378 matured backtest cases covering 929 drugs; 42% of them got a generic. Features are **point-in-time**: each comes only from the edition the prediction was made from. They include dosage form and route, modified release, combination product, strengths, drug age, years to entry, patent counts and barrier type, orphan and NCE flags, generics of the same molecule in other forms, **Paragraph IV challenges** filed before the edition date (counts and outcomes are excluded because they would leak the future), and **Medicare Part D sales** (brand and whole-molecule spend in the last full year before the edition).

| | AUC | Avg precision | Brier |
|---|---|---|---|
| Base rate | 0.50 | 0.42 | 0.243 |
| Form + route only (temporal holdout) | 0.77 | 0.52 | 0.193 |
| Part D spend only (temporal holdout) | 0.67 | 0.41 | 0.184 |
| Gradient boosting (temporal holdout) | 0.75 | 0.55 | 0.171 |
| Logistic regression, Orange Book features only (temporal holdout) | 0.76 | 0.58 | 0.163 |
| Logistic regression, adding Part D sales (temporal holdout) | 0.80 | 0.63 | 0.149 |
| **Logistic regression, adding Paragraph IV (final, temporal holdout)** | **0.80** | **0.57** | **0.160** |
| **Same final model, grouped 5-fold CV** | **0.87** | **0.83** | **0.145** |

- **Temporal holdout:** trained on editions before 2016 and tested on drugs from 2016+ editions that were never seen in training. Grouped 5-fold cross-validation gives AUC 0.87 (0.80 before the Paragraph IV features).
- **Why logistic regression:** it beat gradient boosting on unseen later drugs and is well calibrated (deciles sit on the diagonal). Gradient boosting was overconfident at the extremes.
- **Honest takeaway:** dosage form and route carry the most signal on their own. Oral tablets got a generic 69% of the time; injectables 14%; implants, IUDs and patches 17%.
- **Sales data:** adding Medicare Part D spend improved every metric. Brand spend is the fourth most important feature. Spend alone is weak, because clinic-given injectables (Part B) barely appear in Part D.
- **Paragraph IV challenges:** the strongest single signal. 81% of drugs already under challenge got a generic, versus 24% without, and this holds in every edition from 2010 to 2022. Grouped CV improves sharply (AUC 0.80 to 0.87, Brier 0.181 to 0.145). The temporal holdout cannot measure this feature: it contains only newly protected drugs, and none of them had a challenge on file yet. Its small changes have bootstrap confidence intervals that include zero.
- **Where the sales data comes from:** data.cms.gov refuses some networks, so `fetch_partd.py` downloads CMS's own published releases from the Internet Archive. Each edition only sees spending from before its date (newer CMS releases override older ones). The 2010 edition has no earlier spending data, so its spend features are missing and imputed.
- **Surprise:** molecules that already have generics in another form are slightly *less* likely to attract one for a new form (39% vs 42%). These are life-cycle products whose market is already served by the cheap version.
- **In the dashboard:** a "Generic likelihood" column, a KPI for drugs losing protection within 3 years that will likely get a generic, the score passed to the LLM explainer, and a **Model** page with evaluation, feature importance and calibration.

## LLM explainer and its fact check
Small local models misread raw JSON: in early tests llama3.2:3b put the carve-out date in the bottom line and relabelled method-of-use patents as formulation patents. So the design is **code analyses, model writes**:

1. `explain.build_brief` turns a drug's facts into short labelled sentences, with dates already computed and patents already grouped by type. Pediatric extensions (`*PED`) carry no type flags in the Orange Book, so each is folded into the patent it extends rather than listed as an "other" patent.
2. The model rewrites that brief for a business reader (`explain.py`, local Ollama).
3. `grounding.py` checks the answer: no patent number, exclusivity code, date, year or percentage that isn't in the brief; the bottom-line date is the true entry date; no patent number sits under the wrong type.
4. In the app, a failed check gets one retry. A second failure shows the verified brief instead, and the disclaimer is always added in code.

Generation is capped at 450 tokens and 90 seconds. Without a cap, the model once fell into a repetition loop that Ollama never stopped.

```powershell
python generic_entry_radar/eval_explainer.py --n 12
```
**Latest runs (same 12 drugs, 4 GB RTX 3050 laptop GPU):**

| Model | Passed every check | Median time | Runs on |
|---|---|---|---|
| llama3.2:3b (default) | 10 / 12 | 13 s | GPU |
| llama3.1:8b | 12 / 12 | 26 s | 58% CPU, 42% GPU |

Results are in `data/explainer_eval.json` and `data/explainer_eval_llama3.1-8b.json` (`--model` and `--out` pick the model and file). Both 3B failures are invented codes ("EX-001", "NDA-198"), which the check catches, so the app retries or shows the brief. The 8B model is more faithful and writes slightly clearer prose, but it is 2–3.5× slower on this hardware, so 3B stays the default.

The model comparison also improved the pipeline. On the first 8B run, a muddled brief led it to a wrong AirDuo date: pediatric extensions were listed as "other" patents, so the brief's last-barrier date did not match any patent group. The checker also wrongly rejected two correct dates, "May 2037" and "May 10th, 2031". All three are fixed, with tests. A manual read of the first batch of stored answers then found codes the model had made up ("EX-485", "U-203") in 54 of 290: the check only knew real Orange Book code formats. It now flags any code-shaped token not in the brief.

### AI answers on the hosted app
The hosted app has no LLM, so the AI answers for about 300 drugs are written in advance on a laptop:

```powershell
python generic_entry_radar/explanations.py --limit 300
```

- **Which drugs:** protected drugs whose listed protection ends within 5 years, then drugs where a carve-out is possible now, then the biggest Medicare Part D spend, then Paragraph IV challenges.
- **Same pipeline as the Explain button:** write, fact-check, one retry. Only answers that pass are kept in `data/explanations.json`, together with the brief each was checked against.
- **Latest batch:** 272 of 300 drugs have a stored answer; the other 28 failed the check twice and show the brief.
- **Checked again at display time:** the app reruns the fact check against today's data and also requires the drug's status to be unchanged. A stored answer whose dates, numbers or status have gone stale is not shown; the verified brief appears instead.

## Tests
```powershell
python generic_entry_radar/tests/test_entry.py
python generic_entry_radar/tests/test_backtest.py
python generic_entry_radar/tests/test_attractiveness.py
python generic_entry_radar/tests/test_piv.py
python generic_entry_radar/tests/test_grounding.py
python generic_entry_radar/tests/test_explanations.py
```

## Files
- `fetch_data.py`: gets and extracts the Orange Book zip
- `orange_book.py`: loads and cleans the three tilde-delimited files
- `entry.py`: estimation logic and per-drug fact sheet
- `explain.py`: Ollama explainer, grounded only on the fact sheet
- `app.py`: Streamlit dashboard (protection-cliff chart, filterable table, drug deep dive)
- `fetch_history.py`, `backtest.py`, `pages/1_Backtest.py`: archived editions and backtest
- `features.py`, `attractiveness.py`, `pages/2_Model.py`: point-in-time features and the attractiveness model
- `fetch_partd.py`, `partd.py`: Medicare Part D spending and its matching to Orange Book drugs
- `piv.py`: Paragraph IV list parsing and point-in-time challenge features
- `grounding.py`, `eval_explainer.py`: fact checks for LLM answers and the evaluation script
- `radar.py`: assembles the scored table and each drug's full facts (shared by the app and batch scripts)
- `explanations.py`: writes and stores fact-checked AI answers for the hosted app, and re-checks them at display time

## Next ideas
- **Broader sales data.** Medicaid State Drug Utilization Data (data.medicaid.gov) would cover a second payer.
- **Patent claim summaries.** Summarize each patent's claims (Google Patents public data) so the LLM can judge how strong a barrier is.
