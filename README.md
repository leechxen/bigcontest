# Big Contest weather-impact modeling

This repository contains the modeling scripts and summarized results. Raw card,
weather, and flow data are not included; obtain them separately and place them
in the folders below.

## Input files

- `card_data/`: the tab-separated card 1 `.txt` or `.tsv` file containing
  `TA_YMD`, `MCT_SGG_CD`, `MCT_RY_CD`, `TS_AT`, `USE_CNT`, and `TIME_GB`.
- `weather_data/`: the tab-separated Korea Meteorological Administration
  heat/cold-wave daily files expected by script 02, plus hourly observation CSVs
  named `OBS_강남_시간별_TIM_*.csv` and `OBS_춘천_시간별_TIM_*.csv`.
- `flow_data/`: weekday population files named `flow_wkdy_pop_*.csv` for script
  05 and 06. The files use `|` delimiters and include `STD_YM`, `BLOCK_CD`, and
  the seven weekday `FLOW_POP_CNT_*` columns.

Script 02 expects heat-wave files for July–September 2025 and cold-wave files
for November–December 2025. The weather and hourly observation inputs must cover
the date range needed by the card data. Script 05/06 only use the weekday flow
files; the age/time flow files are not needed for the checked-in modeling flow.

## Environment

The existing outputs were generated with Python 3.13.1 and the pinned core
versions in `requirements.txt`. Python 3.10 or newer is required by the code;
using Python 3.13.1 is recommended for closest compatibility. The extended
booster models in script 06 are installed separately via
`requirements-extended.txt`. SHAP is not used.

On Windows PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For XGBoost, LightGBM, and CatBoost models in script 06, use this instead of the
last install command (it includes the core requirements):

```powershell
python -m pip install -r requirements-extended.txt
```

## Rebuild datasets and run models

From the repository root, run scripts 01 and 02 first; they can run in either
order. Script 03 combines their outputs. The additional preparation script
creates the previously missing `modeling_dataset_v2_filtered.csv`. Its default
minimum of 18 non-missing October observations per region-industry matches the
filter in the existing checked-in comparison input.

```powershell
python modeling\01_make_modeling_dataset.py
python modeling\02_make_weather_daily.py
python modeling\03_make_modeling_dataset_v1.py
python modeling\prepare_modeling_dataset_v2.py
```

Then run the analyses in dependency order:

```powershell
python modeling\04_train_weather_impact_models.py
python modeling\05_compare_weather_models_features.py
python modeling\06_compare_weather_models_extended.py
python modeling\07_analyze_gb_feature_b.py
```

Scripts 04 and 05 are independent once the prepared dataset exists. Script 06
produces the extended predictions consumed by script 07. By default, scripts
that create an existing output file/directory stop rather than overwrite it;
move the old output aside or use the script's `--overwrite` option where
available before rerunning.

The `Y` column created by `prepare_modeling_dataset_v2.py` is a historical
October-mean comparison retained to reproduce the existing filtered dataset.
Scripts 04–07 do not use it as their model target: they calculate hazard
response `Y1` from seasonal normal days in the Train split only.

## Check the preprocessing step

The preprocessing unit test uses synthetic data and does not need the source
datasets:

```powershell
python -m unittest discover -s tests
```

Scripts 01–07 may take substantially longer than the dataset preparation step,
especially the repeated permutation-importance analyses.
