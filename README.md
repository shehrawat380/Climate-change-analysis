# Climate Change Modeling (Advanced)

This project builds an end‑to‑end pipeline to analyze public discourse around climate change (NASA Facebook comments) **and** model numeric climate indicators for projections.

## Contents
- `data/` — put your CSVs here. For the NASA comments dataset, download from the link in the brief. For climate indicators, use NOAA/NASA/IPCC sources.
- `src/` — reusable Python modules (preprocessing, features, modeling, viz).
- `notebooks/` — step-by-step notebooks: EDA, modeling, and projections.
- `app.py` — optional Streamlit app to upload a CSV and run EDA + a baseline model.
- `artifacts/` — trained models, scalers, and reports.
- `reports/` — generated figures/plots.

## Quickstart
```bash
pip install -r requirements.txt
# optional: streamlit run app.py
```

### Expected Columns (NLP dataset)
- `Date`, `LikesCount`, `ProfileName`, `CommentsCount`, `Text`

### Expected Columns (numeric climate indicators)
- e.g., `temperature_anomaly` (target), `co2_ppm`, `precip_mm`, `sea_level_mm`, etc.

> You can switch targets in the notebooks and in `app.py` via a dropdown.

## Repro Steps
1. Open `notebooks/01_data_exploration.ipynb` and point `DATA_PATH` to your CSV.
2. Run EDA cells. Save curated dataset as `data/clean_climate.csv`.
3. Move to `notebooks/02_modeling.ipynb` to train models. Artifacts saved to `artifacts/`.
4. Use `notebooks/03_projections.ipynb` to run future scenario projections.
5. (Optional) `streamlit run app.py` for a simple UI.
