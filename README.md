# GreenAI — Air Quality Intelligence
🌐 **Live Website:** https://greenai-shareef.streamlit.app
A software-only prototype combining:
- Latest published station observations from OpenAQ API v3.
- Transparent pollutant-pattern indicators for possible source contributors.
- Historical PM2.5 forecasting baseline trained on an uploaded CSV.
- Streamlit dashboard.

## Scope and honesty
This application does not measure air itself. It retrieves measurements collected by external monitoring stations. Coverage depends on the provider. Source indicators are hypotheses, not proof of sources. The forecast is an experimental baseline, not a regulatory or health forecast.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Get an OpenAQ API key at https://explore.openaq.org/register. Copy `.env.example` to `.env` and add:
```env
OPENAQ_API_KEY=your_actual_key
```
Then run:
```bash
streamlit run app.py
```
You may also paste the key into the sidebar. Never commit `.env` or publish your key.

## Historical CSV
CSV needs a timestamp/date column named `timestamp`, `datetime`, `date`, `time`, `Date`, or `Datetime`, plus PM2.5 named `pm25`, `pm2.5`, `PM2.5`, `PM25`, or `pm_2_5`. At least 40 valid rows are required. Use a documented real dataset and retain its source, licence, station, units and date range. The model uses chronological 80/20 evaluation, lag features and Random Forest; it reports holdout MAE.

## Source indicators
Current indicators are simple screening heuristics. Defensible source apportionment requires richer evidence such as chemical speciation, meteorology, emissions inventories and validated receptor/dispersion models. Do not describe these rules as a validated source classifier.

## AQI
AQI is not calculated here because official pollutant averaging periods and jurisdiction-specific breakpoints must be applied correctly.

## Demo sequence
1. Fetch live observations for a city with OpenAQ station coverage.
2. Show station, pollutant, unit and timestamp.
3. Upload documented historical PM2.5 data and train the baseline.
4. Explain holdout MAE and chronological evaluation.
5. Present source indicators with their limitations.

## References
- OpenAQ API: https://docs.openaq.org/
- API key guide: https://docs.openaq.org/using-the-api/api-key
- CPCB AQI portal: https://airquality.cpcb.gov.in/AQI_India/
