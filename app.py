
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.live_data import (
    fetch_openaq_latest,
    fetch_openaq_historical,
)
from src.analysis import summarize_measurements, estimate_sources
from src.model import train_and_predict


# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

st.set_page_config(
    page_title="GreenAI | Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    color: #A0A0A0;
    font-size: 16px;
    margin-bottom: 22px;
}
.section-title {
    font-size: 25px;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🌿 GreenAI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Air Quality Intelligence | Live Monitoring | '
    'Historical Analysis | Machine Learning'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "GreenAI retrieves published measurements from external "
    "air-quality monitoring stations. It does not directly "
    "measure air quality."
)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "live_df": None,
    "live_meta": None,
    "live_city_key": None,
    "history_df": None,
    "history_meta": None,
    "history_city_key": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    city = st.text_input(
        "Enter city",
        value="Chennai",
        placeholder="Example: Hyderabad",
    ).strip()

    country = st.text_input(
        "Country code",
        value="IN",
        max_chars=2,
    ).strip().upper()

    api_key = st.text_input(
        "OpenAQ API key",
        value=os.getenv("OPENAQ_API_KEY", ""),
        type="password",
    ).strip()

    st.caption("OpenAQ provides monitoring observations.")

    st.divider()

    st.subheader("Historical Dataset")

    uploaded = st.file_uploader(
        "Optional: Upload historical CSV",
        type=["csv"],
    )

    st.caption(
        "CSV requires timestamp/date and PM2.5 columns."
    )

    st.divider()

    st.caption("Geocoding: OpenStreetMap Nominatim")
    st.caption("Map data © OpenStreetMap contributors")


city_key = f"{city.lower()}|{country}"

# Clear old live results when the selected city changes.
if st.session_state["live_city_key"] != city_key:
    st.session_state["live_df"] = None
    st.session_state["live_meta"] = None

# Clear old historical results when the selected city changes.
if st.session_state["history_city_key"] != city_key:
    st.session_state["history_df"] = None
    st.session_state["history_meta"] = None


# =====================================================
# NAVIGATION
# =====================================================

live_tab, ml_tab, about_tab = st.tabs([
    "📍 Live Monitoring",
    "🤖 Historical ML",
    "ℹ️ Methodology",
])


# =====================================================
# LIVE MONITORING
# =====================================================

with live_tab:

    st.markdown(
        '<div class="section-title">'
        'Live Air Quality Monitoring'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(f"Selected location: **{city}, {country}**")

    if st.button(
        "🔄 Fetch Latest Data",
        type="primary",
        key="fetch_live",
    ):

        if not api_key:
            st.warning("Please enter your OpenAQ API key.")

        elif not city:
            st.warning("Please enter a city name.")

        elif len(country) != 2:
            st.warning("Enter a valid two-letter country code.")

        else:

            try:

                with st.spinner(
                    f"Finding monitoring stations near {city}..."
                ):

                    df, meta = fetch_openaq_latest(
                        city,
                        country,
                        api_key,
                    )

                if df.empty:
                    raise ValueError(
                        "No monitoring observations were returned."
                    )

                st.session_state["live_df"] = df
                st.session_state["live_meta"] = meta
                st.session_state["live_city_key"] = city_key

                st.success(
                    f"Successfully retrieved observations for {city}!"
                )

            except Exception as error:

                st.session_state["live_df"] = None
                st.session_state["live_meta"] = None

                st.error(f"Data retrieval failed: {error}")

    df = st.session_state["live_df"]
    meta = st.session_state["live_meta"] or {}

    if isinstance(df, pd.DataFrame) and not df.empty:

        st.caption(
            f"Source: {meta.get('source', 'OpenAQ API v3')} | "
            f"Retrieved: {meta.get('retrieved_at', 'Not provided')}"
        )

        st.success(
            f"Resolved location: {meta.get('resolved_location', city)}"
        )

        latitude = meta.get("latitude")
        longitude = meta.get("longitude")

        if latitude is not None and longitude is not None:
            coordinate_text = (
                f"{float(latitude):.4f}, {float(longitude):.4f}"
            )
        else:
            coordinate_text = "Not provided"

        st.caption(
            f"Coordinates: {coordinate_text} | "
            f"Radius: {meta.get('search_radius_km', 25)} km | "
            f"Locations found: {meta.get('locations_found', 'N/A')} | "
            f"Stations with readings: {meta.get('stations', 'N/A')}"
        )

        chart_df = df.copy()

        required = {
            "value",
            "timestamp",
            "parameter",
            "unit",
            "station",
        }

        missing = required.difference(chart_df.columns)

        if missing:

            st.error(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

        else:

            chart_df["value"] = pd.to_numeric(
                chart_df["value"],
                errors="coerce",
            )

            chart_df["timestamp"] = pd.to_datetime(
                chart_df["timestamp"],
                errors="coerce",
                utc=True,
            )

            chart_df["parameter"] = (
                chart_df["parameter"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
            )

            chart_df["unit"] = (
                chart_df["unit"]
                .fillna("Unit not reported")
                .astype(str)
                .str.strip()
            )

            chart_df.loc[
                chart_df["unit"] == "",
                "unit",
            ] = "Unit not reported"

            chart_df = chart_df.dropna(
                subset=["value", "timestamp"]
            )

            if chart_df.empty:

                st.warning(
                    "No valid timestamped measurements were returned."
                )

            else:

                # -------------------------------------
                # OBSERVATION FRESHNESS
                # -------------------------------------

                st.divider()
                st.subheader("Observation Freshness")

                latest_time = chart_df["timestamp"].max()

                age_hours = (
                    pd.Timestamp.now(tz="UTC") - latest_time
                ).total_seconds() / 3600

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Latest measurement",
                    latest_time.strftime("%d %b %Y"),
                )

                c2.metric(
                    "Measurement age",
                    f"{max(0, age_hours / 24):.1f} days",
                )

                c3.metric(
                    "Valid observations",
                    f"{len(chart_df):,}",
                )

                if age_hours <= 24:

                    st.success(
                        "Newest measurement is less than 24 hours old."
                    )

                elif age_hours <= 168:

                    st.warning(
                        "Newest measurement is between one and seven days old."
                    )

                else:

                    st.warning(
                        "Newest measurement is older than seven days. "
                        "It should not be interpreted as current air quality."
                    )

                # -------------------------------------
                # POLLUTANT SUMMARY
                # -------------------------------------

                st.divider()
                st.subheader("Pollutant Summary by Unit")

                summary_table = (
                    chart_df
                    .groupby(
                        ["parameter", "unit"],
                        as_index=False,
                    )
                    .agg(
                        readings=("value", "count"),
                        average=("value", "mean"),
                        latest=("value", "last"),
                    )
                )

                summary_table["parameter"] = (
                    summary_table["parameter"].str.upper()
                )

                summary_table = summary_table.rename(
                    columns={
                        "parameter": "Pollutant",
                        "unit": "Unit",
                        "readings": "Readings",
                        "average": "Average",
                        "latest": "Latest value",
                    }
                )

                st.dataframe(
                    summary_table.round(2),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Different measurement units are not combined."
                )

                # -------------------------------------
                # POLLUTANT CHART
                # -------------------------------------

                st.divider()

                st.subheader(
                    "Pollutant Observations by Monitoring Station"
                )

                pollutants = sorted(
                    chart_df["parameter"].unique()
                )

                selected_pollutant = st.selectbox(
                    "Select pollutant",
                    pollutants,
                    format_func=lambda value: value.upper(),
                    key="live_pollutant",
                )

                pollutant_df = chart_df[
                    chart_df["parameter"] == selected_pollutant
                ]

                units = sorted(
                    pollutant_df["unit"].unique()
                )

                selected_unit = st.selectbox(
                    "Select measurement unit",
                    units,
                    key="live_unit",
                )

                selected_df = pollutant_df[
                    pollutant_df["unit"] == selected_unit
                ]

                if not selected_df.empty:

                    fig = px.scatter(
                        selected_df,
                        x="timestamp",
                        y="value",
                        color="station",
                        hover_data=[
                            "station",
                            "parameter",
                            "unit",
                            "timestamp",
                        ],
                        title=(
                            f"{selected_pollutant.upper()} "
                            f"Measurements ({selected_unit})"
                        ),
                        labels={
                            "timestamp": "Measurement time",
                            "value": f"Concentration ({selected_unit})",
                            "station": "Monitoring station",
                        },
                    )

                    fig.update_layout(
                        height=520,
                        template="plotly_dark",
                        hovermode="closest",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                # -------------------------------------
                # RAW DATA
                # -------------------------------------

                with st.expander(
                    "📋 View Raw Monitoring Observations"
                ):

                    st.dataframe(
                        chart_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download CSV",
                        data=chart_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="greenai_observations.csv",
                        mime="text/csv",
                    )

                # -------------------------------------
                # SOURCE INDICATORS
                # -------------------------------------

                st.divider()

                st.subheader(
                    "Possible Pollution Source Indicators"
                )

                try:

                    analysis_summary = summarize_measurements(df)
                    flags = estimate_sources(analysis_summary)

                    if flags:

                        for flag in flags:
                            st.write("•", flag)

                    else:

                        st.info(
                            "Insufficient evidence for useful source indicators."
                        )

                except Exception as error:

                    st.info(
                        f"Source indicators unavailable: {error}"
                    )

                st.warning(
                    "Source indicators are screening-level hypotheses, "
                    "not confirmed pollution sources."
                )

    else:

        st.info(
            "Select a city and click Fetch Latest Data."
        )


# =====================================================
# HISTORICAL MACHINE LEARNING
# =====================================================

with ml_tab:

    st.markdown(
        '<div class="section-title">'
        'Historical Air Quality Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Historical PM2.5 analysis using Random Forest regression."
    )

    st.caption(
        f"Selected city: {city}, {country}"
    )

    history_days = st.selectbox(
        "Historical period",
        options=[90, 180, 365],
        index=2,
        format_func=lambda value: f"Last {value} days",
        key="history_period",
    )

    history_source = st.radio(
        "Historical data source",
        [
            "Fetch automatically from OpenAQ",
            "Upload my own CSV",
        ],
        horizontal=True,
        key="history_source",
    )

    raw = None
    history_meta = None
    source_label = None

    # ---------------------------------------------
    # AUTOMATIC OPENAQ HISTORY
    # ---------------------------------------------

    if history_source == "Fetch automatically from OpenAQ":

        if st.button(
            "📚 Fetch Historical Data & Run ML",
            type="primary",
            key="fetch_history",
        ):

            if not api_key:

                st.warning(
                    "Enter your OpenAQ API key in the sidebar."
                )

            elif not city:

                st.warning("Enter a city name.")

            else:

                try:

                    with st.spinner(
                        f"Fetching historical PM2.5 data for {city}..."
                    ):

                        history_df, history_meta = (
                            fetch_openaq_historical(
                                city,
                                country,
                                api_key,
                                days=history_days,
                            )
                        )

                    st.session_state["history_df"] = history_df
                    st.session_state["history_meta"] = history_meta
                    st.session_state["history_city_key"] = city_key

                    st.success(
                        f"Historical data retrieved for {city}."
                    )

                except Exception as error:

                    st.session_state["history_df"] = None
                    st.session_state["history_meta"] = None

                    st.error(
                        f"Historical data retrieval failed: {error}"
                    )

        if st.session_state["history_city_key"] == city_key:

            raw = st.session_state["history_df"]
            history_meta = st.session_state["history_meta"]
            source_label = "OpenAQ historical daily data"

        else:

            st.info(
                "Click Fetch Historical Data & Run ML to load data for this city."
            )

    # ---------------------------------------------
    # OPTIONAL CSV UPLOAD
    # ---------------------------------------------

    else:

        if uploaded is not None:

            try:

                raw = pd.read_csv(uploaded)
                source_label = "User-uploaded CSV"

            except Exception as error:

                st.error(f"Could not read CSV: {error}")

        else:

            st.info(
                "Upload a CSV containing timestamp/date and PM2.5 columns."
            )

    # ---------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------

    if raw is not None:

        try:

            result = train_and_predict(raw)

            st.success(
                f"Historical dataset processed successfully — {source_label}."
            )

            if history_meta:

                st.caption(
                    f"Resolved city: {history_meta.get('resolved_location', city)} | "
                    f"Period: {history_meta.get('period_days')} days | "
                    f"Sensors with data: {history_meta.get('sensors_with_data')} | "
                    f"Daily observations: {history_meta.get('daily_rows')} | "
                    f"Unit: {history_meta.get('unit') or 'not reported'}"
                )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Holdout MAE",
                f"{result['mae']:.2f} µg/m³",
            )

            c2.metric(
                "Next-step RF estimate",
                f"{result['forecast']:.2f} µg/m³",
            )

            c3.metric(
                "Test observations",
                f"{result['test_rows']:,}",
            )

            st.caption(
                f"Training rows: {result['train_rows']:,} | "
                f"Testing rows: {result['test_rows']:,}"
            )

            st.divider()

            st.subheader("Actual vs Predicted PM2.5")

            plot_df = result["plot_df"].copy()

            fig = px.line(
                plot_df,
                x="timestamp",
                y=["actual", "predicted"],
                labels={
                    "timestamp": "Observation time",
                    "value": "PM2.5 (µg/m³)",
                    "variable": "Measurement",
                },
                title="Random Forest Holdout Evaluation",
            )

            fig.update_layout(
                height=500,
                template="plotly_dark",
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.subheader("Model Explanation")

            st.write(
                f"""
                **Algorithm:** Random Forest Regressor

                **Data source:** {source_label}

                **Holdout MAE:** {result['mae']:.2f} µg/m³

                **Next-step estimate:** {result['forecast']:.2f} µg/m³

                **Training observations:** {result['train_rows']}

                **Test observations:** {result['test_rows']}
                """
            )

            st.info(
                "The next-step estimate is experimental. It depends on "
                "historical data quality and sampling intervals. It is "
                "not an official AQI or public-health forecast."
            )

            with st.expander("View historical dataset"):

                st.dataframe(
                    raw.head(500),
                    use_container_width=True,
                )

                st.download_button(
                    "Download historical data as CSV",
                    data=raw.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"{city.lower()}_historical_pm25.csv",
                    mime="text/csv",
                )

        except Exception as error:

            st.error(f"Machine learning error: {error}")


# =====================================================
# METHODOLOGY
# =====================================================

with about_tab:

    st.markdown(
        '<div class="section-title">'
        'System Methodology'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("GreenAI Processing Pipeline")

    st.markdown("""
    **1. City Geocoding**

    Convert the entered city name into geographic coordinates using
    OpenStreetMap Nominatim.

    **2. Live Data Acquisition**

    Retrieve OpenAQ monitoring locations within 25 km of the selected city.

    **3. Data Processing**

    Match sensor readings to pollutant parameters and preserve units
    and timestamps.

    **4. Air Quality Analysis**

    Calculate pollutant summaries and possible source indicators.

    **5. Historical Data Retrieval**

    Retrieve daily PM2.5 measurements from OpenAQ, or accept an optional
    user-uploaded CSV.

    **6. Machine Learning**

    Train a Random Forest regression model using historical PM2.5 data.

    **7. Model Evaluation**

    Evaluate predictions using chronological holdout testing and
    Mean Absolute Error.
    """)

    st.divider()

    st.subheader("Technology Stack")

    c1, c2, c3 = st.columns(3)

    c1.markdown("""
    - Python
    - Streamlit
    - Pandas
    """)

    c2.markdown("""
    - Scikit-learn
    - Random Forest
    - Plotly
    """)

    c3.markdown("""
    - OpenAQ API v3
    - OpenStreetMap
    - Requests
    """)

    st.divider()

    st.subheader("System Limitations")

    st.warning("""
    - OpenAQ monitoring coverage varies by city.
    - Search covers a 25 km radius around the selected city centre.
    - API retrieval time differs from measurement time.
    - Measurements may use different units.
    - Historical availability depends on station coverage.
    - Model performance depends on data quality.
    - Source indicators are not validated source apportionment.
    - Official AQI is not calculated in this version.
    """)

    st.markdown("""
    **References**

    - [OpenAQ API Documentation](https://docs.openaq.org/)
    - [OpenStreetMap Nominatim](https://nominatim.org/)
    - [Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)
    - [CPCB Air Quality Portal](https://airquality.cpcb.gov.in/AQI_India/)
    """)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "GreenAI | Air Quality Intelligence | Experimental Academic Project"
)
