import streamlit as st
from data.edgar import get_quarterly_revenue
from data.yfinance_data import (
    get_analyst_consensus, get_company_info, get_stock_price_history,
    get_news, get_52w_signal, get_beat_rate
)
from data.fred import get_macro_context
from models.runner import run_all_models
from pipeline.stage1_data import validate_data
from pipeline.stage3_backtest import run_all_backtests
from pipeline.stage4_ensemble import run_backtest_weighted_ensemble
from pipeline.stage5_scenarios import build_scenarios
from pipeline.stage6_blend import blend_forecast
from pipeline.stage7_signals import score_news_sentiment, build_signal_summary
from pipeline.stage8_validate import run_validation
from chatbot.analyst import get_auto_brief, chat_with_analyst
from groq import Groq
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd
import os

load_dotenv()

# model registry used for walk-forward backtest (maps fn -> name)
from models.individual.seasonal_naive import run_seasonal_naive
from models.individual.naive_drift import run_naive_drift
from models.individual.holts_linear import run_holts_linear
from models.individual.ets import run_ets
from models.individual.sarima import run_sarima
from models.individual.tbats_model import run_tbats
from models.individual.prophet_model import run_prophet
from models.individual.autoarima import run_autoarima
from models.individual.theta import run_theta
from models.individual.linear_regression import run_linear_regression

MODEL_REGISTRY = [
    (run_seasonal_naive, "Seasonal Naive"),
    (run_naive_drift,    "Naive Drift"),
    (run_holts_linear,   "Holts Linear"),
    (run_ets,            "ETS"),
    (run_sarima,         "SARIMA"),
    (run_tbats,          "TBATS"),
    (run_prophet,        "Prophet"),
    (run_autoarima,      "AutoARIMA"),
    (run_theta,          "Theta"),
    (run_linear_regression, "Linear Regression"),
]

st.set_page_config(
    page_title="Financial Forecaster",
    page_icon="📈",
    layout="wide"
)

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("auto_brief", None),
    ("forecast_ready", False),
    ("forecast_data", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: AI Analyst ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 AI Analyst")
    st.caption("Powered by Llama 3.3 via Groq")
    st.divider()

    if not st.session_state.forecast_ready:
        st.caption("Run a forecast to activate the AI analyst.")
    else:
        if st.session_state.auto_brief:
            st.markdown("**📋 Analysis Brief**")
            brief_display = st.session_state.auto_brief
            if isinstance(brief_display, dict):
                st.markdown(f"**{brief_display.get('headline', '')}**")
                st.caption(brief_display.get("forecast", ""))
                if brief_display.get("risk"):
                    st.error(f"⚠️ {brief_display['risk']}")
            else:
                st.info(brief_display)
            st.divider()

        st.markdown("**💬 Ask a follow-up**")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("e.g. What is the biggest risk?")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            fd = st.session_state.forecast_data
            response = chat_with_analyst(
                user_message=user_input,
                company_info=fd["company"],
                ensemble_result=fd["ensemble"],
                macro_context=fd["macro"],
                ticker=fd["ticker"],
                chat_history=st.session_state.chat_history[:-1],
                scenarios=fd.get("scenarios"),
                signals=fd.get("signals"),
                validation=fd.get("validation")
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# ── Main title ───────────────────────────────────────────────────────────────
st.title("📈 Financial Forecaster")
st.caption("13-step ensemble pipeline · 10 models · walk-forward backtest · bear/base/bull scenarios")

tab1, tab2 = st.tabs(["📊 Earnings Forecast", "📉 Stock Trend"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: EARNINGS FORECAST
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input(
            "Enter ticker symbol",
            value="DAL",
            placeholder="e.g. DAL, GS, AAPL"
        ).upper()
    with col2:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    exclude_covid = st.checkbox(
        "Exclude COVID period (Mar 2020 – Jun 2021)",
        value=True,
        help="Recommended — prevents COVID crash from distorting trend models"
    )

    if run_button and ticker:
        # reset state for fresh run
        st.session_state.chat_history = []
        st.session_state.auto_brief = None
        st.session_state.forecast_ready = False

        # ── STAGE 0: Fetch data ────────────────────────────────────────────
        progress = st.progress(0, text="Fetching revenue data...")
        with st.spinner(f"Fetching data for {ticker}..."):
            df = get_quarterly_revenue(ticker)
            company = get_company_info(ticker)
            macro = get_macro_context()
            consensus_df = get_analyst_consensus(ticker)

        if df is None or df.empty:
            st.error(f"Could not find data for {ticker}. Please check the symbol.")
            st.stop()

        # optionally exclude COVID quarters
        if exclude_covid:
            df = df[~((df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30"))].reset_index(drop=True)

        # extract single consensus number (current quarter estimate)
        consensus = None
        if consensus_df is not None and not consensus_df.empty:
            try:
                consensus = float(
                    consensus_df[consensus_df["period"] == "0q"]["avg"].values[0]
                )
            except Exception:
                consensus = None

        # ── STAGE 1: Data validation ───────────────────────────────────────
        progress.progress(10, text="Step 1: Validating data quality...")
        validation_data = validate_data(df)
        if validation_data["errors"]:
            for err in validation_data["errors"]:
                st.warning(f"⚠️ {err}")
        if validation_data["warnings"]:
            for warn in validation_data["warnings"]:
                # suppress the data-gap warning when exclude_covid is active —
                # the gap is intentional, not a data quality problem
                if exclude_covid and "gap" in warn.lower():
                    continue
                st.info(f"ℹ️ {warn}")

        # ── STAGE 2: Run 10 models ─────────────────────────────────────────
        progress.progress(20, text="Step 2: Running 10 forecast models...")
        with st.spinner("Running 10 forecast models..."):
            results = run_all_models(df, analyst_consensus=consensus)

        if results is None:
            st.error("Could not run models. Not enough data.")
            st.stop()

        # ── STAGE 3: Walk-forward backtest ────────────────────────────────
        progress.progress(40, text="Step 3: Running 12-quarter walk-forward backtest...")
        with st.spinner("Backtesting models (12-quarter walk-forward)..."):
            backtest_results = run_all_backtests(df, MODEL_REGISTRY)

        # ── STAGE 4: Backtest-weighted ensemble ───────────────────────────
        progress.progress(55, text="Step 4: Building backtest-weighted ensemble...")
        ensemble = run_backtest_weighted_ensemble(
            results["individual_results"], backtest_results, analyst_consensus=consensus
        )

        # ── STAGE 5: Scenarios ────────────────────────────────────────────
        progress.progress(65, text="Step 5: Building Bear/Base/Bull scenarios...")
        scenarios = build_scenarios(df)

        # ── STAGE 6: Blend ────────────────────────────────────────────────
        progress.progress(70, text="Step 6: Blending ensemble with scenario...")
        blend = blend_forecast(ensemble["ensemble_forecast"], scenarios)

        # ── STAGES 7-9: Signals ───────────────────────────────────────────
        progress.progress(75, text="Steps 7–9: Gathering signals (news, price, beat rate)...")
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with st.spinner("Gathering market signals..."):
            news_items = get_news(ticker)
            price_signal = get_52w_signal(ticker)
            beat_rate = get_beat_rate(ticker)
            news_sentiment = score_news_sentiment(news_items, ticker, groq_client)
        signals = build_signal_summary(price_signal, news_sentiment, beat_rate)

        # ── STAGES 11-13: Validation ──────────────────────────────────────
        progress.progress(88, text="Steps 11–13: Running validation gates...")
        validation = run_validation(
            blended_forecast=blend["blended_forecast"],
            scenarios=scenarios,
            analyst_consensus=consensus,
            ensemble_forecast=ensemble["ensemble_forecast"]
        )

        progress.progress(95, text="Generating AI brief...")
        with st.spinner("Generating AI analyst brief..."):
            brief = get_auto_brief(company, ensemble, macro, ticker,
                                   scenarios=scenarios, signals=signals, validation=validation)

        progress.progress(100, text="Done!")
        progress.empty()

        # ════════════════════════════════════════════════════════════════════
        # DISPLAY RESULTS
        # ════════════════════════════════════════════════════════════════════

        # Company header
        st.subheader(f"{company['name']} ({ticker})")
        st.caption(f"{company['sector']} · {company['industry']}")
        st.divider()

        # ── Top metric cards ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        blended_val = blend["blended_forecast"]
        with m1:
            st.metric("Blended Forecast", f"${blended_val/1e9:.2f}B",
                      help="60% backtest-weighted ensemble + 40% base scenario")
        with m2:
            if consensus:
                gap = (blended_val - consensus) / consensus * 100
                st.metric("Analyst Consensus", f"${consensus/1e9:.2f}B",
                          delta=f"{gap:+.1f}% model vs consensus",
                          delta_color="normal" if gap > 0 else "inverse")
            else:
                st.metric("Analyst Consensus", "N/A")
        with m3:
            conf_color = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
            conf_icon = conf_color.get(validation["confidence_level"], "🟡")
            st.metric(
                "Confidence",
                f"{conf_icon} {validation['confidence_pct']}%",
                delta=validation["confidence_level"].capitalize(),
                delta_color="off"
            )
        with m4:
            eligible_models = sum(1 for bt in backtest_results if bt["eligible"])
            all_successful = sum(1 for r in results["individual_results"] if r["status"] == "success")
            st.metric("Models in Ensemble", f"{all_successful}/10",
                      delta=f"{eligible_models} passed MAPE filter",
                      delta_color="off",
                      help="All successful models contribute. Models with MAPE ≤20% get higher weight.")

        # ── Validation gate banners ───────────────────────────────────────
        if validation["passes_all"]:
            st.success("✅ All 3 validation gates passed (Steps 11–13) — forecast is locked.")
        else:
            flags = validation["flags"]
            flag_count = len(flags)

            # single always-visible status line — not alarming, just informative
            st.warning(
                f"⚠️ {flag_count} validation flag{'s' if flag_count > 1 else ''} detected "
                f"— confidence: {validation['confidence_pct']}% "
                f"({validation['confidence_level']}). See details below."
            )

            with st.expander("🔍 Validation flag details", expanded=False):
                for flag in flags:
                    st.markdown(f"- {flag}")

                # special explanatory note when ensemble falls below bear case
                ens_val = ensemble.get("ensemble_forecast")
                if ens_val is not None and ens_val < scenarios["bear"]:
                    st.markdown("---")
                    st.markdown(
                        "**ℹ️ Why is the ensemble below the bear scenario?**  \n"
                        "The ensemble models are trained on the full revenue history, which often includes "
                        "periods of slower growth. When a company has had a strong recent run, the historical "
                        "average anchors the ensemble conservatively, while the same-quarter YoY scenarios "
                        "reflect only recent momentum. This is a model conservatism signal — it means the "
                        "statistical models are more cautious than the recent trend implies. "
                        "Use the blended forecast (60% ensemble + 40% base scenario) as your working number."
                    )

        st.divider()

        # ── Bear / Base / Bull scenario cards ────────────────────────────
        st.subheader("📐 Scenarios (Bear / Base / Bull)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("🐻 Bear", f"${scenarios['bear']/1e9:.2f}B",
                      delta=f"{scenarios['percentiles']['p25']:+.1f}% YoY (p25)",
                      delta_color="inverse")
        with sc2:
            st.metric("📊 Base", f"${scenarios['base']/1e9:.2f}B",
                      delta=f"{scenarios['percentiles']['p50']:+.1f}% YoY (median)",
                      delta_color="off")
        with sc3:
            st.metric("🐂 Bull", f"${scenarios['bull']/1e9:.2f}B",
                      delta=f"{scenarios['percentiles']['p75']:+.1f}% YoY (p75)",
                      delta_color="normal")
        with sc4:
            blend_icon = {"high": "✅", "moderate": "ℹ️", "low": "⚠️"}
            st.metric(
                "Blend Convergence",
                blend_icon.get(blend["convergence"], "ℹ️") + " " + blend["convergence"].capitalize(),
                delta=f"Ensemble vs base diff: {blend['diff_pct']}%",
                delta_color="off"
            )

        # ── Signal row ────────────────────────────────────────────────────
        st.subheader("📡 Market Signals")
        sg1, sg2, sg3, sg4 = st.columns(4)

        sentiment_icon = {"POSITIVE": "🟢 Positive", "NEUTRAL": "🟡 Neutral", "NEGATIVE": "🔴 Negative"}
        price_icon = {"HIGH_BAR": "📈 Elevated bar", "NEUTRAL": "➡️ Neutral", "LOW_BAR": "📉 Easy bar"}
        bias_icon = {"POSITIVE": "🟢 Positive", "NEUTRAL": "🟡 Neutral", "NEGATIVE": "🔴 Negative"}

        with sg1:
            ns = signals["news_sentiment"]
            st.metric(
                "News Sentiment",
                sentiment_icon.get(ns["sentiment"], "🟡 Neutral"),
                delta=f"{ns['items_scored']} headlines scored",
                delta_color="off",
                help=ns.get("reason", "")
            )
        with sg2:
            ps = signals["price_signal"]
            st.metric(
                "Price Signal (52W)",
                price_icon.get(ps["signal"], "➡️ Neutral"),
                delta=ps.get("note", "")[:50],
                delta_color="off"
            )
        with sg3:
            br = signals["beat_rate"]
            if br["beat_rate"] is not None:
                beat_icon = "🟢" if br["beat_rate"] > 60 else ("🔴" if br["beat_rate"] < 40 else "🟡")
                st.metric(
                    "Beat Rate",
                    f"{beat_icon} {br['beat_rate']}%",
                    delta=f"{br['beats']}/{br['quarters_checked']} quarters beat",
                    delta_color="off"
                )
            else:
                st.metric("Beat Rate", "N/A")
        with sg4:
            st.metric(
                "Overall Signal Bias",
                bias_icon.get(signals["overall_bias"], "🟡 Neutral"),
                delta_color="off"
            )

        st.divider()

        # ── Main chart ────────────────────────────────────────────────────
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["date"],
            y=df["revenue"] / 1e9,
            name="Historical Revenue",
            marker_color="steelblue",
            opacity=0.75,
            hovertemplate="<b>%{x|%Y Q%q}</b><br>Revenue: $%{y:.2f}B<extra></extra>"
        ))

        if not exclude_covid:
            fig.add_vrect(
                x0="2020-03-01", x1="2021-06-30",
                fillcolor="red", opacity=0.08, layer="below", line_width=0,
                annotation_text="COVID-19", annotation_position="top left",
                annotation_font_size=11, annotation_font_color="red"
            )

        next_date = df["date"].iloc[-1] + pd.DateOffset(months=3)
        next_label = f"Q{pd.Timestamp(next_date).quarter} {pd.Timestamp(next_date).year}"

        colors = {
            "Seasonal Naive":    "#FFA500",
            "Naive Drift":       "#FF8C00",
            "Holts Linear":      "#FFD700",
            "Linear Regression": "#00CC96",
            "ETS":               "#EF553B",
            "SARIMA":            "#AB63FA",
            "TBATS":             "#B347EA",
            "Prophet":           "#FF6692",
            "AutoARIMA":         "#19D3F3",
            "AutoARIMA(log)":    "#00B5F7",
            "Theta":             "#636EFA"
        }
        successful = [r for r in results["individual_results"] if r["status"] == "success"]
        for r in successful:
            fig.add_trace(go.Scatter(
                x=[next_date], y=[r["forecast"] / 1e9],
                mode="markers+text", name=r["model"],
                text=[r["model"]], textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=9, color=colors.get(r["model"], "gray")),
                hovertemplate=(
                    f"<b>{r['model']}</b><br>"
                    f"Forecast: ${r['forecast']/1e9:.2f}B<br>"
                    f"RMSE: ${r.get('rmse', 0)/1e6:.0f}M<extra></extra>"
                )
            ))

        # Bear / Base / Bull scenario lines
        for label, val, color, dash in [
            ("Bear", scenarios["bear"] / 1e9, "#ef4444", "dot"),
            ("Base", scenarios["base"] / 1e9, "#94a3b8", "dash"),
            ("Bull", scenarios["bull"] / 1e9, "#22c55e", "dot"),
        ]:
            fig.add_shape(
                type="line",
                x0=next_date - pd.DateOffset(months=1),
                x1=next_date + pd.DateOffset(months=2),
                y0=val, y1=val,
                line=dict(color=color, width=1.5, dash=dash)
            )
            fig.add_annotation(
                x=next_date + pd.DateOffset(months=2),
                y=val,
                text=f"{label} ${val:.2f}B",
                showarrow=False,
                font=dict(size=9, color=color),
                xanchor="left"
            )

        # blended forecast star
        blended_b = blended_val / 1e9
        fig.add_trace(go.Scatter(
            x=[next_date], y=[blended_b],
            mode="markers+text", name="⭐ Blended Forecast",
            text=[f"${blended_b:.2f}B"],
            textposition="bottom center",
            textfont=dict(size=11, color="gold"),
            marker=dict(size=18, color="gold", symbol="star"),
            hovertemplate=f"<b>Blended Forecast</b><br>${blended_b:.2f}B<extra></extra>"
        ))

        if consensus:
            fig.add_hline(
                y=consensus / 1e9,
                line_dash="dash", line_color="white", opacity=0.35,
                annotation_text=f"Consensus ${consensus/1e9:.2f}B",
                annotation_position="bottom right",
                annotation_font_size=10
            )

        fig.update_layout(
            title=dict(
                text=f"{ticker} Quarterly Revenue + Next Quarter Forecast ({next_label})",
                font=dict(size=16)
            ),
            xaxis=dict(title="Quarter", showgrid=False),
            yaxis=dict(title="Revenue ($ Billions)", showgrid=True,
                       gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=520,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Model breakdown table ─────────────────────────────────────────
        st.subheader("Model Breakdown")
        bt_map = {bt["model"]: bt for bt in backtest_results}
        table_data = []
        for r in results["individual_results"]:
            bt = bt_map.get(r["model"], {})
            row = {
                "Model": r["model"],
                "Forecast": f"${r['forecast']/1e9:.2f}B" if r["status"] == "success" else "—",
                "RMSE": f"${r.get('rmse', 0)/1e6:.0f}M" if r.get("rmse") else "—",
                "Backtest MAPE": f"{bt['mape']:.1f}%" if bt.get("mape") else "—",
                "Eligible": "✅" if bt.get("eligible") else "❌",
                "Weight": f"{ensemble['model_weights'].get(r['model'], 0):.1f}%",
                "Status": "✅" if r["status"] == "success" else "⏭️ Skipped",
            }
            table_data.append(row)
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # ── Validation gates detail ───────────────────────────────────────
        st.subheader("🔒 Validation Gates (Steps 11–13)")
        vg1, vg2, vg3 = st.columns(3)
        check_icon = lambda p: "✅" if p else "❌"
        cc = validation["consensus_check"]
        eb = validation["error_band_check"]
        cv = validation["convergence_check"]
        with vg1:
            st.markdown(f"**{check_icon(cc['passes'])} Step 11: Consensus Anchor**")
            st.caption(cc["note"])
        with vg2:
            st.markdown(f"**{check_icon(eb['passes'])} Step 12: Error Band (3%)**")
            st.caption(eb["note"])
        with vg3:
            conv_icon = {"high": "✅", "moderate": "ℹ️", "low": "❌"}
            st.markdown(f"**{conv_icon.get(cv['level'], 'ℹ️')} Step 13: Convergence**")
            st.caption(cv["note"])

        # ── Macro environment ─────────────────────────────────────────────
        st.subheader("🌍 Macro Environment")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("CPI Inflation",  f"{macro.get('cpi_yoy_pct', 'N/A')}%", delta="YoY", delta_color="off")
        mc2.metric("Fed Funds Rate", f"{macro.get('fed_rate', 'N/A')}%")
        mc3.metric("Jobless Claims", f"{macro.get('jobless_claims', 0):,}")
        mc4.metric("GDP Growth",     f"{macro.get('gdp_growth', 'N/A')}%")

        # ── AI Analyst Brief ─────────────────────────────────────────────
        st.subheader("🤖 AI Analyst Brief")

        if isinstance(brief, dict):
            # Headline banner
            headline = brief.get("headline", "Analysis complete")
            conf_badge = {"high": "🟢 High confidence", "moderate": "🟡 Moderate confidence",
                          "low": "🔴 Low confidence"}.get(validation["confidence_level"], "")
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.06);border-left:3px solid #6366f1;
                border-radius:6px;padding:14px 18px;margin-bottom:14px;">
                <span style="font-size:15px;font-weight:600;">{headline}</span>
                &nbsp;&nbsp;<span style="font-size:12px;opacity:0.6;">{conf_badge}</span>
                </div>""",
                unsafe_allow_html=True
            )

            # 4-column detail cards
            bc1, bc2, bc3, bc4 = st.columns(4)
            card_sections = [
                (bc1, "📊 Forecast",   brief.get("forecast", "")),
                (bc2, "📐 Scenarios",  brief.get("scenarios", "")),
                (bc3, "📡 Signals",    brief.get("signals", "")),
                (bc4, "🌍 Macro",      brief.get("macro", "")),
            ]
            for col, label, text in card_sections:
                with col:
                    st.markdown(
                        f"""<div style="background:rgba(255,255,255,0.04);border:0.5px solid
                        rgba(255,255,255,0.1);border-radius:8px;padding:12px 14px;height:100%;">
                        <div style="font-size:11px;font-weight:600;opacity:0.6;margin-bottom:6px;">
                        {label}</div>
                        <div style="font-size:13px;line-height:1.5;">{text}</div></div>""",
                        unsafe_allow_html=True
                    )

            # Key risk — full width, warning style
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            risk_text = brief.get("risk", "")
            if risk_text:
                st.markdown(
                    f"""<div style="background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;
                    border-radius:6px;padding:12px 16px;">
                    <span style="font-size:11px;font-weight:600;color:#ef4444;">⚠️ KEY RISK</span>
                    &nbsp;&nbsp;<span style="font-size:13px;">{risk_text}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
        else:
            # fallback for plain string
            st.info(brief)

        st.session_state.auto_brief = brief   # dict with headline/forecast/scenarios/signals/macro/risk
        st.session_state.forecast_ready = True
        st.session_state.forecast_data = {
            "company": company,
            "ensemble": ensemble,
            "macro": macro,
            "ticker": ticker,
            "scenarios": scenarios,
            "signals": signals,
            "validation": validation,
            "blend": blend
        }

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: STOCK TREND (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Stock Trend Analyzer")
    st.caption("⚠️ This is a trend model only — not a price prediction.")

    ticker2 = st.text_input(
        "Enter ticker", value="DAL",
        placeholder="e.g. DAL, AAPL, GS", key="ticker2"
    ).upper()

    period_choice = st.selectbox(
        "History window", options=["1y", "2y", "5y"], index=1
    )

    run_trend = st.button("📉 Analyze Trend", type="primary")

    if run_trend and ticker2:
        with st.spinner(f"Fetching price data for {ticker2}..."):
            price_df = get_stock_price_history(ticker2, period=period_choice)

        if price_df is None or price_df.empty:
            st.error("Could not fetch price data.")
            st.stop()

        import numpy as np
        price_df["ma50"] = price_df["close_price"].rolling(50).mean()
        price_df["ma90"] = price_df["close_price"].rolling(90).mean()

        x = np.arange(len(price_df))
        slope, intercept = np.polyfit(x, price_df["close_price"], 1)
        future_days = 90
        future_x = np.arange(len(price_df), len(price_df) + future_days)
        future_dates = pd.date_range(
            start=price_df["date"].iloc[-1], periods=future_days + 1, freq="B"
        )[1:]
        trend_projection = slope * future_x + intercept

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["close_price"],
            name="Close Price", line=dict(color="steelblue", width=1.5)))
        fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ma50"],
            name="50-day MA", line=dict(color="orange", width=1.5, dash="dot")))
        fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ma90"],
            name="90-day MA", line=dict(color="green", width=1.5, dash="dash")))
        fig2.add_trace(go.Scatter(x=future_dates, y=trend_projection,
            name="Trend Projection", line=dict(color="gold", width=2, dash="dot")))
        fig2.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(trend_projection * 1.05) + list(trend_projection[::-1] * 0.95),
            fill="toself", fillcolor="rgba(255,215,0,0.1)",
            line=dict(color="rgba(255,215,0,0)"), name="±5% Band"
        ))

        fig2.update_layout(
            title=f"{ticker2} Price Trend + 90-Day Projection",
            xaxis_title="Date", yaxis_title="Price (USD)", height=500,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        current = price_df["close_price"].iloc[-1]
        high = price_df["close_price"].max()
        low = price_df["close_price"].min()
        change = (current - price_df["close_price"].iloc[0]) / price_df["close_price"].iloc[0] * 100
        s1.metric("Current Price", f"${current:.2f}")
        s2.metric("Period High",   f"${high:.2f}")
        s3.metric("Period Low",    f"${low:.2f}")
        s4.metric(f"{period_choice} Change", f"{change:+.1f}%",
                  delta_color="normal" if change > 0 else "inverse")
