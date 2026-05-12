from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


def build_context(company_info, ensemble_result, macro_context, ticker,
                  scenarios=None, signals=None, validation=None):
    """
    Builds the full context string that gets injected into every Groq API call.
    We pass all pipeline results so the AI always has the complete picture —
    not just the ensemble, but also scenarios, signals, and validation outcome.
    """
    forecasts = ensemble_result["model_forecasts"]
    weights = ensemble_result["model_weights"]
    ensemble = ensemble_result["ensemble_forecast"]

    context = f"""You are a financial analyst assistant. You explain forecasts in clear business English.
Never use technical jargon. Always be concise and confident.
Never give direct investment advice. Always end with key risks.

COMPANY: {company_info['name']} ({ticker})
SECTOR: {company_info['sector']}
INDUSTRY: {company_info['industry']}

FORECAST SUMMARY:
Backtest-Weighted Ensemble: ${ensemble:,.0f}

INDIVIDUAL MODEL FORECASTS (with weights):
"""
    for model, forecast in forecasts.items():
        weight = weights.get(model, 0)
        context += f"  {model}: ${forecast:,.0f} (weight: {weight}%)\n"

    # add scenario context if available
    if scenarios:
        context += f"""
BEAR/BASE/BULL SCENARIOS (based on historical same-quarter YoY growth):
  Bear (p25 growth {scenarios['percentiles']['p25']:+.1f}%): ${scenarios['bear']:,.0f}
  Base (p50 growth {scenarios['percentiles']['p50']:+.1f}%): ${scenarios['base']:,.0f}
  Bull (p75 growth {scenarios['percentiles']['p75']:+.1f}%): ${scenarios['bull']:,.0f}
"""

    # add market signals if available
    if signals:
        ns = signals.get("news_sentiment", {})
        ps = signals.get("price_signal", {})
        br = signals.get("beat_rate", {})
        context += f"""
MARKET SIGNALS:
  News sentiment: {ns.get('sentiment', 'N/A')} — {ns.get('reason', '')}
  Price signal: {ps.get('signal', 'N/A')} — {ps.get('note', '')}
  Beat rate: {br.get('beat_rate', 'N/A')}% over last {br.get('quarters_checked', 0)} quarters
  Overall signal bias: {signals.get('overall_bias', 'N/A')}
"""

    # add validation results if available
    if validation:
        cc = validation.get("consensus_check", {})
        cv = validation.get("convergence_check", {})
        context += f"""
VALIDATION GATES:
  Consensus anchor (Step 11): {'PASSED' if cc.get('passes') else 'FAILED'} — {cc.get('note', '')}
  Convergence (Step 13): {cv.get('level', 'N/A').upper()} — {cv.get('note', '')}
  Overall confidence: {validation.get('confidence_pct', 'N/A')}% ({validation.get('confidence_level', 'N/A')})
  Flags: {', '.join(validation.get('flags', [])) or 'None'}
"""

    context += f"""
MACRO ENVIRONMENT:
  CPI Inflation: {macro_context.get('cpi_yoy_pct', 'N/A')}% year over year
  Federal Funds Rate: {macro_context.get('fed_rate', 'N/A')}%
  Weekly Jobless Claims: {macro_context.get('jobless_claims', 'N/A'):,}
  GDP Growth: {macro_context.get('gdp_growth', 'N/A')}%

COMPANY DESCRIPTION:
{company_info['description'][:500]}
"""
    return context


def get_auto_brief(company_info, ensemble_result, macro_context, ticker,
                   scenarios=None, signals=None, validation=None):
    """
    Generates a structured analyst brief returned as a dict with 6 sections.
    Each section is one crisp sentence — displayed as cards in the UI.
    Falls back to a plain string if JSON parsing fails.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = build_context(company_info, ensemble_result, macro_context, ticker,
                            scenarios=scenarios, signals=signals, validation=validation)

    prompt = """Return ONLY a JSON object — no extra text, no markdown, no code fences.
Use exactly these 6 keys. Each value must be one sentence, max 25 words, plain English:

{
  "headline": "bold one-line verdict on the forecast (e.g. 'Models converge strongly; base case intact')",
  "forecast": "what the blended number is and whether models agree or diverge",
  "scenarios": "where the forecast sits relative to the bear/base/bull range",
  "signals": "what the news, price position, and beat rate collectively suggest",
  "macro": "how the macro environment (rates, inflation, GDP) affects this company",
  "risk": "the single biggest risk that could make this forecast wrong"
}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown fences if the model wrapped it anyway
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        import json
        brief = json.loads(raw)
        # validate all 6 keys exist
        required = ["headline", "forecast", "scenarios", "signals", "macro", "risk"]
        if all(k in brief for k in required):
            return brief
    except Exception:
        pass

    # fallback: return a plain string so the UI doesn't crash
    fallback_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": "Write a 3-sentence analyst brief. Plain English only."}
        ],
        max_tokens=200
    )
    return {"headline": "Analysis complete", "forecast": fallback_response.choices[0].message.content,
            "scenarios": "", "signals": "", "macro": "", "risk": ""}


def chat_with_analyst(
    user_message,
    company_info,
    ensemble_result,
    macro_context,
    ticker,
    chat_history=None,
    scenarios=None,
    signals=None,
    validation=None
):
    """
    Handles follow-up questions. Sends the full pipeline context + full
    conversation history every time so the AI never loses context.
    """
    if chat_history is None:
        chat_history = []

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = build_context(company_info, ensemble_result, macro_context, ticker,
                            scenarios=scenarios, signals=signals, validation=validation)

    messages = [{"role": "system", "content": context}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=400
    )
    return response.choices[0].message.content
