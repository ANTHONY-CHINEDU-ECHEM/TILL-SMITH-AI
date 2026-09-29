"""Tillsmith AI web interface. Start with: python tillsmith.py ui"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tillsmith_ai.engine import TillsmithEngine  # noqa: E402
from tillsmith_ai.utils import fmt_money  # noqa: E402

st.set_page_config(page_title="Tillsmith AI", layout="wide")

EXAMPLES = [
    "Checkout conversion collapsed straight after our last release but traffic looks normal. What should we do?",
    "Genuine customers on our Shopify fashion store say their cards are being refused.",
    "Since we migrated to the new site we vanished from google and organic revenue has halved.",
    "Our pages take ages to load on phones since we installed several new apps.",
    "Products we have in the warehouse show as sold out and we keep cancelling orders.",
]


@st.cache_resource(show_spinner="Loading the Tillsmith index")
def engine():
    return TillsmithEngine.shared()


def pct(x):
    return "{:.0%}".format(x)


def page_ask(eng):
    st.header("Diagnose an incident")
    st.caption("Describe what is happening to your store in your own words. Tillsmith diagnoses the incident, "
               "finds comparable incidents and ranks fixes by how often they genuinely recovered conversion.")
    example = st.selectbox("Start from an example or write your own", ["Write my own"] + EXAMPLES)
    question = st.text_area("What is happening?", value="" if example == "Write my own" else example, height=110)
    opts = eng.options()
    with st.expander("Filters and generation settings"):
        c1, c2, c3 = st.columns(3)
        verticals = c1.multiselect("Vertical", opts["verticals"])
        platforms = c2.multiselect("Platform", opts["platforms"])
        models = c3.multiselect("Business model", opts["business_models"])
        c4, c5 = st.columns(2)
        provider = c4.selectbox("Answer provider", ["extractive", "anthropic", "ollama"])
        k = c5.slider("Precedent incidents in context", 3, 15, 8)
    if not st.button("Diagnose", type="primary") or not question.strip():
        return
    filters = {c: v for c, v in (("vertical", verticals), ("platform", platforms), ("business_model", models)) if v}
    result = eng.ask(question, filters=filters or None, k=k, provider=provider)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diagnosis", result["diagnosis"][0]["incident"] if result["diagnosis"] else "Unclear")
    m2.metric("Confidence", "{} ({:.2f})".format(result["confidence"]["label"], result["confidence"]["score"]))
    m3.metric("Grounding score", "{:.2f}".format(result["verification"]["grounding_score"]))
    m4.metric("Answer time", "{:.1f} ms".format(result["timings_ms"]["total"]))
    for note in result["notes"]:
        st.info(note)
    st.markdown(result["answer"])

    tab_e, tab_c, tab_v = st.tabs(["Evidence", "Precedent incidents", "Verification"])
    with tab_e:
        for block in result["recommendations"]:
            st.subheader(block["incident"])
            st.caption("Reference class: {}. Baseline full recovery {}.".format(
                block["reference_class"], pct(block["baseline_full_recovery"])))
            table = pd.DataFrame(block["all_fixes"])
            if not table.empty:
                table = table[["fix", "cases", "full_recovery_rate", "confidence_lower_bound",
                               "median_revenue_recovered_gbp", "median_days_to_recover", "median_fix_cost_gbp",
                               "repeat_rate_90d"]]
                st.dataframe(table, hide_index=True, width="stretch")
                st.bar_chart(table.set_index("fix")["confidence_lower_bound"])
    with tab_c:
        for c in result["cases"]:
            with st.expander("{}  {}  ({})".format(c["incident_id"], c["store_name"], c["recovery_status"])):
                st.write("**Incident.** " + c["incident_summary"])
                st.write("**Resolution.** " + c["resolution_narrative"])
                st.write("**Lesson.** " + c["lessons_learned"])
                st.caption("{} | {} | {} | revenue at risk {} | recovered {}".format(
                    c["vertical"], c["platform"], c["business_model"], fmt_money(c["revenue_at_risk_gbp"]),
                    fmt_money(c["revenue_recovered_gbp"])))
    with tab_v:
        st.json(result["verification"])
        st.json(result["timings_ms"])


def page_search(eng):
    st.header("Incident explorer")
    query = st.text_input("Search the incident base", "payment declines after changing fraud settings")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval method", ["hybrid", "fusion", "bm25", "dense"])
    k = c2.slider("Results", 5, 50, 15)
    if not query.strip():
        return
    out = eng.search(query, k=k, mode=mode)
    frame = pd.DataFrame(out["results"])
    st.caption("Retrieved in {} ms".format(out["timings_ms"]["retrieve"]))
    st.dataframe(frame[["incident_id", "score", "vertical", "platform", "incident_type", "fix_strategy",
                        "recovery_status", "revenue_recovered_gbp"]], hide_index=True, width="stretch")
    pick = st.selectbox("Open an incident", frame["incident_id"].tolist())
    c = eng.case(pick)
    st.write("**Incident.** " + c["incident_summary"])
    st.write("**Resolution.** " + c["resolution_narrative"])
    st.write("**Lesson.** " + c["lessons_learned"])


def page_assess(eng):
    st.header("Store risk profile")
    st.caption("Profile a store to see its predicted incident exposure, the drivers behind it and a prepared "
               "playbook for the incidents it is most likely to face.")
    opts = eng.options()
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        profile = {
            "vertical": c1.selectbox("Vertical", opts["verticals"]),
            "platform": c2.selectbox("Platform", opts["platforms"]),
            "business_model": c3.selectbox("Business model", opts["business_models"]),
            "region": c1.selectbox("Region", opts["regions"]),
            "store_size_band": c2.selectbox("Size band", opts["size_bands"], index=1),
            "fulfilment_model": c3.selectbox("Fulfilment", opts["fulfilment_models"]),
            "monitoring_maturity": c1.selectbox("Monitoring maturity", opts["monitoring"], index=1),
            "annual_revenue_gbp": c2.number_input("Annual online revenue in GBP", 10000, 5_000_000_000, 3_000_000, step=100000),
            "average_order_value_gbp": c3.number_input("Average order value in GBP", 1, 5000, 70),
            "baseline_conversion_rate_pct": c1.number_input("Conversion rate percent", 0.1, 20.0, 2.2),
            "mobile_traffic_share_pct": c2.slider("Mobile traffic share percent", 0, 100, 72),
            "paid_traffic_share_pct": c3.slider("Paid traffic share percent", 0, 100, 35),
            "sku_count": c1.number_input("Live SKUs", 1, 1_000_000, 1500),
            "app_integration_count": c2.number_input("Apps and integrations", 0, 200, 16),
            "payment_provider_count": c3.number_input("Payment providers", 1, 10, 2),
            "release_frequency_per_month": c1.number_input("Releases per month", 0, 200, 6),
        }
        submitted = st.form_submit_button("Assess store", type="primary")
    if not submitted:
        return
    out = eng.assess(profile)
    cols = st.columns(len(out["risks"]))
    for col, risk in zip(cols, out["risks"].values()):
        col.metric(risk["label"], pct(risk["probability"]), help="Base rate {}".format(pct(risk["base_rate"])))
        for d in risk["drivers"]:
            col.caption("{}: {}".format(d["factor"], d["effect"]))
    st.subheader("Prepared playbook")
    st.dataframe(pd.DataFrame(out["playbook"]), hide_index=True, width="stretch")


def page_insights(eng):
    st.header("Incident insights")
    s = eng.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", "{:,}".format(s["incidents"]))
    c2.metric("Revenue at risk", fmt_money(s["total_revenue_at_risk_gbp"]))
    c3.metric("Median detection time", "{:.1f} hours".format(s["median_detection_lag_hours"]))
    c4.metric("Repeat within 90 days", pct(s["repeat_rate_90d"]))
    a, b = st.columns(2)
    a.subheader("Incident types")
    a.bar_chart(pd.Series(s["incident_types"]))
    b.subheader("Recovery")
    b.bar_chart(pd.Series(s["recovery"]))
    st.subheader("What fixes each incident")
    opts = eng.options()
    incident = st.selectbox("Incident type", opts["incident_types"])
    platform = st.selectbox("Platform context", ["All"] + opts["platforms"])
    block = eng.recommend(incident, platform=None if platform == "All" else platform)
    st.caption("Reference class: {}".format(block["reference_class"]))
    table = pd.DataFrame(block["all_fixes"]).set_index("fix")
    st.bar_chart(table["full_recovery_rate"])
    st.dataframe(table, width="stretch")


def main():
    eng = engine()
    st.sidebar.title("Tillsmith AI")
    st.sidebar.caption("Evidence grounded fixes for ecommerce performance incidents.")
    page = st.sidebar.radio("Workspace", ["Diagnose", "Incident explorer", "Store risk profile", "Incident insights"])
    st.sidebar.divider()
    st.sidebar.caption("{:,} historical incidents indexed".format(eng.index.size))
    {"Diagnose": page_ask, "Incident explorer": page_search, "Store risk profile": page_assess,
     "Incident insights": page_insights}[page](eng)


main()
