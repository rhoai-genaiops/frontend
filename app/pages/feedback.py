import os
import json
import requests
import streamlit as st
from urllib.parse import urljoin

BACKEND_ENDPOINT = os.getenv("BACKEND_ENDPOINT", "http://localhost:8000")

st.set_page_config(
    page_title="Canopy - Feedback Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2e8b57;'>Canopy - Feedback Dashboard</h1>
        <p style='font-size: 1.2em;'>Review user feedback and export evaluation datasets 📊</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if feedback feature is enabled
try:
    flags_resp = requests.get(urljoin(BACKEND_ENDPOINT, "/feature-flags"), timeout=10)
    flags_resp.raise_for_status()
    feature_flags = flags_resp.json()
except Exception:
    feature_flags = {}

if not feature_flags.get("feedback", False):
    st.warning("The feedback feature is not enabled. Enable it in the backend configuration.")
    st.stop()

# Fetch feedback data
try:
    feedback_resp = requests.get(urljoin(BACKEND_ENDPOINT, "/feedback"), timeout=10)
    feedback_resp.raise_for_status()
    feedback_data = feedback_resp.json()
except Exception as e:
    st.error(f"Failed to load feedback data: {e}")
    st.stop()

total = feedback_data.get("total", 0)
entries = feedback_data.get("feedback", [])

thumbs_up = [e for e in entries if e["rating"] == "thumbs_up"]
thumbs_down = [e for e in entries if e["rating"] == "thumbs_down"]

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback", total)
col2.metric("👍 Positive", len(thumbs_up))
col3.metric("👎 Negative", len(thumbs_down))

st.markdown("---")

if entries:
    # Filter
    filter_option = st.radio(
        "Filter by:",
        ["All", "Positive only", "Negative only"],
        horizontal=True,
        key="feedback_filter",
    )

    if filter_option == "Positive only":
        display_entries = thumbs_up
    elif filter_option == "Negative only":
        display_entries = thumbs_down
    else:
        display_entries = entries

    # Display feedback entries
    if display_entries:
        for e in display_entries:
            rating_icon = "👍" if e["rating"] == "thumbs_up" else "👎"
            with st.expander(f"#{e['id']} {rating_icon} — {e['timestamp'][:19]}", expanded=False):
                st.markdown("**Prompt:**")
                st.text(e["input_text"])
                st.markdown("**Response:**")
                st.text(e["response_text"])
    else:
        st.info("No feedback entries match the selected filter.")
else:
    st.info("No feedback collected yet. Use the Summarization feature and rate some responses!")

# Export section
if thumbs_down:
    st.markdown("---")
    st.subheader("Export for Evaluation")
    st.markdown(
        f"**{len(thumbs_down)}** negative feedback entries can be exported as an evaluation dataset. "
        "Fill in the `expected_answer` field to create regression test cases."
    )
    try:
        export_resp = requests.get(urljoin(BACKEND_ENDPOINT, "/feedback/export"), timeout=10)
        export_resp.raise_for_status()
        st.download_button(
            label="Download feedback-eval-dataset.yaml",
            data=export_resp.text,
            file_name="feedback-eval-dataset.yaml",
            mime="application/x-yaml",
            key="download_eval_yaml",
        )
    except Exception as e:
        st.error(f"Failed to load export data: {e}")

# A/B Testing feedback section
if feature_flags.get("ab_testing", False):
    st.markdown("---")
    st.subheader("🔬 A/B Prompt Comparison Results")

    try:
        ab_resp = requests.get(urljoin(BACKEND_ENDPOINT, "/feedback/ab"), timeout=10)
        ab_resp.raise_for_status()
        ab_data = ab_resp.json()
    except Exception as e:
        ab_data = None
        st.error(f"Failed to load A/B feedback data: {e}")

    if ab_data:
        ab_total = ab_data.get("total", 0)
        ab_a_wins = ab_data.get("prompt_a_wins", 0)
        ab_b_wins = ab_data.get("prompt_b_wins", 0)

        ab_col1, ab_col2, ab_col3 = st.columns(3)
        ab_col1.metric("Total Comparisons", ab_total)
        ab_col2.metric("Prompt A Wins", ab_a_wins)
        ab_col3.metric("Prompt B Wins", ab_b_wins)

        # Intelligence / recommendation
        if ab_total > 0:
            rate_a = ab_a_wins / ab_total * 100
            rate_b = ab_b_wins / ab_total * 100

            st.markdown("#### Analysis")

            # Win rate bar
            bar_col1, bar_col2 = st.columns(2)
            bar_col1.progress(rate_a / 100, text=f"Prompt A: {rate_a:.0f}%")
            bar_col2.progress(rate_b / 100, text=f"Prompt B: {rate_b:.0f}%")

            # Recommendation
            if ab_total < 3:
                st.info(
                    f"**Too early to call.** Only {ab_total} comparison(s) collected. "
                    "Run more A/B tests to get a reliable signal."
                )
            elif rate_a > rate_b and rate_a >= 70:
                st.success(
                    f"**Prompt A is the clear winner** with a {rate_a:.0f}% win rate over {ab_total} comparisons. "
                    "Consider promoting it to your primary `summarization.prompt`."
                )
            elif rate_b > rate_a and rate_b >= 70:
                st.success(
                    f"**Prompt B is the clear winner** with a {rate_b:.0f}% win rate over {ab_total} comparisons. "
                    "Consider promoting it to your primary `summarization.prompt`."
                )
            elif abs(rate_a - rate_b) <= 20:
                st.warning(
                    f"**No clear winner yet.** Prompt A: {rate_a:.0f}% vs Prompt B: {rate_b:.0f}%. "
                    "The prompts are performing similarly. Collect more comparisons or try a more distinct Prompt B."
                )
            else:
                leader = "A" if rate_a > rate_b else "B"
                lead_rate = max(rate_a, rate_b)
                st.info(
                    f"**Prompt {leader} is ahead** at {lead_rate:.0f}% over {ab_total} comparisons. "
                    "A few more comparisons would strengthen confidence in the result."
                )

        ab_entries = ab_data.get("entries", [])
        if ab_entries:
            st.markdown("#### Comparison History")
            for ae in ab_entries:
                pref = ae.get("preference", "")
                pref_label = {"a": "Preferred A", "b": "Preferred B"}.get(pref, pref)
                winner = ae.get("winning_prompt", "")
                with st.expander(
                    f"#{ae['id']} {pref_label} (winner: {winner}) — {ae['timestamp'][:19]}",
                    expanded=False,
                ):
                    st.markdown("**Input:**")
                    st.text(ae.get("input_text", ""))
                    resp_col1, resp_col2 = st.columns(2)
                    with resp_col1:
                        st.markdown("**Response A:**")
                        st.text(ae.get("response_a", ""))
                    with resp_col2:
                        st.markdown("**Response B:**")
                        st.text(ae.get("response_b", ""))
                    mapping = ae.get("prompt_mapping", {})
                    st.caption(f"A = {mapping.get('a', '?')} | B = {mapping.get('b', '?')}")
        else:
            st.info("No A/B comparisons collected yet. Use the Summarize button to start comparing!")
