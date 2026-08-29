"""Streamlit demo for the Track C/D router (M4).

Usage:
    pip install -e ".[demo]"          # streamlit, kept out of the `router` extra
    streamlit run scripts/router_demo_app.py

Then open the printed http://localhost:8501 URL in any browser (Streamlit
just needs a Chromium/Firefox-class browser to render — Brave is fine;
pass --server.headless true to stop it trying to auto-open one).

Runs the curated scenarios in `metis.router.demo` through the real router
— confidence diagnostic + frozen surrogate checkpoint + precomputed DNS
lookup, no synthetic/mocked data. This demo shows the deterministic
M1/M2 core; it does not call the M3 LLM layer (`metis.router.agent.ask`),
since that needs Anthropic API credentials this demo shouldn't depend on
to run reliably.
"""
from __future__ import annotations

import streamlit as st

from metis.router.demo import run_scenarios

st.set_page_config(page_title="METIS router demo", page_icon="\U0001f9ed")

st.title("Transcritical surrogate/solver router")
st.caption(
    "Confidence-gated routing between a fast neural-operator surrogate and "
    "the group's DNS database, for high-pressure transcritical channel flow. "
    "See PROJECT_CONTEXT.md and docs/agent_implementation_plan.md for how "
    "the confidence diagnostic was validated."
)

if st.button("Run all scenarios", type="primary"):
    with st.spinner("Routing every scenario through the real router..."):
        st.session_state["results"] = run_scenarios()

results = st.session_state.get("results")
if results is None:
    st.info("Click **Run all scenarios** to route all 8 curated cases through the live router.")
else:
    n_surrogate = sum(1 for r in results if r.get("source") == "surrogate")
    n_solver = sum(1 for r in results if r.get("source") == "full_solver")
    n_error = sum(1 for r in results if "error" in r)
    st.write(
        f"**{n_surrogate}** routed to the surrogate, **{n_solver}** routed to the "
        f"precomputed solver, **{n_error}** had no precomputed answer available."
    )

    for r in results:
        scenario = r["scenario"]
        with st.expander(f"{scenario.name} — {scenario.description}"):
            st.write(
                f"`Pb_Pc`={scenario.Pb_Pc}  `Thw_Tc`={scenario.Thw_Tc}  "
                f"`Tcw_Tc`={scenario.Tcw_Tc}"
            )
            if "error" in r:
                st.error(r["error"])
                continue

            if r["source"] == "surrogate":
                st.success(f"Routed to: **surrogate** (confidence: {r['confidence_level']})")
            else:
                st.warning(f"Routed to: **full solver** (confidence: {r['confidence_level']})")
            st.caption(r["confidence_reason"])

            st.write("Predicted field means (u', T', cp'):")
            st.json(r["field_means"])
