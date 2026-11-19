from pathlib import Path
import runpy

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

if st.query_params.get("workspace") == "detector":
    try:
        runpy.run_path(str(ROOT / "pcam_workstation.py"), run_name="__main__")
    finally:
        if st.sidebar.button("Return to Aetheris", key="return_to_aetheris"):
            del st.query_params["workspace"]
            st.rerun()
else:
    st.set_page_config(page_title="Aetheris PCam Detector", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        "<style>header[data-testid=stHeader]{display:none}"
        ".stMainBlockContainer,.block-container{max-width:none;padding:0}"
        "[data-testid=stAppViewContainer]{background:#faf9f6}</style>",
        unsafe_allow_html=True,
    )
    workspace = components.declare_component("aetheris", path=str(ROOT / "aetheris" / "dist"))
    if workspace(key="aetheris_workspace", default=None) == "detector":
        st.query_params["workspace"] = "detector"
        st.rerun()
