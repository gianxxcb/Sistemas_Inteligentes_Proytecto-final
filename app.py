"""QuantLab ML desktop research interface."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys


def _launch_if_direct() -> None:
    has_streamlit = find_spec("streamlit") is not None
    if has_streamlit:
        from streamlit import runtime

        if runtime.exists():
            return

    app = Path(__file__).resolve()
    venv = app.parent / ".venv"
    project_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not project_python.is_file() and not has_streamlit:
        raise SystemExit("Streamlit no está instalado. Prepara el entorno indicado en README.md.")

    python = sys.executable if has_streamlit else project_python
    try:
        code = subprocess.call(
            [str(python), "-m", "streamlit", "run", str(app), "--server.address=127.0.0.1", *sys.argv[1:]],
            cwd=app.parent,
        )
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    _launch_if_direct()

import streamlit as st

from apps import research_pages_v3 as views
from apps.terminal_visuals import terminal_style
from apps.technique_guide import guide

st.set_page_config(page_title="QuantLab · Investigación V3", page_icon=":material/monitoring:", layout="wide")

terminal_style()
pages = st.navigation({
    "Terminal": [
        st.Page(views.market, title="Mercado", icon=":material/candlestick_chart:", default=True),
        st.Page(views.performance, title="Rendimiento", icon=":material/leaderboard:", url_path="rendimiento"),
        st.Page(views.portfolio, title="Cartera", icon=":material/account_balance_wallet:", url_path="resultados"),
    ],
    "Laboratorio": [
        st.Page(views.simulator, title="Simulador", url_path="simulador"),
        st.Page(views.scenarios, title="Escenarios", url_path="escenarios"),
        st.Page(views.specialists, title="Especialistas", url_path="especialistas"),
        st.Page(views.orchestrator, title="Orquestador", url_path="orquestador"),
        st.Page(views.validation, title="Validación", url_path="validacion"),
    ],
    "Proyecto": [
        st.Page(views.about, title="Acerca de", url_path="acerca-de"),
        st.Page(guide, title="Guía de técnicas", url_path="tecnicas"),
    ],
})
st.sidebar.markdown("**QuantLab ML**")
st.sidebar.caption("Investigación de futuros · V3")
pages.run()
