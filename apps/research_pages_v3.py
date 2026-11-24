"""Streamlit views for the local Futures research datasets and frozen runs."""
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from apps import trading_desk_v3 as desk
from apps import terminal_visuals as visual

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validation"
STYLES = {"tactical":"Táctico · 1 h", "swing_short":"Swing corto · 4 h", "swing_positional":"Posicional · 1 día"}


def generation():
    pointer = OUT/"integrated_current.txt"
    if not pointer.exists():
        st.info("Genera la corrida con python scripts/evaluate_integrated_v3.py para consultar esta vista.")
        st.stop()
    return OUT/pointer.read_text().strip()


def report():
    return json.loads((generation()/"report.json").read_text(encoding="utf-8"))


def money(value):
    return f"{value:,.2f} USDT"


def style_choice():
    return st.segmented_control("Estilo", list(STYLES), default="tactical", format_func=STYLES.get, required=True)


def metrics(items):
    visual.stat_strip([(label,str(value),"") for label,value in items])


@st.cache_data(max_entries=12)
def last_bars(path, revision, limit):
    return pd.read_parquet(path, columns=["date","open","close","high","low","volume"]).sort_values("date").tail(limit)


def market():
    st.title("Mercado de futuros")
    st.caption("Contratos perpetuos de Binance · Precios en USDT · Horario UTC")
    raw = ROOT/"data/raw/futures"
    left,scale,window,clock = st.columns([1.3,1.5,.8,1.7],vertical_alignment="bottom")
    names = sorted(p.stem for p in (raw/"1h").glob("*.parquet"))
    if not names:
        st.info("Faltan las velas locales de futuros.")
        return
    symbol = left.selectbox("Contrato", names, index=names.index("BTCUSDT") if "BTCUSDT" in names else 0)
    tf = scale.segmented_control("Intervalo de vela", ["15m","1h","4h","1d"],default="1h",required=True)
    path = raw/tf/f"{symbol}.parquet"
    if not path.exists():
        st.info("Este contrato no tiene velas en el intervalo seleccionado.")
        return
    n = window.selectbox("Velas",[48,96,192,384],index=1)
    # Keep enough preceding bars for the oldest selectable close.
    bars = last_bars(path,path.stat().st_mtime_ns,2*n-1)
    if bars.empty:
        st.info("El archivo seleccionado está vacío.")
        return
    times = bars.date.tail(n).tolist()
    moment = clock.selectbox("Vela de análisis (UTC)",times[::-1],format_func=lambda t:t.strftime("%d/%m/%Y %H:%M"),
                          key=f"market_moment_{symbol}_{tf}_{n}")
    bars = bars[bars.date <= moment].tail(n)
    plot,signals = st.columns([2.25,1],gap="medium")
    with plot.container(gap=None):
        visual.quote_chart(symbol,tf,bars)
    with signals.container(border=True):
        plans = desk.signal_board(symbol,tf,moment)
    if plans is not None:
        st.subheader("Planes sobre el cierre")
        st.dataframe(plans,hide_index=True,height="content",
                     column_config={"Lectura de la vela":st.column_config.TextColumn(width="large")})
        st.caption("Niveles orientativos del archivo histórico. La cartera aplica sus filtros de estabilidad y presupuesto antes de simular una entrada.")


def scenarios():
    st.title("Estabilidad de escenarios")
    st.write("El índice ARI mide cuánto se mantienen los grupos al volver a ajustar el modelo. El umbral del proyecto es 0,70.")
    original = json.loads((ROOT/"experiments/v3/representation/stability_review/training_summary.json").read_text())
    revised = json.loads((ROOT/"experiments/v3/representation/scenario_recheck/summary.json").read_text())
    evidence = {r["style"]:r for r in revised["styles"]}
    rows = {r["style"]:r for r in report()["styles"]}
    table = []
    for r in original:
        s = r["style"]
        table.append({"Estilo":STYLES[s],"Autoencoder V3":round(r["scenario_stability"]["mean_ari"],3),
                      "Autoencoder revisado":round(evidence[s]["mean_ari"],3),
                      "ARI en prueba posterior":round(evidence[s]["test_ari"],3),
                      "Grupos del ensayo 2025":round(rows[s]["ari"],3),
                      "Decisión del ensayo":"Admitido" if rows[s]["ari"]>=.7 else "Abstención"})
    st.table(pd.DataFrame(table).set_index("Estilo"))
    st.caption("La revisión agrupa los embeddings del autoencoder. La proyección se elige dentro del entrenamiento. El ARI compara cinco reajustes por bloques en validación y en una partición cronológica posterior. El régimen diario de Mercado se calcula con un HMM de BTC separado del ensayo reservado de 2025.")
    original_panel,trial_panel = st.columns(2,border=True)
    indexed = pd.DataFrame(table).set_index("Estilo")
    with original_panel:
        st.subheader("Autoencoder V3")
        visual.horizontal_values(indexed["Autoencoder revisado"],"Estabilidad ARI",reference=.7)
    with trial_panel:
        st.subheader("Ensayo reservado")
        visual.horizontal_values(indexed["Grupos del ensayo 2025"],"Estabilidad ARI",reference=.7)
    st.caption("El ensayo usa grupos de indicadores ajustados antes de 2023 y compara cinco remuestreos sobre observaciones fijas de 2024.")
    style = style_choice()
    desk.catalog(style)
    st.subheader("Escenarios del ensayo reservado")
    st.caption("Estos tres grupos se ajustaron antes de 2023. La tabla describe lo observado en cada grupo durante 2025.")
    forecasts = pd.read_parquet(generation()/"forecasts.parquet",filters=[("style","=",style)])
    summary = forecasts.groupby("scenario_id").agg(Señales=("net","size"),Largos=("sign",lambda x:(x>0).mean()),
                                                  Acierto=("net",lambda x:(x>0).mean()),Retorno=("net","mean"))
    st.table(summary.rename_axis("Grupo").style.format({"Largos":"{:.1%}","Acierto":"{:.1%}","Retorno":"{:+.2%}"}))


def performance():
    st.title("Rendimiento por especialista")
    desk.returns_panel(style_choice(),generation())


def specialists():
    st.title("Especialistas")
    s = style_choice()
    source = ROOT/f"experiments/v3/specialists/architecture_comparison/architecture_comparison_{s}.json"
    original = json.loads(source.read_text())["experts"]
    table = []
    current = report()
    for r in current["experts"]:
        if r["style"] != s:
            continue
        baseline = original[r["expert"]]
        table.append({"Especialista":r["expert"],"Benchmark original TCN / GRU":"Muestra insuficiente" if "blocked" in baseline["status"] else "Evaluado",
                      "Ajuste histórico":r["train"],"Pruebas fuera de activo":r["unseen_rows"],"Activos evaluados":r["assets"],
                      "Brier":round(r["unseen_brier"],3) if r["unseen_brier"] is not None else None,
                      "Muestra del ensayo":"Suficiente" if r["sufficient"] else "Insuficiente"})
    overview = pd.DataFrame(table).set_index("Especialista").rename(index=desk.NAMES)
    visual.horizontal_values(overview.Brier,"Error Brier · menor es mejor")
    st.dataframe(overview,height="content")
    st.caption("Cada prueba de 2024 usa contratos excluidos del ajuste en ese pliegue. Brier mide el error de probabilidad (menor es mejor). El benchmark original TCN/GRU conserva su propio criterio de evidencia.")
    st.subheader("Reevaluación secuencial por especialista")
    st.write("La reevaluación usa episodios anteriores a 2025. Cada pliegue prueba contratos distintos de los usados para ajustar y calibrar. Las cuatro columnas comparan el error Brier sobre los mismos episodios evaluados. Un valor menor indica mejor estimación de la probabilidad.")
    comparison = current["confirmation"]["benchmark"]["sequence_recheck"]
    records = [row for row in comparison.get("comparisons", []) if row["style"] == s]
    if not records:
        st.info("Esta corrida no incluye el detalle de la reevaluación para este estilo.")
        return
    rows, details = [], []
    for row in records:
        name = desk.NAMES[row["expert"]]
        rows.append({"Especialista": name, "Resultado": "Evaluado" if row["status"] == "evaluated_diagnostic_only" else "Inconcluso",
                     "Pliegues evaluados": f"{row['folds']}/{row['total_folds']}", "Episodios evaluados": row["samples"],
                     "Contratos evaluados": row["assets"], **row["brier"]})
        for fold in row["blockers"]:
            reasons = []
            if fold["train"] < 30:
                reasons.append("Menos de 30 casos de ajuste")
            if fold["calibration"] < 30:
                reasons.append("Menos de 30 casos de calibración")
            if fold["unseen_asset_test"] == 0:
                reasons.append("Sin episodios de prueba")
            for partition, label in [("train","ajuste"), ("calibration","calibración")]:
                if f"{partition} has one target class" in fold["reasons"]:
                    reasons.append(f"Falta una clase en {label}")
            details.append({"Especialista": name, "Pliegue": int(fold["fold"].split("_")[-1])+1,
                            "Ajuste":fold["train"], "Calibración":fold["calibration"],
                            "Prueba disponible":fold["unseen_asset_test"], "Motivo":" · ".join(reasons)})
    scores = ["HistGradientBoosting","TCN","GRU","Null"]
    overview = pd.DataFrame(rows).reindex(columns=["Especialista", "Resultado", "Pliegues evaluados",
                                      "Episodios evaluados", "Contratos evaluados", *scores]).rename(columns={"Null":"Referencia"})
    st.dataframe(overview, hide_index=True, height="content",
                 column_config={k:st.column_config.NumberColumn(format="%.4f")
                                for k in ["HistGradientBoosting","TCN","GRU","Referencia"]})
    st.caption("Cada pliegue requiere 30 episodios de ajuste y 30 de calibración, con ambas clases en cada grupo. La comparación completa exige todos los pliegues y al menos 30 episodios de prueba en tres contratos. Los resultados parciales se muestran con su muestra real.")
    if details:
        st.dataframe(pd.DataFrame(details), hide_index=True, height="content")


def orchestrator():
    st.title("Orquestador")
    s = style_choice()
    comparison = OUT/"e7_orchestration/summary.json"
    if comparison.is_file():
        research = next(r for r in json.loads(comparison.read_text(encoding="utf8")) if r["style"] == s)
        st.subheader("Comparación E7 en contratos no reservados")
        st.write(f"Política seleccionada en el diagnóstico: **{research['selected_policy']}**. "
                 f"Se evaluaron {research['observations']:,} oportunidades distintas de 2023–2024 en "
                 f"{len(research['splits'])} particiones purgadas y {research['paths']} trayectorias completas.")
        policy_rows = [{"Política":name,"Log loss":m["log_loss"],"Retorno por oportunidad":m["net_per_opportunity"],
                        "Participación":m["participation"],"Sharpe deflactado (DSR)":m["dsr"]}
                       for name,m in research["policies"].items()]
        st.table(pd.DataFrame(policy_rows).set_index("Política").style.format({
            "Log loss":"{:.4f}", "Retorno por oportunidad":"{:+.4%}", "Participación":"{:.1%}",
            "Sharpe deflactado (DSR)":"{:.3f}"}))
        st.caption("Cada oportunidad usa un horizonte de una vela y financiación observada. Una abstención aporta retorno cero. La participación mide la fracción de oportunidades admitidas. El DSR se calcula sobre retornos diarios de investigación.")
        if research["policies"][research["selected_policy"]]["participation"] == 0:
            st.info("La política seleccionada se abstuvo en todas las oportunidades. Su retorno fue cero y su Sharpe carece de interpretación económica en esta muestra.")
        st.markdown("**H5 · ¿Mejora la selección al añadir el régimen HMM?**")
        for year,h in research["h5"].items():
            st.write(f"{year} · {h['rows']:,} casos · mejora de log loss del experto elegido: {h['log_loss_gain']:+.2%}. "
                     f"Diferencia de expectativa, IC 95 %: [{h['ci'][0]:+.4%}, {h['ci'][1]:+.4%}].")
        st.caption("Se requiere una mejora superior al 3 % en log loss y un límite inferior positivo en expectativa neta.")
        st.markdown("**H6 · ¿Hay evidencia para activar la red neuronal?**")
        st.write("La red supera el criterio escalonado." if research["gate_supported"] else
                 "La red permanece en diagnóstico. La promoción exige una mejora previa del nivel 1, ventaja en log loss y expectativa tras corregir por comparaciones múltiples, y Sharpe deflactado significativo.")
        st.table(pd.DataFrame([{"Referencia":name,"p ajustado · log loss":round(h["loss_p_fdr"],4),
                                "p ajustado · retorno":round(h["return_p_fdr"],4)}
                               for name,h in research["h6"].items()]).set_index("Referencia"))
        st.subheader("Ensayo reservado de 2025")
    row = next(r for r in report()["styles"] if r["style"]==s)
    levels = ["Promedio de expertos","Competencia jerárquica","Gating neuronal"]
    metrics([("Combinación seleccionada",levels[row["level"]]),("Pronósticos reservados",f"{row['holdout']:,}"),
             ("Cobertura P10–P90",f"{row['coverage']:.1%}")])
    st.write("Cada experto activo estima la probabilidad de que su plan cubra los costes. El orquestador compara pesos iguales, competencia por escenario y régimen, y una red que puede abstenerse.")
    st.caption("El intervalo se calibró entre octubre y diciembre de 2024. Su objetivo es contener el 80 % de los retornos del ensayo antes de funding.")
    frame = pd.read_parquet(generation()/"forecasts.parquet",filters=[("style","=",s)])
    monthly = frame.assign(mes=frame.entry_date.dt.strftime("%Y-%m"),
                           cubierto=frame.net.between(frame.p10,frame.p90)).groupby("mes").cubierto.mean()
    st.subheader("Cobertura del intervalo por mes")
    fig = go.Figure(go.Scatter(x=monthly.index,y=monthly,mode="lines+markers",line_color=visual.TEAL,
                                hovertemplate="%{x} · %{y:.1%}<extra></extra>"))
    fig.add_hline(y=.8,line_dash="dot",line_color="#738789",annotation_text="Objetivo 80 %")
    fig.update_layout(height=220,margin=dict(l=0,r=15,t=12,b=0),yaxis=dict(tickformat=".0%",range=[0,1.06]))
    st.plotly_chart(fig,width="stretch",alt="Cobertura mensual observada de P10–P90 frente al objetivo de 80 %")
    st.subheader("Últimas decisiones de 2025")
    view = frame.sort_values("entry_date",ascending=False).head(10).copy()
    view["Acción"] = view.eligible.map({True:"Candidata",False:"Abstención"})
    view["Dirección"] = view.sign.map({1:"Largo",-1:"Corto"})
    for k in ["probability","p10","p90"]:
        view[k] = view[k].map("{:.1%}".format)
    st.dataframe(view[["entry_date","symbol","Dirección","probability","p10","p90","Acción"]].rename(
        columns={"entry_date":"Entrada UTC","symbol":"Contrato","probability":"P(ganar)","p10":"P10","p90":"P90"}),hide_index=True)


def read_portfolio_file(run, filename):
    try:
        return pd.read_parquet(run/filename)
    except FileNotFoundError:
        st.warning(f"Corrida incompleta ({run.name}). Falta el archivo {filename}.")
        st.stop()


def portfolio():
    st.title("Cartera simulada")
    st.caption("Activos reservados · Enero a diciembre de 2025 · Capital inicial de 10.000 USDT")
    control = st.radio("Política",["Con filtro de estabilidad","Control exploratorio sin filtro"],horizontal=True)
    prefix = "control_" if control.startswith("Control") else ""
    run = generation()
    margin_pointer = OUT/"margin_current.txt"
    with_margin = False
    if margin_pointer.is_file():
        version = st.segmented_control("Simulación",["Margen aislado E9","Ensayo histórico original"],
                                       default="Margen aislado E9",required=True)
        with_margin = version=="Margen aislado E9"
        if with_margin:
            run = OUT/margin_pointer.read_text().strip()
    eq = read_portfolio_file(run, f"{prefix}equity.parquet")
    trades = read_portfolio_file(run, f"{prefix}trades.parquet")
    metrics([("Capital final",money(eq.equity.iloc[-1])),("Operaciones",len(trades)),
             ("Caída máxima",f"{(1-eq.equity/eq.equity.cummax()).max():.2%}")])
    capital, risk = st.columns([1.65,1],border=True)
    with capital:
        st.subheader("Evolución del patrimonio")
        st.line_chart(eq,x="date",y="equity",color=visual.TEAL,height=215,y_label="USDT")
    with risk:
        st.subheader("Caída desde máximos")
        drawdown = (eq.set_index("date").equity / eq.set_index("date").equity.cummax() - 1)*100
        st.area_chart(drawdown.rename("Caída (%)"),color=visual.RUST,height=215,y_label="Caída (%)")
    if with_margin:
        st.subheader("Margen y concentración")
        st.write("Al abrir una posición se limita el importe al 10 % del patrimonio por contrato. "
                 "Los contratos con correlación superior a 0,70 con BTC comparten un límite del 40 %. "
                 "La exposición total puede alcanzar el 100 %. El valor de las posiciones abiertas varía con el precio.")
        st.line_chart(eq.set_index("date")[["gross","btc","margin"]].rename(columns={
            "gross":"Exposición total","btc":"Exposición correlacionada con BTC","margin":"Margen aislado"}),
            height=220,y_label="USDT")
        metrics([("Liquidaciones de margen",int(trades.exit_reason.eq("Liquidación").sum())),
                 ("Margen máximo utilizado",money(eq.margin.max())),("Financiación neta pagada",money(trades.funding.sum()))])
    if trades.empty:
        st.info("La política mantuvo el capital en efectivo durante todo el periodo.")
    else:
        if trades.missing.sum():
            st.warning(f"Faltan {trades.missing.sum()} observaciones esperadas de funding en posiciones abiertas. El balance incluye únicamente las tasas disponibles.")
        st.subheader("Operaciones recientes")
        by_coin = st.selectbox("Moneda del libro",["Todas",*sorted(trades.symbol.unique())])
        if by_coin != "Todas":
            trades = trades[trades.symbol==by_coin]
        metrics([("Ganancias de operaciones",money(trades.pnl.clip(lower=0).sum())),
                 ("Pérdidas de operaciones",money(trades.pnl.clip(upper=0).sum())),
                 ("Resultado de la selección",money(trades.pnl.sum()))])
        if with_margin:
            book = trades.sort_values("entry_date",ascending=False).reset_index(drop=True)
            idx = st.selectbox("Plan de operación", range(len(book)),format_func=lambda i:
                f"{book.at[i,'symbol']} · {book.at[i,'entry_date']:%d/%m/%Y %H:%M} UTC · {STYLES[book.at[i,'style']]}")
            trade = book.iloc[idx]
            candles = pd.read_parquet(ROOT/f"data/raw/futures/15m/{trade.symbol}.parquet",
                columns=["date","open","high","low","close"],filters=[
                    ("date",">=",trade.entry_date),("date","<",trade.exit_date)])
            fig = go.Figure(go.Candlestick(x=candles.date,open=candles.open,high=candles.high,
                low=candles.low,close=candles.close,name="Precio"))
            for key,label,color in [("entry","Entrada",visual.TEAL),("target","Objetivo","#4285be"),
                                    ("stop","Stop",visual.RUST),("liquidation","Liquidación","#c63c51")]:
                fig.add_hline(y=trade[key],line_dash="dot",line_color=color,annotation_text=label)
            fig.add_scatter(x=[trade.exit_date],y=[trade.exit_price],mode="markers",name=trade.exit_reason)
            fig.update_layout(height=325,xaxis_rangeslider_visible=False,yaxis_title="Precio (USDT)")
            st.plotly_chart(fig,width="stretch")
            st.caption(f"Salida por {trade.exit_reason.lower()} · Apalancamiento {trade.leverage:g}× · "
                       f"Comisiones y deslizamiento {money(trade.fees)} · Financiación neta {money(trade.funding)}.")
        st.dataframe(trades.tail(12).iloc[::-1][["symbol","entry_date","exit_date","pnl"]].rename(columns={
            "symbol":"Contrato","entry_date":"Entrada UTC","exit_date":"Salida UTC","pnl":"Resultado (USDT)"}),hide_index=True)
    st.caption("Valoración horaria. Hasta tres entradas por semana y un contrato abierto por activo. La simulación descuenta comisiones y deslizamiento.")


@st.cache_data(max_entries=6)
def episode_symbols(path, revision):
    return sorted(pd.read_parquet(path, columns=["symbol"]).symbol.unique())


@st.cache_data(max_entries=12)
def contract_episodes(path, revision, symbol):
    columns = ["symbol", "expert_name", "direction", "decision_date", "outcome_status", "episode_return_R_net"]
    return pd.read_parquet(path, columns=columns, filters=[("symbol", "=", symbol)])


def simulator():
    st.title("Simulador de episodios")
    s = style_choice()
    path = desk.replay_path(s)
    independent = ROOT/"data/episodes/independent_v3"
    if (independent/"summary.json").is_file():
        choice = st.radio("Muestra de episodios", ["Replay completo","Muestra E4 sin solapamiento"], horizontal=True)
        if choice.startswith("Muestra"):
            path = independent/f"v3_episodes_{s}.parquet"
            rows = [r for r in json.loads((independent/"summary.json").read_text()) if r["style"]==s]
            st.caption("Se conserva la primera entrada disponible por contrato y especialista. El siguiente plan requiere que la posición anterior ya haya cerrado. La tabla resume los episodios resueltos de los 25 contratos.")
            st.table(pd.DataFrame([{"Especialista":desk.NAMES[r["expert"]],"Resueltos originales":r["before"],
                                    "Resueltos admitidos":r["resolved"],"Unicidad original":round(r["before_uniqueness"],3),
                                    "Unicidad E4":round(r["uniqueness"],3)} for r in rows]).set_index("Especialista"))
    revision = path.stat().st_mtime_ns
    symbols = episode_symbols(path, revision)
    if not symbols:
        st.info("No hay episodios disponibles para este estilo.")
        return
    sym = st.selectbox("Contrato", symbols)
    selected = contract_episodes(path, revision, sym)
    metrics([("Episodios",len(selected)),("Resueltos",int(selected.outcome_status.eq("resolved").sum())),
             ("Filtrados",int(selected.outcome_status.eq("filtered").sum()))])
    counts = selected.outcome_status.value_counts().rename(index={
        "resolved":"Resueltos","filtered":"Filtrados","liquidated":"Liquidados","censored":"Censurados"})
    st.subheader("Desenlace de los episodios")
    visual.horizontal_values(counts,"Episodios")
    st.dataframe(selected.sort_values("decision_date",ascending=False).head(25).drop(columns="symbol").rename(columns={
        "decision_date":"Decisión UTC","direction":"Dirección","expert_name":"Experto","outcome_status":"Estado","episode_return_R_net":"Resultado neto (R)"}),hide_index=True)
    st.caption("Replay causal por eventos sobre 25 contratos. La cartera reservada de 2025 mantiene su ensayo congelado de una vela.")
    st.caption("Los costes incluyen comisión y deslizamiento configurados. Cuando falta una tasa de financiación, el simulador imputa 0,01 % por liquidación y registra el conteo en funding_estimated_count.")


def validation():
    st.title("Validación y calidad")
    r = report()
    summary_path = OUT/"e8/summary.json"
    if summary_path.is_file():
        e8 = json.loads(summary_path.read_text(encoding="utf8"))
        st.subheader("Resultados de las nueve hipótesis")
        st.write("Las pruebas usan contratos de desarrollo hasta diciembre de 2024. La tabla indica qué criterio "
                 "se alcanzó. En las comparaciones de arquitecturas, la falta de superioridad tampoco prueba equivalencia.")
        st.dataframe(pd.DataFrame(e8["hypotheses"])[["id","title","result","evidence","adjusted_p"]].rename(columns={
            "id":"Hipótesis","title":"Pregunta","result":"Resultado","evidence":"Evidencia","adjusted_p":"p ajustado (FDR)"}),
            hide_index=True,height="content")
        st.caption(e8["cpcv_description"])
    verdict_path = OUT/"e10/decision.json"
    if verdict_path.is_file():
        decision = json.loads(verdict_path.read_text(encoding="utf8"))
        st.subheader("Dictamen final de la V3")
        st.write(decision["conclusion"])
    st.subheader("Evaluación de despliegue")
    if r["reasons"]:
        st.warning("Despliegue bloqueado")
        for reason in r["reasons"]:
            st.write("• " + reason)
    else:
        st.info("Criterios del ensayo superados. Se requiere una réplica independiente.")
    st.subheader("Predicción frente a la referencia")
    st.write("Entre enero y septiembre de 2024 se comparó el nivel 0 con la frecuencia de éxitos del entrenamiento. Para pasar, el Brier del modelo debe ser estrictamente menor que el de la referencia en cada pliegue temporal. Los pliegues comparten observaciones.")
    prediction, intervals = [], []
    for style in r["styles"]:
        folds = style.get("baseline_folds", [])
        worst = max(folds, key=lambda f:f["model_brier"]-f["reference_brier"]) if folds else None
        prediction.append({"Estilo":STYLES[style["style"]], "Señales de selección":style.get("selection_rows"),
                           "Pliegues superados":f"{sum(f['model_brier'] < f['reference_brier'] for f in folds)}/{len(folds)}" if folds else "Sin desglose",
                           "Brier modelo¹":round(worst["model_brier"],5) if worst else None,
                           "Brier referencia¹":round(worst["reference_brier"],5) if worst else None,
                           "Resultado":"Cumple" if style["baseline_pass"] else "Falla la comparación"})
        coverage = style["coverage"]
        intervals.append({"Estilo":STYLES[style["style"]], "Casos de calibración (2024)":style["calibration_rows"],
                          "Pronósticos de prueba (2025)":style["holdout"], "Cobertura observada":f"{coverage:.2%}",
                          "Objetivo mínimo":"80 %", "Diferencia (puntos porcentuales)":round(100*(coverage-.8),2),
                          "Resultado":"Cumple" if coverage >= .8 else "Por debajo"})
    st.dataframe(pd.DataFrame(prediction), hide_index=True, height="content")
    st.caption("¹ Ambos errores corresponden al pliegue con la mayor diferencia Brier del modelo menos Brier de referencia. Un empate también incumple el criterio.")
    for row in prediction:
        if row["Resultado"] == "Falla la comparación":
            st.write(f"**{row['Estilo']}** queda fuera de la política con filtro: el nivel 0 no mejoró la probabilidad constante de referencia en todos los pliegues.")
    st.subheader("Calibración del intervalo P10–P90")
    st.write("Los límites del intervalo se ajustaron entre octubre y diciembre de 2024. La cobertura indica qué proporción de retornos de los cinco contratos reservados quedó dentro del intervalo en 2025. El objetivo del ensayo es al menos el 80 % por estilo.")
    st.dataframe(pd.DataFrame(intervals), hide_index=True, height="content")
    for row in intervals:
        if row["Resultado"] == "Por debajo":
            gap = abs(row["Diferencia (puntos porcentuales)"])
            st.write(f"**{row['Estilo']}** quedó a {gap:.2f} puntos porcentuales del objetivo. "
                     "Esto mide cuántos retornos quedaron dentro del intervalo, independientemente de que fueran positivos o negativos.")
    audit = pd.read_csv(OUT/"audit_v3.csv")
    st.subheader("Coherencia entre temporalidades")
    st.write("El control compara precios y volumen al agregar velas completas. Los intervalos discrepantes se registran en cuarentena.")
    visual.horizontal_values(audit.groupby("timeframe").mismatched_bars.sum(),"Intervalos discrepantes")
    st.table(audit.groupby("timeframe")[["total_bars","missing_bars_in_grid","mismatched_bars"]].sum().astype(int).rename(
        columns={"total_bars":"Velas","missing_bars_in_grid":"Huecos","mismatched_bars":"Discrepancias"}).rename_axis("Temporalidad"))
    st.caption("La cuarentena conserva las velas originales para poder rastrear cada diferencia.")
    q = pd.read_csv(OUT/"quarantine_v3.csv")
    if not q.empty:
        st.subheader("Intervalos en cuarentena")
        st.dataframe(q.sort_values("date",ascending=False).rename(columns={
            "symbol":"Contrato","timeframe":"Escala","date":"Apertura UTC"}),height=210,hide_index=True)


def about():
    st.title("Acerca de QuantLab")
    st.write("QuantLab investiga si cinco especialistas pueden combinar sus señales para anticipar movimientos en futuros de criptomonedas.")
    st.write("La aplicación es un laboratorio histórico. Trabaja con archivos locales de 25 contratos perpetuos "
             "de Binance. Un contrato perpetuo sigue el precio de un activo y puede generar pagos periódicos "
             "de financiación entre posiciones largas y cortas. Aquí puedes reconstruir una decisión pasada "
             "y ver cuánto habría cambiado el capital al aplicar las reglas del ensayo.")
    st.subheader("Sigue una decisión desde el gráfico hasta la cartera")
    st.markdown("""
1. Abre **Mercado**, elige BTCUSDT, el intervalo de una hora y una vela de análisis. Cada vela reúne apertura, máximo, mínimo y cierre. Las barras inferiores indican el volumen. El porcentaje de la cabecera mide el cambio desde la apertura de la primera vela visible hasta el último cierre mostrado.
2. Lee la **Mesa de especialistas** junto al gráfico. Un especialista en espera ha detectado condiciones de interés. Una señal activa añade dirección, nivel de salida por pérdida y objetivo. En el gráfico de 15 minutos, la mesa utiliza la última vela horaria de decisión que ya cerró.
3. En **Escenarios** y **Especialistas**, elige el estilo correspondiente al intervalo. Táctico trabaja a una hora, swing corto a cuatro horas y posicional a un día. Revisa si los grupos se reproducen y cuántos ejemplos respaldan a cada experto.
4. Abre **Orquestador** para ver cómo se combinan los pronósticos de los expertos activos. La probabilidad estima si el resultado cubrirá los costes. Una candidata todavía debe pasar los filtros de cartera. Una abstención deja el capital disponible.
5. En **Cartera**, alterna entre la política con filtro de estabilidad y el control exploratorio. Compara el capital final, la caída desde el máximo y las operaciones. **Validación** explica qué condiciones cumple la corrida y qué evidencia sigue pendiente.
""")
    st.subheader("Cómo comparar Simulador, Rendimiento y Cartera")
    st.write("Simulador reconstruye los episodios históricos por eventos desde el cierre de la vela de decisión. Un plan puede resolverse, quedar filtrado, "
             "liquidarse o terminar sin desenlace observado. La unidad R expresa el resultado en múltiplos del "
             "riesgo inicial del plan.")
    st.write("En Rendimiento, la prueba reservada compara las señales durante una vela completa, con entrada "
             "en la apertura siguiente a la decisión. La comparación asigna 1.000 USDT a cada señal y descuenta "
             "comisiones y deslizamiento. Si dos expertos coinciden, la señal figura en el historial de ambos.")
    st.write("Cartera parte de 10.000 USDT compartidos. Admite hasta tres entradas por semana y una posición "
             "abierta por contrato. Aplica la financiación observada y valora las posiciones cada hora. "
             "Así, un experto puede acumular señales en Rendimiento mientras la cartera sigue en efectivo.")
    st.subheader("Cómo interpretar las métricas con un ejemplo")
    st.markdown("""
- **Brier** mide el error de las probabilidades. Si un experto estima un 70 % y el resultado es favorable, el error de ese caso es (0,70 − 1)² = 0,09. Se promedian los errores de todos los casos. Conviene leer el valor junto al tamaño de muestra.
- **P10–P90** delimita la franja central prevista para el retorno. Si 80 de cada 100 resultados observados caen dentro, la cobertura es del 80 %. Un retorno negativo también cuenta como cubierto si cae dentro de esa franja.
- **ARI** mide cuánto se conserva la agrupación de las observaciones al repetir el ajuste. El umbral de estabilidad del proyecto es 0,70. Para entender los indicadores que alimentan estos grupos, consulta **Guía de técnicas**.
- **Caída máxima** mide el peor retroceso del patrimonio desde un máximo anterior. Por ejemplo, bajar de 10.000 a 9.200 USDT supone una caída del 8 %.
""")
    st.subheader("Arquitectura del proyecto")
    visual.architecture()
    st.subheader("Reglas de los cinco especialistas")
    st.dataframe(desk.configuration(),hide_index=True,row_height=84,height="content",
                 column_config={"Qué busca":st.column_config.TextColumn(width="large"),
                                "Cuándo se activa":st.column_config.TextColumn(width="large")})
    st.caption("ATR mide el rango típico en proporción al precio. Cada plan incluye una distancia mínima para el stop y el objetivo.")
    run = generation()
    result = json.loads((run/"report.json").read_text(encoding="utf-8"))
    sample_count = sum(r["holdout"] for r in result["styles"])
    metrics([("Especialistas",len(desk.NAMES)),("Estilos",len(STYLES)),
             ("Pronósticos reservados",f"{sample_count:,}"),("Periodo de prueba","2025")])
    st.subheader("Qué especialista estimó mejor las probabilidades")
    sample = pd.DataFrame(result["experts"]).dropna(subset=["unseen_brier"])
    if not sample.empty:
        sample["Especialista"] = sample.expert.map(desk.NAMES)
        sample["Estilo"] = sample["style"].map(STYLES)
        best = sample.loc[sample.groupby("Estilo").unseen_brier.idxmin(),["Estilo","Especialista","unseen_brier","unseen_rows"]]
        st.write("El menor error Brier identifica la mejor estimación de probabilidad dentro de cada estilo. La comparación de ganancias por moneda está en Rendimiento.")
        graph,leaders = st.columns([1.2,1],border=True)
        with graph:
            st.bar_chart(sample.pivot(index="Especialista",columns="Estilo",values="unseen_brier"),stack=False,
                         height=235,y_label="Error Brier fuera de activo")
        with leaders:
            st.markdown("**Menor error por estilo**")
            st.table(best.set_index("Estilo").rename(columns={"unseen_brier":"Brier","unseen_rows":"Muestra"}).style.format({"Brier":"{:.3f}"}))
    st.subheader("Resultado de la prueba reservada")
    coverage = pd.DataFrame(result["styles"])
    coverage["Estilo"] = coverage["style"].map(STYLES)
    st.table(coverage.set_index("Estilo")[["holdout","coverage","ari"]].rename(columns={
        "holdout":"Pronósticos","coverage":"Cobertura P10–P90","ari":"Estabilidad ARI"}).style.format({
        "Cobertura P10–P90":"{:.1%}","Estabilidad ARI":"{:.3f}"}))
    st.write(f"Operaciones de la política con filtro: **{result['trades']}**. " + " ".join(result["reasons"]))
    st.subheader("Quién obtuvo el mayor retorno medio por señal")
    winners = []
    for style in STYLES:
        episodes = ROOT/f"data/episodes/v3_episodes_{style}.parquet"
        observations = desk.performance_data(style,"Prueba reservada 2025",str(run),episodes.stat().st_mtime_ns)
        grouped = observations.groupby("expert_name").net.agg(["mean","size"])
        if not grouped.empty:
            leader = grouped["mean"].idxmax()
            winners.append({"Estilo":STYLES[style],"Especialista":desk.NAMES[leader],
                            "Señales":int(grouped.loc[leader,"size"]),"Retorno medio":grouped.loc[leader,"mean"]})
    if winners:
        st.table(pd.DataFrame(winners).set_index("Estilo").style.format({"Retorno medio":"{:+.2%}"}))
        st.caption("Se comparan todas las señales de los cinco activos reservados con un horizonte de una vela, tras costes fijos y antes de financiación. El mayor retorno puede seguir siendo negativo.")

