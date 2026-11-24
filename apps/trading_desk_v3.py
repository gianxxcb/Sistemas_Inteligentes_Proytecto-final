"""Local research terminal. All clocks are UTC and all quotes come from the snapshot."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.experts import EXPERT_CLASSES
from apps import terminal_visuals as visual

ROOT = Path(__file__).resolve().parents[1]
TF = {"tactical":"1h", "swing_short":"4h", "swing_positional":"1d"}
NAMES = {"TrendExpert":"Tendencia", "ReversalExpert":"Reversión", "BreakoutExpert":"Ruptura",
         "VolatilityExpert":"Volatilidad", "MomentumExpert":"Impulso"}
RULES = {
    "TrendExpert": ("ADX ≥ 22, distancia a SMA50 superior al 1 % y pendiente a favor.",
                    "Retroceso a menos de 3,5 % de SMA20 con MACD a favor.", "1,5 ATR (mín. 3 %)", "2,5 ATR (mín. 6 %)"),
    "ReversalExpert": ("RSI ≤ 33 o ≥ 67, o extremo en Bollinger o distancia a SMA20.",
                       "Regreso desde el extremo con confirmación estocástica.", "1,6 ATR (mín. 4 %)", "Distancia a SMA20 (mín. 4 %)"),
    "BreakoutExpert": ("Vigila compresión y los dos bordes del canal Donchian de 20 velas.",
                       "Precio a menos del 1,2 % del borde y volumen relativo ≥ 1,15.", "1,5 ATR (mín. 3,5 %)", "2,8 ATR (mín. 6 %)"),
    "VolatilityExpert": ("Compresión de Bollinger o expansión de volatilidad relativa.",
                         "Expansión tras compresión; sesgo direccional a más del 2 % de SMA20.", "1,5 ATR (mín. 4 %)", "3 ATR (mín. 8 %)"),
    "MomentumExpert": ("ROC de 10 y 5 velas en la misma dirección, por encima de 3 % y 1 % en magnitud.",
                       "Flujo comprador o vendedor confirmado por taker ratio u OBV.", "1,4 ATR (mín. 3 %)", "2,5 ATR (mín. 6 %)"),
}


def configuration():
    return pd.DataFrame([(NAMES[k], *v) for k,v in RULES.items()], columns=[
        "Especialista","Qué busca","Cuándo se activa","Distancia al stop","Distancia al objetivo"])


def feature_path(tf):
    return ROOT/f"data/features/crypto/features_{tf}.parquet"


def signal_board(symbol, tf, timestamp):
    # At 15m the desk shows the most recent completed tactical decision candle.
    interval = "1h" if tf == "15m" else tf
    cutoff = timestamp + pd.Timedelta(tf) - pd.Timedelta(interval)
    close = timestamp + pd.Timedelta(tf)
    spread_file = ROOT/"data/processed/v3_microstructure/spread_proxy.parquet"
    if spread_file.exists():
        spread = pd.read_parquet(spread_file, columns=["bid_ask_spread_bps"],
                                 filters=[("symbol","=",symbol),("available_at","=",close)])
        if not spread.empty:
            st.caption(f"Spread estimado al cierre · {spread.bid_ask_spread_bps.iloc[0]:.2f} pb "
                       "(proxy de rango de 15 min calibrado con bid/ask)")
    path = feature_path(interval)
    frame = last_observation(str(path), path.stat().st_mtime_ns, symbol, cutoff)
    if frame.empty:
        st.info("No hay una vela de decisión cerrada para este momento.")
        return
    row = frame.iloc[0]
    st.markdown("#### Mesa de especialistas")
    st.caption(f"Decisión {interval} · {(row.date+pd.Timedelta(interval)):%d/%m %H:%M} UTC")
    rows = []
    for cls in EXPERT_CLASSES:
        expert = cls(version="V1")
        required = set(expert.get_confirmation_feature_names()) | {"atr_relative","dist_to_sma_20","close"}
        if row.reindex(list(required)).isna().any():
            rows.append({"Especialista":NAMES[expert.name],"Estado":"Datos insuficientes"})
            continue
        plan = expert.evaluate_playbook(row)
        side = {"largo":1,"corto":-1}.get(plan["direccion"],0)
        active = side and plan["estado"] != "inactivo"
        rows.append({"Especialista":NAMES[expert.name],"Estado":{"disparado":"Señal activa","setup":"En espera","inactivo":"Inactivo"}[plan["estado"]],
                     "Dirección":plan["direccion"].capitalize(), "Stop (USDT)":float(row.close)*(1-side*abs(plan["invalidacion"])) if active else None,
                     "Objetivo (USDT)":float(row.close)*(1+side*abs(plan["objetivo"])) if active else None,
                     "Lectura de la vela":" · ".join(plan["traza_reglas"])})
    visual.signal_list(rows)
    style = next(s for s, scale in TF.items() if scale == interval)
    assignment = ROOT/f"experiments/v3/representation/scenario_recheck/{style}/scenario_assignments_{style}.parquet"
    observations = pd.read_parquet(assignment,columns=["date","partition","primary_scenario","scenario_models_eligible"],
                                   filters=[("symbol","=",symbol)])
    observations["date"] = pd.to_datetime(observations.date,utc=True)
    training = json.loads((ROOT/"experiments/v3/representation/scenario_recheck/summary.json").read_text(encoding="utf8"))
    cutoff = pd.Timestamp(next(item["train_end"] for item in training["styles"] if item["style"] == style))
    known = observations[(observations.date <= row.date) & (observations.date > cutoff)].sort_values("date")
    regimes = pd.read_parquet(ROOT/"experiments/v3/representation/scenario_recheck/daily_regime.parquet")
    known_regimes = regimes.loc[pd.Timestamp("2023-01-01",tz="UTC"):row.date+pd.Timedelta(interval)]
    if not known_regimes.empty:
        state = int(known_regimes.regime.iloc[-1])
        st.caption(f"Régimen diario de BTC · {['Bajista','Neutro','Alcista'][state]}")
    if not known.empty:
        last = known.iloc[-1]
        label = "Estructura desconocida" if last.primary_scenario < 0 else f"Grupo {last.primary_scenario}"
        eligible = "Estabilidad suficiente" if last.scenario_models_eligible else "Estabilidad insuficiente"
        st.caption(f"Último escenario · {last.date:%d/%m/%Y %H:%M} UTC")
        st.markdown(f"**{label}** · {eligible} en la revisión retrospectiva")
    else:
        st.caption("Este contrato aún no tenía un escenario clasificado en la fecha seleccionada.")
    return pd.DataFrame(rows)


@st.cache_data(max_entries=64)
def last_observation(path, revision, symbol, cutoff):
    # The bounded time range enables Parquet predicate pushdown on large files.
    filters = [("symbol","=",symbol),("date","<=",cutoff),("date",">=",cutoff-pd.Timedelta(days=7))]
    return pd.read_parquet(path, filters=filters).sort_values("date").tail(1)


@st.cache_data(max_entries=12)
def performance_data(style, source, run, revision):
    path = ROOT/f"data/episodes/v3_episodes_{style}.parquet"
    keys = ["symbol","decision_date","direction","expert_name"]
    if source == "Prueba reservada 2025":
        votes = pd.read_parquet(path, columns=keys).drop_duplicates()
        f = pd.read_parquet(Path(run)/"forecasts.parquet",filters=[("style","=",style)])
        f["direction"] = f.sign.map({1:"largo",-1:"corto"})
        return f.rename(columns={"date":"decision_date"}).merge(votes,on=keys[:3],validate="one_to_many")[
            keys+["exit_date","net"]]
    path = replay_path(style)
    df = pd.read_parquet(path,columns=keys+["exit_date","net_return","outcome_status"])
    return df[df.outcome_status.isin(["resolved","liquidated"])].dropna(subset=["net_return","exit_date"]).rename(columns={"net_return":"net"})


def replay_path(style):
    return ROOT/f"data/episodes/causal_v3/v3_episodes_{style}.parquet"


def returns_panel(style, run):
    source = st.radio("Fuente de resultados",["Prueba reservada 2025","Replay causal · 25 contratos"],horizontal=True)
    path = ROOT/f"data/episodes/v3_episodes_{style}.parquet"
    if source.startswith("Replay"):
        path = replay_path(style)
    frame = performance_data(style,source,str(run),path.stat().st_mtime_ns)
    st.caption("Prueba reservada: retorno de una vela tras comisión y deslizamiento, antes de financiación. Una señal compartida cuenta en el historial de cada experto." if source.startswith("Prueba") else
               "Replay por eventos con entrada desde el cierre de la vela de decisión, incluidos los episodios liquidados. Los costes incluyen comisiones, deslizamiento y financiación.")
    left,right = st.columns(2)
    symbols = sorted(frame.symbol.unique())
    symbol = left.selectbox("Moneda",["Todas",*symbols])
    if symbol != "Todas":
        frame = frame[frame.symbol==symbol]
    if frame.empty:
        st.info("No hay episodios con resultado para esta selección.")
        return
    ranking = frame.groupby("expert_name").net.agg(Muestras="size",Promedio="mean",Acierto=lambda x:(x>0).mean()).sort_values("Promedio",ascending=False)
    expert = right.selectbox("Especialista",ranking.index,format_func=NAMES.get)
    selected = frame[frame.expert_name==expert].sort_values("exit_date")
    pnl = selected.net * 1000
    total = pnl.sum()
    visual.stat_strip([("Resultado acumulado",f"{total:+,.2f} USDT","positive" if total>=0 else "negative"),
                       ("Ganancias brutas",f"{pnl.clip(lower=0).sum():,.2f} USDT","positive"),
                       ("Pérdidas brutas",f"{pnl.clip(upper=0).sum():,.2f} USDT","negative")])
    st.caption("Comparación hipotética con 1.000 USDT de exposición fija por señal. Puede haber señales simultáneas. La cartera con límites y capital compartido se consulta en Cartera.")
    series = pnl.groupby(selected.exit_date.dt.floor("D")).sum().cumsum().rename("Resultado acumulado (USDT)")
    history,comparison = st.columns([1.5,1],border=True)
    with history:
        st.subheader("Resultado por día de cierre")
        st.caption(f"{NAMES[expert]} · {len(selected):,} señales")
        st.line_chart(series,color=visual.TEAL if total>=0 else visual.RUST,height=240,y_label="USDT acumulados")
    with comparison:
        st.subheader("Retorno medio por señal")
        visual.horizontal_values(ranking.Promedio.rename(index=NAMES)*100,"Retorno (%)",signed=True)
    st.subheader("Comparación de especialistas")
    top = ranking.iloc[0]
    st.write(f"**{NAMES[ranking.index[0]]}** obtuvo el mayor retorno medio por señal en esta selección: **{top.Promedio:+.2%}** sobre {int(top.Muestras):,} observaciones.")
    view = ranking.rename(index=NAMES).rename_axis("Especialista").reset_index()
    st.table(view.set_index("Especialista").style.format({"Promedio":"{:+.2%}","Acierto":"{:.1%}"}))
    st.subheader("Últimos resultados del especialista")
    st.dataframe(selected.tail(15).iloc[::-1].assign(**{"Resultado (USDT)":pnl}).rename(columns={
        "symbol":"Moneda","decision_date":"Vela UTC","exit_date":"Salida UTC","direction":"Dirección"})[
            ["Moneda","Vela UTC","Salida UTC","Dirección","Resultado (USDT)"]],hide_index=True)


@st.cache_data(max_entries=6)
def scenario_profiles(style, assignment_file, feature_file, revisions):
    cols = ["date","symbol","primary_scenario"]
    groups = pd.read_parquet(assignment_file,columns=cols)
    groups["date"] = pd.to_datetime(groups.date,utc=True)
    indicators = pd.read_parquet(feature_file,columns=["date","symbol","dist_to_sma_50","atr_relative","rsi_14"],
                                 filters=[("date",">=",groups.date.min()),("date","<=",groups.date.max())])
    joined = groups.merge(indicators,on=["date","symbol"],how="left",validate="many_to_one")
    table = joined.groupby("primary_scenario").agg(Ventanas=("symbol","size"),Activos=("symbol","nunique"),
          RSI=("rsi_14","median"),ATR=("atr_relative","median"),Media=("dist_to_sma_50","median"))
    descriptions = []
    for idx,r in table.iterrows():
        if idx == -1:
            descriptions.append("Estructura desconocida. La ventana cae fuera de los grupos reconocidos.")
        elif pd.isna(r.Media):
            descriptions.append("Sin indicadores coincidentes para describir el grupo.")
        else:
            position = "encima" if r.Media >= 0 else "debajo"
            descriptions.append(f"Precio {abs(r.Media):.1%} por {position} de la SMA50; RSI {r.RSI:.0f} y rango ATR del {r.ATR:.1%} en la ventana mediana.")
    table["Perfil observado"] = descriptions
    table.index = ["Fuera de distribución" if x==-1 else f"Grupo {x}" for x in table.index]
    return table


def catalog(style):
    folder = ROOT/f"experiments/v3/representation/scenario_recheck/{style}"
    a = folder/f"scenario_assignments_{style}.parquet"
    f = feature_path(TF[style])
    table = scenario_profiles(style,a,f,(a.stat().st_mtime_ns,f.stat().st_mtime_ns))
    st.subheader("Catálogo de escenarios del autoencoder")
    st.caption("Cada perfil resume las velas asignadas a ese grupo en este estilo. Los números de grupo son propios de cada modelo.")
    st.table(table[["Perfil observado","Ventanas","Activos"]].rename_axis("Escenario"))
    st.bar_chart(table.Ventanas,height=190,color=visual.TEAL,y_label="Ventanas asignadas")

