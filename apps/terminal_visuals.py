"""Presentation components for the QuantLab research terminal."""
from html import escape

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

TEAL = "#238778"
RUST = "#B85C43"


def stat_strip(items):
    cells = []
    for label, value, tone in items:
        cells.append(f'<div><dt>{escape(label)}</dt><dd class="{tone}">{escape(str(value))}</dd></div>')
    st.html('<dl class="ql-stats">' + "".join(cells) + '</dl>')


def quote_chart(symbol, tf, bars):
    last = bars.iloc[-1]
    change = last.close / bars.open.iloc[0] - 1
    direction = "up" if change >= 0 else "down"
    st.html(f"""<div class="ql-quote">
        <div><span>{escape(symbol)} · {escape(tf)} · USDT</span>
        <strong>{last.close:,.7g}</strong></div>
        <div><span>Variación · {len(bars)} velas</span>
        <b class="{direction}">{change:+.2%}</b></div>
        <div><span>Mínimo / máximo</span>
        <b>{bars.low.min():,.6g} / {bars.high.max():,.6g}</b></div>
        </div>""")
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.055,row_heights=[.77,.23])
    fig.add_trace(go.Candlestick(x=bars.date,open=bars.open,high=bars.high,low=bars.low,close=bars.close,
        increasing=dict(line=dict(color=TEAL,width=1),fillcolor=TEAL),
        decreasing=dict(line=dict(color=RUST,width=1),fillcolor=RUST),name="Precio"),row=1,col=1)
    fig.add_trace(go.Bar(x=bars.date,y=bars.volume,marker_color=[
        TEAL if c>=o else RUST for c,o in zip(bars.close,bars.open)],opacity=.85,name="Volumen"),row=2,col=1)
    fig.add_hline(y=last.close,line_color="#9DACBB",line_width=1,line_dash="dot",row=1,col=1)
    fig.update_layout(height=345,margin=dict(l=9,r=13,t=8,b=26),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(size=11),
                      showlegend=False,hovermode="x unified",dragmode="zoom",bargap=.15)
    fig.update_xaxes(rangeslider_visible=False,showgrid=False,nticks=5)
    fig.update_yaxes(zeroline=False,nticks=4,side="right")
    fig.update_yaxes(title_text="Vol.",nticks=2,row=2,col=1)
    st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"displayModeBar":"hover","scrollZoom":False},
                    alt=f"Precio y volumen de {symbol}, {len(bars)} velas de {tf}, al {last.date:%d/%m/%Y %H:%M} UTC.")


def signal_list(rows):
    entries = []
    for r in rows:
        side = r.get("Dirección","Neutral")
        state = r["Estado"]
        tone = "flat" if state != "Señal activa" or side=="Neutral" else ("up" if side=="Largo" else "down")
        glyph = {"up":"↗", "down":"↘","flat":"—"}[tone]
        entries.append(f'<li><div><b>{escape(r["Especialista"])}</b><small>{escape(state)}</small></div>'
                       f'<span class="{tone}">{glyph} {escape(side)}</span></li>')
    st.html('<ul class="ql-signals">'+"".join(entries)+'</ul>')


def horizontal_values(values, unit, *, reference=None, signed=False):
    """Compare a short set of labelled measures against zero or a common threshold."""
    series = values.dropna().sort_values()
    if series.empty:
        st.caption("Sin observaciones para esta comparación.")
        return
    colors = [RUST if signed and x < 0 else TEAL for x in series]
    fig = go.Figure(go.Bar(y=series.index,x=series.values,orientation="h",marker_color=colors,
                           hovertemplate="%{y} · %{x:.3f}<extra></extra>"))
    if reference is not None:
        fig.add_vline(x=reference,line_color="#738789",line_dash="dot",annotation_text=f"Umbral {reference:g}")
    fig.update_layout(height=max(190,len(series)*34+70),margin=dict(l=0,r=30,t=22,b=0),
                      xaxis_title=unit,yaxis_title=None,showlegend=False,bargap=.48)
    fig.update_xaxes(rangemode="tozero")
    st.plotly_chart(fig,width="stretch",alt=f"{unit} por categoría",config={"displayModeBar":False})


def architecture():
    stages = [
        ("Mercado","Velas OHLCV y financiación","Control de huecos, duplicados y coherencia entre 15 min, 1 h, 4 h y 1 día."),
        ("Indicadores","Eventos CUSUM","Medias, RSI, ATR y volumen convierten movimientos de precio en oportunidades de análisis."),
        ("Escenarios","Representación y estabilidad","El autoencoder comprime ventanas. ARI mide si los grupos se reproducen y el detector señala estructuras desconocidas."),
        ("Especialistas","Cinco planes de operación","Cada regla devuelve estado, dirección, stop y objetivo. El replay por eventos sigue el plan hasta su cierre."),
        ("Entrenamiento","Clasificación por experto","El ensayo usa hasta 55 iteraciones y 12 hojas. Cuatro grupos de activos evalúan transferencia; el benchmark original compara HGB, TCN, GRU y referencia nula."),
        ("Orquestador","Probabilidad e intervalo","Combina expertos con pesos uniformes, jerárquicos o neuronales. La validación decide el nivel y calibra P10–P90."),
        ("Cartera","Presupuesto y valoración","10.000 USDT iniciales, hasta tres entradas por semana y valoración horaria con costes y financiación disponible."),
        ("Validación","Decisión de despliegue","Contrasta rendimiento neto, cobertura, estabilidad, muestra y caída máxima. Una señal puede terminar en abstención.")
    ]
    cards = "".join(f'<li><small>Etapa {i} · {name}</small><b>{output}</b><p>{note}</p></li>'
                    for i,(name,output,note) in enumerate(stages,1))
    st.html('<ol class="ql-flow" aria-label="Arquitectura en ocho etapas">'+cards+'</ol>')
    st.subheader("Calendario del ensayo reservado")
    periods = [("Hasta 2022","Ajuste de expertos"),("2023","Competencia por régimen"),
               ("Ene–sep 2024","Selección de nivel"),("Oct–dic 2024","Calibración P10–P90"),("2025","Prueba en cinco activos")]
    st.html('<ol class="ql-periods">'+ "".join(f'<li><b>{t}</b><span>{s}</span></li>' for t,s in periods)+'</ol>')
    st.caption("El ensayo usa un horizonte de una vela y tres grupos de indicadores. Su estabilidad se contrasta con cinco remuestreos sobre observaciones fijas de 2024.")


def terminal_style():
    st.html("""<style>
    .ql-quote,.ql-stats,.ql-signals,.ql-flow,.ql-periods{font-family:inherit;font-variant-numeric:tabular-nums;box-sizing:border-box}
    .ql-quote{background:transparent;color:inherit;border:1px solid #81949265;border-bottom:0;display:flex;align-items:flex-end;flex-wrap:wrap;gap:12px 28px;padding:16px 18px 12px;border-radius:6px 6px 0 0}
    .ql-quote div,.ql-stats div{display:grid;gap:4px}
    .ql-quote span{font-size:.77rem;color:inherit}.ql-quote strong{font-size:clamp(1.55rem,2.1vw,2rem);line-height:1.08;letter-spacing:-.035em}
    .ql-quote b{font-size:.94rem}.ql-quote .up{color:#278175}.ql-quote .down{color:#bb5941}
    .ql-stats{display:flex;flex-wrap:wrap;gap:16px 30px;margin:0;border-block:1px solid #81949265;padding:13px 0}
    .ql-stats div{flex:1 1 130px}.ql-stats dt{font-size:.83rem;color:inherit}.ql-stats dd{margin:0;font-weight:600;font-size:clamp(1.05rem,1.75vw,1.55rem);line-height:1.24}
    .ql-stats .positive,.ql-signals .up{color:#278175}.ql-stats .negative,.ql-signals .down{color:#bb5941}
    .ql-signals{padding:0;margin:0;list-style:none}.ql-signals li{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:9px 0;border-bottom:1px solid #81949265;margin:0}
    .ql-signals b,.ql-signals small{display:block}.ql-signals b{font-size:.95rem}.ql-signals small{font-size:.77rem;color:inherit}
    .ql-signals span{font-weight:600;font-size:.88rem}.ql-signals .flat{color:inherit}
    .ql-flow,.ql-periods{padding:0;margin:0;list-style:none;display:grid;gap:0;background:transparent;border:1px solid #81949265}
    .ql-flow{grid-template-columns:repeat(4,minmax(0,1fr))}.ql-flow li,.ql-periods li{margin:0;padding:15px 14px;border-right:1px solid #81949265}
    .ql-flow li{position:relative;border-top:3px solid #11685d}.ql-flow li:nth-child(n+5){border-top-color:#19324d}
    .ql-flow li:not(:nth-child(4n))::after{content:"›";position:absolute;right:5px;top:10px;color:#587677;font-size:1.25rem}
    .ql-flow small{color:inherit;font-weight:600}.ql-flow b,.ql-periods span{display:block}.ql-flow b{margin:.4rem 0;line-height:1.3}
    .ql-flow p{margin:0;font-size:.87rem;line-height:1.45;color:inherit}
    .ql-periods{grid-template-columns:repeat(5,minmax(0,1fr));background:transparent}.ql-periods span{font-size:.84rem}.ql-periods li:last-child{background:#19324d;color:white}
    @media(max-width:1100px){.ql-flow{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:650px){.ql-flow,.ql-periods{grid-template-columns:1fr}.ql-flow li,.ql-periods li{padding:10px 12px}.ql-periods li{border-bottom:1px solid #c9d9d5}.ql-stats div{flex-basis:100px}.ql-quote{gap:10px 16px}}
    </style>""")

