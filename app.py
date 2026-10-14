import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# =========================
# CONFIGURAÇÃO DO APLICATIVO
# =========================
st.set_page_config(
    page_title="WiFiLab Pro",
    page_icon="📶",
    layout="wide"
)

# Estilização limpa, profissional e iluminada (Fundo claro e moderno)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp { 
            background-color: #f8fafc; 
            color: #1e293b; 
            font-family: 'Inter', sans-serif;
        }
        .block-container { padding-top: 1.5rem; }
        
        [data-testid='stSidebar'] { 
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0; 
        }
        [data-testid='stSidebar'] * { color: #334155; }
        
        .metric-card { 
            background: #ffffff; 
            border: 1px solid #e2e8f0; 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 15px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .commercial-card { 
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
            border: 2px solid #0284c7; 
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 15px; 
        }

        .router-card {
            background: #ffffff;
            border-left: 5px solid #0284c7;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #0284c7 0%, #06b6d4 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOGO OFICIAL & CRÉDITOS NA SIDEBAR
# =========================
try:
    st.sidebar.image("WA_1791973476738.png", use_container_width=True)
except Exception:
    st.sidebar.markdown(
        "<div style='text-align: center; padding: 15px; background: #e0f2fe; border-radius: 12px; border: 1px solid #0284c7; margin-bottom: 15px;'>"
        "<h2 style='color: #0284c7; margin: 0;'>WiFiLab Pro</h2>"
        "</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    "<div style='text-align: center; padding: 8px 0 15px 0;'>"
    "<p style='font-size: 14px; color: #1e293b; margin: 0; font-weight: bold;'>Criado por Dayrlane Sousa</p>"
    "<p style='font-size: 11px; color: #0284c7; margin: 2px 0 0 0;'>TCC Engenharia de Redes</p>"
    "</div>",
    unsafe_allow_html=True
)

# =========================
# DADOS DE REFERÊNCIA DA RNA
# =========================
X_REF = np.array([[1.5, 0], [3.0, 0], [4.0, 0], [5.5, 1], [6.5, 2], [5.0, 2], [8.0, 3]], dtype=float)
Y_REF = np.array([-40, -55, -60, -68, -72, -75, -85], dtype=float)

@st.cache_resource
def treinar_rna():
    modelo = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8, 4), activation="relu", solver="lbfgs", max_iter=2000, random_state=42))
    modelo.fit(X_REF, Y_REF)
    return modelo

rna = treinar_rna()

def estimar_rssi(distancia, obstaculos):
    entrada = np.array([[max(float(distancia), 0.1), max(int(obstaculos), 0)]])
    return float(rna.predict(entrada)[0])

def classificar_qualidade(rssi):
    if rssi >= -50:
        return "🟢 Excelente (Jogos e 4K perfeitos)"
    elif rssi >= -60:
        return "🟢 Bom (Estável para o dia a dia)"
    elif rssi >= -67:
        return "🟡 Razoável (Navegação Padrão)"
    elif rssi >= -75:
        return "🟠 Fraco (Pode travar)"
    return "🔴 Muito Fraco (Zona de Sombra)"

def estimar_velocidade(rssi, plano, freq, num_wifi):
    f_ator = 1.0 if "5GHz" in freq else 0.85
    congest_ator = max(0.6, 1.0 - (num_wifi * 0.02))
    
    if rssi >= -50: vel = plano
    elif rssi >= -60: vel = int(plano * 0.8)
    elif rssi >= -67: vel = int(plano * 0.5)
    elif rssi >= -75: vel = int(plano * 0.2)
    else: vel = int(plano * 0.05)
    
    return max(5, int(vel * f_ator * congest_ator))

ARQUIVO_HISTORICO = Path("historico_wifilab.csv")

def carregar_historico():
    if ARQUIVO_HISTORICO.exists(): return pd.read_csv(ARQUIVO_HISTORICO)
    return pd.DataFrame()

def salvar_atendimento(registro):
    historico = carregar_historico()
    historico = pd.concat([historico, pd.DataFrame([registro])], ignore_index=True)
    historico.to_csv(ARQUIVO_HISTORICO, index=False, encoding="utf-8-sig")

# =========================
# MENU PRINCIPAL
# =========================
pagina = st.sidebar.radio(
    "MENU PRINCIPAL",
    [
        "Simulador Visual de Planta & Casa",
        "Consultoria Comercial & Upgrade Mesh (ISP)",
        "Catálogo Oficial de Fabricantes & LEDs",
        "Suporte Técnico & Histórico de O.S."
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab Pro | Plataforma Oficial ISPs")

# =========================
# 1. SIMULADOR VISUAL DE PLANTA & CASA
# =========================
if pagina == "Simulador Visual de Planta & Casa":
    st.subheader("🏡 Maquete Interativa da Residência & Distribuição de Sinal")
    st.write(
        "Configure o cenário da residência (localização do roteador, quantidade de dispositivos conectados via Wi-Fi/Cabo "
        "e a frequência utilizada) para simular o comportamento real do sinal de internet na planta baixa."
    )

    with st.form("form_simulador_avancado"):
        c_op1, c_op2, c_op3 = st.columns(3)
        with c_op1:
            local_roteador = st.selectbox(
                "📍 Posição do Roteador",
                ["Sala", "Cozinha", "Quarto", "Banheiro", "Pátio", "2º Andar"]
            )
            plano_contratado = st.number_input("🚀 Plano Contratado (Mbps)", value=500, step=50)
        with c_op2:
            num_wifi = st.selectbox(
                "📱 Dispositivos Wi-Fi Conectados",
                ["1", "2", "3", "4", "5", "6", "Até 15"]
            )
            freq_wifi = st.selectbox("📡 Frequência do Wi-Fi", ["5GHz (Maior velocidade)", "2.4GHz (Maior alcance)"])
        with c_op3:
            tem_cabo = st.radio("💻 Possui aparelhos via Cabo?", ["Não", "Sim"])
            num_cabo = st.selectbox("Qtd. de Aparelhos via Cabo", [str(i) for i in range(1, 11)]) if tem_cabo == "Sim" else "0"

        btn_rodar = st.form_submit_button("Gerar Simulação na Planta Baixa")

    n_wifi_val = 15 if num_wifi == "Até 15" else int(num_wifi)
    n_cabo_val = int(num_cabo)

    coords_map = {
        "Sala": (2.0, 6.0),
        "Cozinha": (8.0, 6.0),
        "Quarto": (2.0, 2.0),
        "Banheiro": (5.0, 4.0),
        "Pátio": (8.0, 1.0),
        "2º Andar": (5.0, 7.5)
    }
    rx, ry = coords_map.get(local_roteador, (2.0, 6.0))

    fig_casa = go.Figure()

    fig_casa.add_shape(type="rect", x0=0, y0=4, x1=5, y1=8, fillcolor="#e0f2fe", opacity=0.5, line=dict(color="#0284c7", width=3))
    fig_casa.add_annotation(x=2.5, y=6.0, text="<b>Sala</b>", showarrow=False, font=dict(color="#0369a1", size=12))

    fig_casa.add_shape(type="rect", x0=5, y0=4, x1=10, y1=8, fillcolor="#fef3c7", opacity=0.5, line=dict(color="#d97706", width=3))
    fig_casa.add_annotation(x=7.5, y=6.0, text="<b>Cozinha</b>", showarrow=False, font=dict(color="#92400e", size=12))

    fig_casa.add_shape(type="rect", x0=0, y0=0, x1=5, y1=4, fillcolor="#f1f5f9", opacity=0.5, line=dict(color="#64748b", width=3))
    fig_casa.add_annotation(x=2.5, y=2.0, text="<b>Quarto</b>", showarrow=False, font=dict(color="#334155", size=12))

    fig_casa.add_shape(type="rect", x0=5, y0=0, x1=10, y1=4, fillcolor="#fce7f3", opacity=0.5, line=dict(color="#db2777", width=3))
    fig_casa.add_annotation(x=7.5, y=2.0, text="<b>Banheiro / Pátio</b>", showarrow=False, font=dict(color="#9d174d", size=12))

    fig_casa.add_trace(go.Scatter(
        x=[rx], y=[ry],
        mode='markers+text',
        marker=dict(size=28, color='#0284c7', symbol='square'),
        text=["<b>📡 [ Roteador Wi-Fi ]</b>"],
        textposition="top center"
    ))

    fig_casa.update_layout(
        title=f"Planta Baixa — Roteador posicionado em: {local_roteador}",
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 8.5], showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_casa, use_container_width=True)

    st.markdown("### 📊 Diagnóstico de Cobertura por Ambiente")
    
    ambientes = [
        {"nome": "Sala", "x": 2.5, "y": 6.0, "paredes": 0 if local_roteador == "Sala" else 1},
        {"nome": "Cozinha", "x": 7.5, "y": 6.0, "paredes": 0 if local_roteador == "Cozinha" else 1},
        {"nome": "Quarto", "x": 2.5, "y": 2.0, "paredes": 1 if local_roteador == "Quarto" else 2},
        {"nome": "Banheiro / Pátio", "x": 7.5, "y": 2.0, "paredes": 2},
        {"nome": "2º Andar / Superior", "x": 5.0, "y": 7.5, "paredes": 1 if local_roteador == "2º Andar" else 2},
    ]

    col1, col2 = st.columns(2)
    for i, amb in enumerate(ambientes):
        dist = np.sqrt((amb["x"] - rx)**2 + (amb["y"] - ry)**2)
        rssi = estimar_rssi(dist, amb["paredes"])
        vel_util = estimar_velocidade(rssi, plano_contratado, freq_wifi, n_wifi_val)
        qualidade = classificar_qualidade(rssi)

        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h4 style='margin-top: 0;'>📍 {amb['nome']}</h4>
                    <p style='margin: 4px 0;'><b>Qualidade:</b> {qualidade}</p>
                    <p style='margin: 4px 0;'><b>Velocidade Real:</b> <span style='color: #0284c7; font-size: 18px; font-weight: bold;'>{vel_util} Mbps</span> / {plano_contratado} Mbps</p>
                    <p style='margin: 4px 0; font-size: 12px; color: #64748b;'>Distância: {dist:.1f}m | Barreiras: {amb['paredes']} paredes | Wi-Fi: {n_wifi_val} | Cabo: {n_cabo_val}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# 2. CONSULTORIA COMERCIAL & UPGRADE MESH (ISP)
# =========================
elif pagina == "Consultoria Comercial & Upgrade Mesh (ISP)":
    st.subheader("💼 Módulo Comercial ISP: Fechamento de Vendas & Retenção")
    st.write("Ferramenta de vendas para o atendente do provedor demonstrar de forma simples e visual ao assinante a necessidade do Mesh.")

    with st.form("form_comercial_novo"):
        c1, c2 = st.columns(2)
        with c1:
            nome_ass = st.text_input("Nome do Assinante", value="Maria Oliveira")
            plano_atual = st.selectbox("Plano Contratado", [300, 500, 700, 1000], index=1)
            local_fraco = st.selectbox("Onde o sinal está ruim?", ["Quarto", "Banheiro", "Pátio", "2º Andar"])
        with c2:
            distancia_lenta = st.slider("Distância estimada (metros)", 4.0, 18.0, 9.0, 0.5)
            num_paredes = st.slider("Paredes no trajeto", 1, 5, 3)
            preco_mesh = st.number_input("Valor da Mensalidade do Mesh Extra (R$)", value=19.90, step=5.0)

        gerar_proposta = st.form_submit_button("Gerar Proposta Comercial Pronta")

    if gerar_proposta:
        rssi_v = estimar_rssi(distancia_lenta, num_paredes)
        vel_v = estimar_velocidade(rssi_v, plano_atual, "5GHz", 4)
        nos_rec = 1 if num_paredes <= 3 else 2

        st.markdown("---")
        st.markdown(
            f"""
            <div class='commercial-card'>
                <h3 style='color: #0369a1; margin-top: 0;'>📄 Proposta Comercial para {nome_ass}</h3>
                <p><b>Diagnóstico:</b> O plano de <b>{plano_atual} Mbps</b> entrega apenas <span style='color: #dc2626; font-size: 16px; font-weight: bold;'>{vel_v} Mbps</span> no local ({local_fraco}).</p>
                <p><b>Motivo:</b> Bloqueio físico de {num_paredes} paredes de alvenaria.</p>
                <hr style='border-color: #bae6fd;'>
                <h4 style='color: #0284c7; margin-bottom: 5px;'>💡 Solução Sugerida:</h4>
                <p>Instalação de <b>{nos_rec} Roteador(es) Wi-Fi Mesh</b> para cobertura total da residência.</p>
                <h4 style='color: #0369a1; margin-top: 15px; margin-bottom: 5px;'>💰 Vantagem Comercial:</h4>
                <p style='font-size: 18px; color: #1e293b;'><b>
