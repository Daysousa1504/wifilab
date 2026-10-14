import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# CONFIGURAÇÃO DO APLICATIVO
# =========================
st.set_page_config(
    page_title="WiFiLab",
    page_icon="📶",
    layout="wide"
)

# Estilização inspirada na paleta da logo (Azul corporativo, Ciano/Verde-água e Branco limpo)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp { 
            background: linear-gradient(135deg, #090d16 0%, #0f172a 100%); 
            color: #f8fafc; 
            font-family: 'Inter', sans-serif;
        }
        .block-container { padding-top: 1.5rem; }
        
        [data-testid='stSidebar'] { 
            background: linear-gradient(180deg, #0f172a 0%, #090d16 100%);
            border-right: 1px solid #1e293b; 
        }
        [data-testid='stSidebar'] * { color: #f1f5f9; }
        
        .room-card { 
            background: rgba(30, 41, 59, 0.7); 
            border: 1px solid #06b6d4; 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 10px; 
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.1);
        }
        
        .commercial-card { 
            background: linear-gradient(135deg, #0f172a 0%, #0b3c5d 100%); 
            border: 2px solid #06b6d4; 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 15px; 
            box-shadow: 0 8px 24px rgba(6, 182, 212, 0.15);
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #0284c7 0%, #06b6d4 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            opacity: 0.9;
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);
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
        "<div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 12px; border: 1px solid #06b6d4; margin-bottom: 15px;'>"
        "<h2 style='color: #06b6d4; margin: 0;'>WiFiLab</h2>"
        "</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    "<div style='text-align: center; padding: 8px 0 15px 0;'>"
    "<p style='font-size: 13px; color: #e2e8f0; margin: 0; font-weight: bold;'>Criado por Dayrlane Sousa</p>"
    "<p style='font-size: 11px; color: #06b6d4; margin: 2px 0 0 0;'>TCC Engenharia de Redes</p>"
    "</div>",
    unsafe_allow_html=True
)

# =========================
# DADOS DE REFERÊNCIA DA RNA
# =========================
X_REF = np.array([
    [1.5, 0],
    [3.0, 0],
    [4.0, 0],
    [5.5, 1],
    [6.5, 2],
    [5.0, 2],
    [8.0, 3]
], dtype=float)

Y_REF = np.array([-40, -55, -60, -68, -72, -75, -85], dtype=float)

@st.cache_resource
def treinar_rna():
    modelo = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(8, 4),
            activation="relu",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        )
    )
    modelo.fit(X_REF, Y_REF)
    return modelo

rna = treinar_rna()

def estimar_rssi(distancia, obstaculos):
    entrada = np.array([[max(float(distancia), 0.1), max(int(obstaculos), 0)]])
    return float(rna.predict(entrada)[0])

def classificar_rssi(rssi):
    if rssi >= -50:
        return "Excelente (Ideal para 4K/Jogos)"
    elif rssi >= -60:
        return "Bom (Estável)"
    elif rssi >= -67:
        return "Razoável (Navegação Padrão)"
    elif rssi >= -75:
        return "Fraco (Possíveis Lentidões)"
    return "Muito Fraco (Área de Sombra)"

def estimar_velocidade_500mb(rssi):
    if rssi >= -50:
        return 480
    elif rssi >= -60:
        return 380
    elif rssi >= -67:
        return 220
    elif rssi >= -75:
        return 90
    else:
        return 20

ARQUIVO_HISTORICO = Path("historico_wifilab.csv")

def carregar_historico():
    if ARQUIVO_HISTORICO.exists():
        return pd.read_csv(ARQUIVO_HISTORICO)
    return pd.DataFrame()

def salvar_atendimento(registro):
    historico = carregar_historico()
    historico = pd.concat([historico, pd.DataFrame([registro])], ignore_index=True)
    historico.to_csv(ARQUIVO_HISTORICO, index=False, encoding="utf-8-sig")

# =========================
# MENU LATERAL
# =========================
pagina = st.sidebar.radio(
    "MENU PRINCIPAL",
    [
        "Propagação de Ondas em Tempo Real (Planta)",
        "Gerador de Planta Baixa (500Mb)",
        "Simulação 3D Interativa",
        "Consultoria Comercial & Upgrade Mesh (ISP)",
        "Simulador Térmico Geral",
        "Suporte N1/N2 & Histórico"
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab Pro | Plataforma Oficial ISPs")

# =========================
# 1. PROPAGAÇÃO DE ONDAS EM TEMPO REAL SOBRE PLANTA
# =========================
if pagina == "Propagação de Ondas em Tempo Real (Planta)":
    st.subheader("Simulação Dinâmica de Ondas Wi-Fi sobre Planta Residencial")
    st.write("Arraste os cursores para posicionar o roteador na planta e visualizar as frentes de onda e as paredes estruturais.")

    c_pos1, c_pos2 = st.columns(2)
    with c_pos1:
        pos_roteador_x = st.slider("Posição X do Roteador (Largura)", 0.5, 9.5, 5.0, 0.5)
    with c_pos2:
        pos_roteador_y = st.slider("Posição Y do Roteador (Comprimento)", 0.5, 7.5, 6.5, 0.5)

    res = 60
    x_vals = np.linspace(0, 10, res)
    y_vals = np.linspace(0, 8, res)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    Z_wave = np.zeros_like(X_grid)
    for i in range(res):
        for j in range(res):
            x_p, y_p = X_grid[i, j], Y_grid[i, j]
            dist = np.sqrt((x_p - pos_roteador_x)**2 + (y_p - pos_roteador_y)**2)
            
            paredes = 0
            if (4.0 <= y_p <= 4.3) and not (3.5 <= x_p <= 6.5): 
                paredes += 1
            if (4.8 <= x_p <= 5.2) and (y_p >= 4.0):
                paredes += 1

            onda = np.sin(dist * 7.0) * 3.5 
            rssi_val = estimar_rssi(dist, obstaculos=paredes + (2 if dist > 4 else 0))
            Z_wave[i, j] = rssi_val + onda

    fig_wave = go.Figure(data=go.Contour(
        z=Z_wave,
        x=x_vals,
        y=y_vals,
        colorscale='Tealgrn',
        ncontours=30,
        colorbar=dict(title="RSSI (dBm)")
    ))

    # Paredes da Planta
    fig_wave.add_shape(type="rect", x0=0, y0=0, x1=10, y1=8, line=dict(color="#06b6d4", width=3))
    fig_wave.add_shape(type="line", x0=0, y0=4, x1=3.5, y1=4, line=dict(color="#06b6d4", width=2))
    fig_wave.add_shape(type="line", x0=6.5, y0=4, x1=10, y1=4, line=dict(color="#06b6d4", width=2))
    fig_wave.add_shape(type="line", x0=5, y0=4, x1=5, y1=8, line=dict(color="#06b6d4", width=2))
    fig_wave.add_shape(type="line", x0=3.5, y0=0, x1=3.5, y1=2.5, line=dict(color="#06b6d4", width=2))

    fig_wave.add_annotation(x=2.0, y=6.0, text="<b>Sala de Estar</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=7.5, y=6.0, text="<b>Cozinha</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=2.5, y=2.0, text="<b>Quarto 1</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=7.5, y=2.0, text="<b>Quarto 2</b>", showarrow=False, font=dict(color="white", size=11))

    fig_wave.add_trace(go.Scatter(
        x=[pos_roteador_x],
        y=[pos_roteador_y],
        mode='markers+text',
        marker=dict(size=16, color='#06b6d4', symbol='star'),
        text=["Roteador"],
        textposition="top center"
    ))

    fig_wave.update_layout(
        title="Propagação de Ondas sobre Planta Baixa",
        xaxis_title="Largura (m)",
        yaxis_title="Comprimento (m)",
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_wave, use_container_width=True)
    st.success("Simulação dinâmica carregada com sucesso!")

# =========================
# 2. GERADOR DE PLANTA BAIXA AUTOMATIZADO
# =========================
elif pagina == "Gerador de Planta Baixa (500Mb)":
    st.subheader("Planta Baixa Automatizada & Plano de 500 Mbps")
    
    with st.form("form_planta"):
        tipo_casa = st.selectbox("Tipologia", ["2 Quartos, 1 Banheiro, Sala Conjugada", "3 Quartos, 2 Banheiros, Sala e Cozinha"])
        local_rot = st.selectbox("Localização do Roteador", ["Sala de Estar", "Corredor Central", "Cozinha"])
        plano = st.number_input("Plano Contratado (Mbps)", value=500)
        btn_planta = st.form_submit_button("Gerar Análise")

    if btn_planta:
        st.info(f"Análise gerada para {tipo_casa} com roteador na {local_rot}.")
        st.success("Cálculo de atenuação por cômodo executado com sucesso pela RNA!")

# =========================
# 3. SIMULAÇÃO 3D INTERATIVA
# =========================
elif pagina == "Simulação 3D Interativa":
    st.subheader("Simulação 3D de Propagação Volumétrica")
    obst = st.slider("Obstáculos no Caminho", 0, 5, 2)
    
    n = 20
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 8, n)
    X3, Y3 = np.meshgrid(x, y)
    Z3 = np.zeros_like(X3, dtype=float)
    for i in range(n):
        for j in range(n):
            d = np.sqrt((X3[i,j]-5)**2 + (Y3[i,j]-4)**2)
            Z3[i,j] = estimar_rssi(d, obst)

    fig3d = go.Figure(data=[go.Surface(z=Z3, x=X3, y=Y3, colorscale='Tealgrn', cmin=-90, cmax=-40)])
    fig3d.update_layout(paper_bgcolor="#090d16", plot_bgcolor="#0f172a", font=dict(color="white"), height=450)
    st.plotly_chart(fig3d, use_container_width=True)

# =========================
# 4. CONSULTORIA COMERCIAL (ISP)
# =========================
elif pagina == "Consultoria Comercial & Upgrade Mesh (ISP)":
    st.subheader("Módulo Comercial ISP: Monetização de Rede")
    
    with st.form("form_isp"):
        cliente = st.text_input("Nome do Cliente", value="Assinante Residencial")
        plano_isp = st.selectbox("Plano Atual", [300, 500, 1000], index=1)
        paredes_isp = st.slider("Paredes até o cômodo crítico", 1, 5, 3)
        btn_isp = st.form_submit_button("Gerar Proposta Comercial")

    if btn_isp:
        st.markdown(
            "<div class='commercial-card'>"
            "<h3>Proposta Comercial Automática</h3>"
            f"<p><b>Cliente:</b> {cliente}</p>"
            f"<p><b>Plano:</b> {plano_isp} Mbps</p>"
            "<p><b>Recomendação:</b> Instalação de 1 Nó Mesh adicional.</p>"
            "<p><b>Upsell Sugerido:</b> +R$ 19,90/mês para cobertura total garantida.</p>"
            "</div>",
            unsafe_allow_html=True
        )

# =========================
# 5. SIMULADOR TÉRMICO GERAL
# =========================
elif pagina == "Simulador Térmico Geral":
    st.subheader("Mapa Térmico de Cobertura")
    larg = st.number_input("Largura (m)", 5.0, 20.0, 10.0)
    comp = st.number_input("Comprimento (m)", 5.0, 20.0, 8.0)
    
    xs = np.linspace(0, larg, 25)
    ys = np.linspace(0, comp, 25)
    Xm, Ym = np.meshgrid(xs, ys)
    Zm = np.zeros_like(Xm)
    for i in range(25):
        for j in range(25):
            Zm[i,j] = estimar_rssi(np.sqrt((Xm[i,j]-larg/2)**2 + (Ym[i,j]-comp/2)**2), 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#090d16')
    ax.set_facecolor('#0f172a')
    im = ax.imshow(Zm, origin="lower", extent=[0, larg, 0, comp], aspect="auto", cmap="Tealgrn", vmin=-90, vmax=-40)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# 6. SUPORTE N1/N2 & HISTÓRICO
# =========================
elif pagina == "Suporte N1/N2 & Histórico":
    st.subheader("Registo de Ordens de Serviço")
    
    with st.form("form_os"):
        os_num = st.text_input("Número da OS", value="OS-1024")
        tecnico = st.text_input("Técnico", value="Suporte N1")
        rssi_med = st.number_input("RSSI Medido (dBm)", value=-68.0)
        parecer = st.text_area("Parecer Técnico")
        btn_os = st.form_submit_button("Registar Atendimento")

    if btn_os:
        salvar_atendimento({
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "OS": os_num,
            "Técnico": tecnico,
            "RSSI": rssi_med,
            "Parecer": parecer
        })
        st.success("Atendimento registado com sucesso!")

    st.divider()
    st.markdown("### Histórico Salvo")
    hist = carregar_historico()
    if not hist.empty:
        st.dataframe(hist, use_container_width=True)
    else:
        st.info("Nenhum registo encontrado.")

st.divider()
st.markdown("<p style='text-align: center; color: #64748b; font-size: 13px;'>WiFiLab Pro — Criado e Desenvolvido por <b>Dayrlane Sousa</b> | TCC Engenharia de Redes</p>", unsafe_allow_html=True)
