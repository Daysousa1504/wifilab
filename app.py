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

def estimar_velocidade(rssi, plano):
    if rssi >= -50: return plano
    elif rssi >= -60: return int(plano * 0.8)
    elif rssi >= -67: return int(plano * 0.5)
    elif rssi >= -75: return int(plano * 0.2)
    return int(plano * 0.05)

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
        "Visualize o layout estrutural da casa (com paredes e divisões de cômodos). "
        "Escolha onde o roteador está posicionado para ver em tempo real como o sinal interage com o ambiente."
    )

    c_op1, c_op2 = st.columns(2)
    with c_op1:
        local_roteador = st.selectbox(
            "Posição do Roteador na Casa",
            ["Sala de Estar (Perto da TV)", "Corredor Central", "Cozinha"]
        )
    with c_op2:
        plano_contratado = st.number_input("Plano de Internet Contratado (Mbps)", value=500, step=50)

    if "Sala" in local_roteador:
        rx, ry = 2.0, 6.0
    elif "Corredor" in local_roteador:
        rx, ry = 5.0, 4.0
    else:
        rx, ry = 8.0, 6.0

    fig_casa = go.Figure()

    fig_casa.add_shape(type="rect", x0=0, y0=4, x1=5, y1=8, fillcolor="#e0f2fe", opacity=0.5, line=dict(color="#0284c7", width=3))
    fig_casa.add_annotation(x=2.5, y=6.0, text="<b>Sala de Estar</b><br>(Sofá, TV)", showarrow=False, font=dict(color="#0369a1", size=12))

    fig_casa.add_shape(type="rect", x0=5, y0=4, x1=10, y1=8, fillcolor="#fef3c7", opacity=0.5, line=dict(color="#d97706", width=3))
    fig_casa.add_annotation(x=7.5, y=6.0, text="<b>Cozinha</b><br>(Geladeira, Fogão)", showarrow=False, font=dict(color="#92400e", size=12))

    fig_casa.add_shape(type="rect", x0=0, y0=0, x1=5, y1=4, fillcolor="#f1f5f9", opacity=0.5, line=dict(color="#64748b", width=3))
    fig_casa.add_annotation(x=2.5, y=2.0, text="<b>Quarto 1</b><br>(Cama Casal)", showarrow=False, font=dict(color="#334155", size=12))

    fig_casa.add_shape(type="rect", x0=5, y0=0, x1=10, y1=4, fillcolor="#f1f5f9", opacity=0.5, line=dict(color="#64748b", width=3))
    fig_casa.add_annotation(x=7.5, y=2.0, text="<b>Quarto 2 / Suíte</b><br>(Escritório / Cama)", showarrow=False, font=dict(color="#334155", size=12))

    fig_casa.add_trace(go.Scatter(
        x=[rx], y=[ry],
        mode='markers+text',
        marker=dict(size=22, color='#0284c7', symbol='diamond'),
        text=["<b>Roteador Wi-Fi</b>"],
        textposition="top center"
    ))

    fig_casa.update_layout(
        title="Planta Baixa Arquitetônica & Posicionamento de Hardware",
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 8.5], showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_casa, use_container_width=True)

    st.markdown("### 📊 Diagnóstico de Cobertura por Ambiente")
    
    ambientes = [
        {"nome": "Sala de Estar", "x": 2.5, "y": 6.0, "paredes": 0 if "Sala" in local_roteador else 1},
        {"nome": "Cozinha", "x": 7.5, "y": 6.0, "paredes": 1 if "Cozinha" in local_roteador else 2},
        {"nome": "Quarto 1", "x": 2.5, "y": 2.0, "paredes": 2},
        {"nome": "Quarto 2 (Fundos)", "x": 7.5, "y": 2.0, "paredes": 3},
    ]

    col1, col2 = st.columns(2)
    for i, amb in enumerate(ambientes):
        dist = np.sqrt((amb["x"] - rx)**2 + (amb["y"] - ry)**2)
        rssi = estimar_rssi(dist, amb["paredes"])
        vel_util = estimar_velocidade(rssi, plano_contratado)
        qualidade = classificar_qualidade(rssi)

        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h4 style='margin-top: 0;'>📍 {amb['nome']}</h4>
                    <p style='margin: 4px 0;'><b>Qualidade:</b> {qualidade}</p>
                    <p style='margin: 4px 0;'><b>Velocidade Real:</b> <span style='color: #0284c7; font-size: 18px; font-weight: bold;'>{vel_util} Mbps</span> / {plano_contratado} Mbps</p>
                    <p style='margin: 4px 0; font-size: 12px; color: #64748b;'>Distância do Roteador: {dist:.1f}m | Barreiras: {amb['paredes']} paredes</p>
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
            local_fraco = st.selectbox("Onde o sinal está ruim?", ["Quarto dos Fundos", "Escritório / Home Office", "Área de Lazer"])
        with c2:
            distancia_lenta = st.slider("Distância estimada (metros)", 4.0, 18.0, 9.0, 0.5)
            num_paredes = st.slider("Paredes no trajeto", 1, 5, 3)
            preco_mesh = st.number_input("Valor da Mensalidade do Mesh Extra (R$)", value=19.90, step=5.0)

        gerar_proposta = st.form_submit_button("Gerar Proposta Comercial Pronta")

    if gerar_proposta:
        rssi_v = estimar_rssi(distancia_lenta, num_paredes)
        vel_v = estimar_velocidade(rssi_v, plano_atual)
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
                <p style='font-size: 18px; color: #1e293b;'><b>Acrésimo de R$ {preco_mesh:.2f} / mês</b> na fatura do cliente (Garantia de fidelização e zero cancelamentos).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.success("Proposta gerada com sucesso! Pronta para apresentação ao assinante.")

# =========================
# 3. CATÁLOGO OFICIAL DE FABRICANTES & LEDS
# =========================
elif pagina == "Catálogo Oficial de Fabricantes & LEDs":
    st.subheader("🌐 Catálogo Oficial de Fabricantes & Guia de Indicadores (LEDs)")
    st.write(
        "Fichas técnicas oficiais dos principais fabricantes de telecomunicações do mercado (Huawei, TP-Link, Nokia, ZTE, Greatek e D-Link). "
        "Inclui dados de desempenho em laboratório e o diagnóstico completo dos LEDs de status (Power, PON, Loss, WAN, LAN)."
    )

    catalogo = [
        {
            "fabricante": "Huawei",
            "modelo": "Huawei AX2 / AX3S",
            "perfil": "Ideal para planos de alta velocidade (500Mb a 1Gbps) com alta densidade de clientes.",
            "capacidade": "Até 64 dispositivos conectados simultaneamente.",
            "cabo_down": "923.3 Mbps", "cabo_up": "935.2 Mbps",
            "wifi_down": "681.0 Mbps", "wifi_up": "396.4 Mbps",
            "led_power": "🟢 POWER Fixo: Aparelho ligado e alimentado corretamente.",
            "led_wan": "🟢 WAN Fixo: Porta de internet conectada com IP válido.",
            "led_lan": "🟢 LAN Fixo: Dispositivo conectado por cabo de rede.",
            "led_pon_loss": "N/A (Gerido via ONT externa)"
        },
        {
            "fabricante": "Greatek",
            "modelo": "Greatek 1500AX / 1200",
            "perfil": "Excelente custo-benefício para provedores de internet regional (ISPs).",
            "capacidade": "Até 32 dispositivos conectados.",
            "cabo_down": "926.8 Mbps", "cabo_up": "942.6 Mbps",
            "wifi_down": "754.6 Mbps", "wifi_up": "524.1 Mbps",
            "led_power": "🟢 POWER Aceso: Sistema operando normalmente.",
            "led_wan": "🟢 Internet / WAN: Verde ativo com tráfego de dados.",
            "led_lan": "🟢 Portas LAN 1 a 4 ativas.",
            "led_pon_loss": "N/A"
        },
        {
            "fabricante": "Nokia / ZTE",
            "modelo": "ONTs / Roteadores GPON Corporativos",
            "perfil": "Padrão de mercado para entrega de fibra óptica FTTH diretamente nas residências.",
            "capacidade": "Gerenciamento corporativo avançado.",
            "cabo_down": "940.0 Mbps", "cabo_up": "940.0 Mbps",
            "wifi_down": "600.0 Mbps", "wifi_up": "450.0 Mbps",
            "led_power": "🟢 POWER: Energia estabilizada.",
            "led_pon": "🟢 PON Fixo: Sinal de fibra óptica sincronizado com a OLT da operadora.",
            "led_loss": "🔴 LOSS Vermelho/Piscando: Sem sinal óptico (Cabo de fibra partido ou atenuado).",
            "led_lan": "🟢 LAN Ativa."
        },
        {
            "fabricante": "TP-Link",
            "modelo": "TP-Link Archer (Linha AX / C)",
            "perfil": "Alta compatibilidade com dispositivos smart home e facilidade de configuração.",
            "capacidade": "Até 48 dispositivos conectados.",
            "cabo_down": "915.0 Mbps", "cabo_up": "920.0 Mbps",
            "wifi_down": "580.0 Mbps", "wifi_up": "410.0 Mbps",
            "led_power": "🟢 POWER: Sistema ligado.",
            "led_internet": "🟢 Globo/Internet Azul ou Verde: Acesso à web liberado.",
            "led_wifi": "🟢 Wi-Fi 2.4GHz e 5GHz ativos."
        },
        {
            "fabricante": "D-Link",
            "modelo": "D-Link DIR-2150 / DIR-842",
            "perfil": "Roteador robusto para residências de médio porte.",
            "capacidade": "Até 30 dispositivos conectados.",
            "cabo_down": "911.9 Mbps", "cabo_up": "928.2 Mbps",
            "wifi_down": "359.9 Mbps", "wifi_up": "405.7 Mbps",
            "led_power": "🟢 POWER Fixo.",
            "led_wan": "🟢 Internet ativa.",
            "led_lan": "🟢 Portas LAN conectadas."
        }
    ]

    for item in catalogo:
        st.markdown(
            f"""
            <div class='router-card'>
                <h3 style='color: #0284c7; margin-top: 0;'>🏢 {item['fabricante']} — {item['modelo']}</h3>
                <p style='margin: 2px 0;'><b>Perfil de Uso:</b> {item['perfil']}</p>
                <p style='margin: 2px 0;'><b>Capacidade Recomendada:</b> {item['capacidade']}</p>
                <div style='display: flex; gap: 20px; margin: 10px 0;'>
                    <div><b>💻 Cabo (Testes):</b> Down: {item['cabo_down']} | Up: {item['cabo_up']}</div>
                    <div><b>📶 Wi-Fi (Testes):</b> Down: {item['wifi_down']} | Up: {item['wifi_up']}</div>
                </div>
                <hr style='border-color: #e2e8f0; margin: 8px 0;'>
                <p style='margin: 4px 0; font-size: 13px;'><b>Diagnóstico de LEDs (Power, PON, LOSS, WAN, LAN):</b><br>
                {item['led_power']}<br>
                {item.get('led_pon', '')} {item.get('led_loss', '')}<br>
                {item.get('led_wan', '')} {item.get('led_internet', '')}<br>
                {item['led_lan']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# 4. SUPORTE TÉCNICO & HISTÓRICO DE O.S.
# =========================
elif pagina == "Suporte Técnico & Histórico de O.S.":
    st.subheader("🛠️ Registo de Atendimento Técnico e Ordens de Serviço (O.S.)")
    st.write("Guarde o histórico de atendimentos realizados para consulta rápida e auditoria técnica do provedor.")

    with st.form("form_os_novo"):
        c1, c2 = st.columns(2)
        with c1:
            os_num = st.text_input("Número da O.S.", value="OS-4490")
            tecnico = st.text_input("Técnico Responsável", value="Equipe N1")
            comodo_prob = st.text_input("Ambiente Afetado", value="Quarto dos Fundos")
        with c2:
            plano_os = st.number_input("Plano (Mbps)", value=500)
            vel_aferida = st.number_input("Velocidade Medida (Mbps)", value=110.0)

        parecer = st.text_area("Parecer Técnico", value="Orientado o cliente sobre barreiras físicas e sugerido upgrade para Mesh.")
        salvar = st.form_submit_button("Guardar Atendimento")

    if salvar:
        registro = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "O.S.": os_num,
            "Técnico": tecnico,
            "Ambiente": comodo_prob,
            "Plano": plano_os,
            "Velocidade Real": vel_aferida,
            "Parecer": parecer
        }
        salvar_atendimento(registro)
        st.success("Atendimento guardado com sucesso!")

    st.divider()
    st.markdown("### 📋 Histórico Consolidado de O.S.")
    hist_df = carregar_historico()
    if hist_df.empty:
        st.info("Nenhum atendimento registado ainda.")
    else:
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Descarregar Relatório CSV", data=csv, file_name="relatorio_wifilab.csv", mime="text/csv")

st.divider()
st.markdown("<p style='text-align: center; color: #64748b; font-size: 13px;'>WiFiLab Pro — Criado e Desenvolvido por <b>Dayrlane Sousa</b> | TCC Engenharia de Redes</p>", unsafe_allow_html=True)
