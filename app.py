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
            """
            <div class='commercial-card'>
                <h3 style='color: #0369a1; margin-top: 0;'>📄 Proposta Comercial</h3>
                <p><b>Diagnóstico:</b> O plano contratado sofre atenuação devido a barreiras físicas.</p>
                <hr style='border-color: #bae6fd;'>
                <h4 style='color: #0284c7; margin-bottom: 5px;'>💡 Solução Sugerida:</h4>
                <p>Instalação de nós Mesh adicionais para cobertura total da residência.</p>
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
            "led_detalhes": "🟢 POWER Fixo: Aparelho ligado | 🟢 WAN Fixo: Internet conectada | 🟢 LAN Fixo: Cabo de rede ativo."
        },
        {
            "fabricante": "Greatek",
            "modelo": "Greatek 1500AX / 1200",
            "perfil": "Excelente custo-benefício para provedores de internet regional (ISPs).",
            "capacidade": "Até 32 dispositivos conectados.",
            "cabo_down": "926.8 Mbps", "cabo_up": "942.6 Mbps",
            "wifi_down": "754.6 Mbps", "wifi_up": "524.1 Mbps",
            "led_detalhes": "🟢 POWER Aceso: Sistema operando | 🟢 Internet/WAN: Ativo com tráfego | 🟢 Portas LAN ativas."
        },
        {
            "fabricante": "Nokia / ZTE",
            "modelo": "ONTs / Roteadores GPON Corporativos",
            "perfil": "Padrão de mercado para entrega de fibra óptica FTTH diretamente nas residências.",
            "capacidade": "Gerenciamento corporativo avançado.",
            "cabo_down": "940.0 Mbps", "cabo_up": "940.0 Mbps",
            "wifi_down": "600.0 Mbps", "wifi_up": "450.0 Mbps",
            "led_detalhes": "🟢 POWER: Energia ok | 🟢 PON Fixo: Fibra sincronizada | 🔴 LOSS Vermelho/Piscando: Sem sinal óptico (Fibra partida) | 🟢 LAN Ativa."
        },
        {
            "fabricante": "TP-Link",
            "modelo": "TP-Link Archer (Linha AX / C)",
            "perfil": "Alta compatibilidade com dispositivos smart home e facilidade de configuração.",
            "capacidade": "Até 48 dispositivos conectados.",
            "cabo_down": "915.0 Mbps", "cabo_up": "920.0 Mbps",
            "wifi_down": "580.0 Mbps", "wifi_up": "410.0 Mbps",
            "led_detalhes": "🟢 POWER: Sistema ligado | 🟢 Internet (Globo) Verde: Acesso à web liberado | 🟢 Wi-Fi ativo."
        },
        {
            "fabricante": "D-Link",
            "modelo": "D-Link DIR-2150 / DIR-842",
            "perfil": "Roteador robusto para residências de médio porte.",
            "capacidade": "Até 30 dispositivos conectados.",
            "cabo_down": "911.9 Mbps", "cabo_up": "928.2 Mbps",
            "wifi_down": "359.9 Mbps", "wifi_up": "405.7 Mbps",
            "led_detalhes": "🟢 POWER Fixo | 🟢 WAN ativa | 🟢 Portas LAN conectadas."
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
                <p style='margin: 4px 0; font-size: 13px;'><b>Diagnóstico de LEDs (Power, PON, Loss, WAN, LAN):</b><br>
                {item['led_detalhes']}</p>
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
            comodo_prob = st.text_input("Ambiente Afetado", value="Quarto")
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
st.markdown("### 📋 Dados de Coleta de Campo (Google Sheets)")
df_campo = carregar_dados_campo()

if df_campo is not None:
    st.dataframe(df_campo, use_container_width=True)
else:
    st.info("A aguardar dados da planilha de campo...")

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import streamlit as st

st.subheader("📡 Módulo de Diagnóstico Automático e Heatmap do Provedor")

df_campo = carregar_dados_campo()

if df_campo is not None and not df_campo.empty:
    # Certifique-se de que a planilha tem colunas 'X', 'Y' (coordenadas no piso) e 'RSSI_dBm'
    if all(col in df_campo.columns for col in ['X', 'Y', 'RSSI_dBm']):
        
        st.success("🤖 Motor de IA e Interpolação Espacial Ativo: Gerando mapa de calor da residência...")
        
        # Extrair coordenadas e valores de RSSI recolhidos em campo
        points = df_campo[['X', 'Y']].values
        values = df_campo['RSSI_dBm'].values
        
        # Criar a malha (grid) da casa para preencher os espaços entre os cômodos
        grid_x, grid_y = np.mgrid[min(points[:,0]):max(points[:,0]):100j, 
                                  min(points[:,1]):max(points[:,1]):100j]
        
        # Interpolação cúbica/linear dos pontos medidos pelo técnico
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        # Plotar o Mapa de Calor (Heatmap) semelhante ao WiFiman
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(grid_z.T, extent=(min(points[:,0]), max(points[:,0]), min(points[:,1]), max(points[:,1])), 
                        origin='lower', cmap='RdYlGn', interpolation='bicubic', vmin=-80, vmax=-40)
        
        # Adicionar os pontos exatos onde o técnico mediu
        ax.scatter(points[:,0], points[:,1], c=values, cmap='RdYlGn', edgecolor='black', s=100, vmin=-80, vmax=-40)
        
        fig.colorbar(cax, label='Intensidade do Sinal (RSSI dBm)')
        ax.set_title("Mapa de Calor Térmico Automatizado - Cobertura Wi-Fi")
        ax.set_xlabel("Eixo X (Metros)")
        ax.set_ylabel("Eixo Y (Metros)")
        
        st.pyplot(fig)
        
        # Diagnóstico automático para o técnico do provedor
        pior_sinal = df_campo.loc[df_campo['RSSI_dBm'].idxmin()]
        st.warning(f"⚠️ **Ponto Crítico Identificado pela IA:** {pior_sinal.get('Ambiente', 'Área externa')} com RSSI de {pior_sinal['RSSI_dBm']} dBm. Recomenda-se a instalação de um repetidor/mesh nesta zona.")
        
    else:
        st.info("ℹ️ Para gerar o mapa de calor automático, certifique-se de que a planilha de campo possui as colunas de coordenadas **X** e **Y** (além de `RSSI_dBm` e `Ambiente`).")
else:
    st.info("A aguardar dados de campo para processamento automático...")


st.divider()
st.markdown("<p style='text-align: center; color: #64748b; font-size: 13px;'>WiFiLab Pro — Criado e Desenvolvido por <b>Dayrlane Sousa</b> | TCC Ciencias e Tecnologia- ifpa</p>", unsafe_allow_html=True)


        
