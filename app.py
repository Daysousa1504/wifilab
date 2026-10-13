import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# CONFIGURAÇÃO DO APLICATIVO
# =========================
st.set_page_config(
    page_title="WiFiLab Pro - Planta Inteligente",
    page_icon="📶",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    [data-testid="stSidebar"] * { color: #ffffff; }
    .room-card {
        background-color: #1b212c;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📶 WiFiLab — Gerador de Planta Baixa & Diagnóstico de 500Mb")
st.caption("Simulação de Cobertura Residencial Automatizada por Inteligência Artificial (RNA)")

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
y_pred_ref = rna.predict(X_REF)
mae_modelo = mean_absolute_error(Y_REF, y_pred_ref)

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
    """Estima a taxa de entrega útil de um plano de 500Mb baseando-se no RSSI."""
    if rssi >= -50:
        return 480  # Próximo ao limite da porta/Wi-Fi
    elif rssi >= -60:
        return 380
    elif rssi >= -67:
        return 220
    elif rssi >= -75:
        return 90
    else:
        return 20   # Queda severa de pacote

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
        "Gerador de Planta Baixa (500Mb)",
        "Simulador Térmico Geral",
        "Suporte N1/N2 & Histórico"
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab | Módulo Residencial Avançado")

# =========================
# 1. GERADOR DE PLANTA BAIXA AUTOMATIZADO
# =========================
if pagina == "Gerador de Planta Baixa (500Mb)":
    st.subheader("🏠 Planta Baixa Automatizada por Tipologia & Plano de 500 Mbps")
    st.write(
        "Selecione a configuração da residência e onde o roteador está instalado. "
        "O sistema mapeará automaticamente os cômodos, calculando a atenuação do sinal e a velocidade real estimada."
    )

    with st.form("form_planta_automatica"):
        c1, c2 = st.columns(2)
        with c1:
            tipo_casa = st.selectbox(
                "Tipologia da Residência",
                [
                    "2 Quartos, 1 Banheiro, Sala Conjugada",
                    "3 Quartos (1 Suíte), 2 Banheiros, Sala e Cozinha",
                    "Kitnet / Studio (Cômodo Único)"
                ]
            )
            local_roteador = st.selectbox(
                "Localização do Roteador Wi-Fi",
                ["Sala de Estar (perto da TV)", "Corredor Central", "Cozinha", "Quarto Principal"]
            )
        with c2:
            banda_contratada = st.number_input("Plano de Internet Contratado (Mbps)", value=500, step=50)
            frequencia_wi_fi = st.selectbox("Banda de Operação Analisada", ["5 GHz (Alta Velocidade / Menor Alcance)", "2.4 GHz (Maior Alcance / Mais Interferência)"])

        gerar_planta = st.form_submit_button("Gerar Mapeamento da Planta e Análise de Desempenho")

    if gerar_planta:
        st.markdown("---")
        st.markdown(f"### 📐 Relatório Estrutural para: *{tipo_casa}*")
        st.info(f"Roteador posicionado na **{local_roteador}** | Plano de referência: **{banda_contratada} Mbps** ({frequencia_wi_fi})")

        # Definindo matriz simulada de cômodos com base na tipologia escolhida
        if "2 Quartos" in tipo_casa:
            comodos = [
                {"nome": "Sala de Estar", "distancia": 1.5 if "Sala" in local_roteador else 4.0, "paredes": 0 if "Sala" in local_roteador else 1},
                {"nome": "Cozinha", "distancia": 3.0 if "Cozinha" in local_roteador else 5.0, "paredes": 0 if "Cozinha" in local_roteador else 1},
                {"nome": "Banheiro", "distancia": 4.5, "paredes": 2},
                {"nome": "Quarto 1 (Frente)", "distancia": 5.5, "paredes": 2},
                {"nome": "Quarto 2 (Fundos)", "distancia": 7.5, "paredes": 3},
            ]
        elif "3 Quartos" in tipo_casa:
            comodos = [
                {"nome": "Sala de Estar", "distancia": 1.5 if "Sala" in local_roteador else 6.0, "paredes": 0 if "Sala" in local_roteador else 2},
                {"nome": "Cozinha", "distancia": 4.0 if "Cozinha" in local_roteador else 5.0, "paredes": 1},
                {"nome": "Banheiro Social", "distancia": 5.0, "paredes": 2},
                {"nome": "Suíte + Banheiro", "distancia": 8.0, "paredes": 3},
                {"nome": "Quarto Solteiro 1", "distancia": 6.0, "paredes": 2},
                {"nome": "Quarto Solteiro 2", "distancia": 7.0, "paredes": 3},
            ]
        else:  # Kitnet
            comodos = [
                {"nome": "Ambiente Único / Estúdio", "distancia": 2.0, "paredes": 0}
            ]

        # Exibição visual em colunas/cards para cada cômodo
        cols = st.columns(2)
        for idx, comodo in enumerate(comodos):
            # Se for 2.4GHz, adiciona leve penalidade de atenuação extra comparado a 5GHz
            fator_freq = 0.0 if "5 GHz" in frequencia_wi_fi else -4.0
            rssi_calculado = estimar_rssi(comodo["distancia"], comodo["paredes"]) + fator_freq
            velocidade_estimada = estimar_velocidade_500mb(rssi_calculado)
            
            # Limita ao máximo contratado
            velocidade_estimada = min(velocidade_estimada, banda_contratada)

            with cols[idx % 2]:
                st.markdown(f"""
                <div class="room-card">
                    <h4>📍 {comodo['nome']}</h4>
                    <b>Distância ao Roteador:</b> {comodo['distancia']} m<br>
                    <b>Paredes no Caminho:</b> {comodo['paredes']}<br>
                    <b>Sinal Estimado (RSSI):</b> {rssi_calculado:.1f} dBm<br>
                    <b>Qualidade:</b> {classificar_rssi(rssi_calculado)}<br>
                    <b>Velocidade Útil Estimada:</b> <span style="color: #3fb950; font-size: 18px;"><b>{velocidade_estimada} Mbps</b></span> / {banda_contratada} Mbps
                </div>
                """, unsafe_allow_html=True)

        st.success(
            "**Diagnóstico para o Suporte Técnico N1:** Os cômodos com mais de 2 paredes de obstáculo "
            "apresentam queda acentuada na taxa de transferência do plano de 500Mb. "
            "Caso o cliente relate lentidão nos fundos, a recomendação automática do sistema é o remanejamento do ponto central "
            "ou a instalação de um nó Mesh."
        )

# =========================
# 2. SIMULADOR TÉRMICO GERAL
# =========================
elif pagina == "Simulador Térmico Geral":
    st.subheader("🗺️ Mapa Térmico Espacial de Cobertura")

    c1, c2, c3 = st.columns(3)
    with c1:
        largura = st.number_input("Largura (m)", 2.0, 30.0, 10.0, 0.5)
    with c2:
        comprimento = st.number_input("Comprimento (m)", 2.0, 30.0, 8.0, 0.5)
    with c3:
        obstaculos = st.number_input("Obstáculos Médios", 0, 10, 2, 1)

    xs = np.linspace(0, largura, 30)
    ys = np.linspace(0, comprimento, 30)
    XX, YY = np.meshgrid(xs, ys)
    distancias = np.sqrt((XX - largura/2)**2 + (YY - comprimento/2)**2)

    Z = np.zeros_like(distancias)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = estimar_rssi(distancias[i, j], obstaculos)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1b212c')
    
    mapa = ax.imshow(Z, origin="lower", extent=[0, largura, 0, comprimento], aspect="auto", cmap="RdYlGn", vmin=-90, vmax=-40)
    ax.scatter([largura/2], [comprimento/2], marker="*", s=250, color="#58a6ff", label="Roteador Central")
    ax.set_xlabel("Largura (m)", color="white")
    ax.set_ylabel("Comprimento (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Propagação de Sinal na Residência", color="white")
    
    cbar = fig.colorbar(mapa, ax=ax)
    cbar.set_label("RSSI (dBm)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    st.pyplot(fig)
    plt.close(fig)

# =========================
# 3. SUPORTE N1/N2 & HISTÓRICO
# =========================
elif pagina == "Suporte N1/N2 & Histórico":
    st.subheader("🛠️ Registo de Ordens de Serviço & Histórico Técnico")

    with st.form("form_os"):
        c1, c2 = st.columns(2)
        with c1:
            cliente = st.text_input("Nome do Cliente / OS", value="OS-8821")
            tecnico = st.text_input("Técnico Responsável", value="Suporte N1")
            comodo_problema = st.selectbox("Cômodo com Relato de Lentidão", ["Quarto 1", "Quarto 2", "Cozinha", "Área Externa"])
        with c2:
            plano_assinado = st.number_input("Plano (Mbps)", value=500)
            velocidade_aferida = st.number_input("Velocidade Medida no Local (Mbps)", value=120.0)
            rssi_medido = st.number_input("RSSI Real Medido (dBm)", value=-72.0)

        obs = st.text_area("Parecer Técnico / Solução Aplicada")
        salvar = st.form_submit_button("Gravar Atendimento")

    if salvar:
        registro = {
            "Data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Cliente": cliente,
            "Técnico": tecnico,
            "Cômodo Afetado": comodo_problema,
            "Plano (Mbps)": plano_assinado,
            "Velocidade Medida (Mbps)": velocidade_aferida,
            "RSSI Medido": rssi_medido,
            "Status": "Analisado via RNA",
            "Parecer": obs
        }
        salvar_atendimento(registro)
        st.success("Atendimento gravado com sucesso no banco de dados do sistema!")

    st.divider()
    st.markdown("### 📋 Histórico Consolidado de Atendimentos")
    historico = carregar_historico()
    if historico.empty:
        st.info("Nenhum atendimento registado até o momento.")
    else:
        st.dataframe(historico, use_container_width=True)
        csv = historico.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar Relatório CSV", data=csv, file_name="wifilab_historico_atendimentos.csv", mime="text/csv")

st.divider()
st.caption("WiFiLab Pro — Desenvolvido para TCC de Engenharia / Redes.")
