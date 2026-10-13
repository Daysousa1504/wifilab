import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# =========================
# CONFIGURAÇÃO DO APLICATIVO
# =========================
st.set_page_config(
    page_title="WiFiLab",
    page_icon="📶",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    [data-testid="stSidebar"] {background-color: #10243a;}
    [data-testid="stSidebar"] * {color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📶 WiFiLab")
st.caption("Análise de cobertura Wi-Fi residencial e suporte técnico N1/N2")

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
    entrada = np.array([[
        max(float(distancia), 0.1),
        max(int(obstaculos), 0)
    ]])
    return float(rna.predict(entrada)[0])

def classificar_rssi(rssi):
    if rssi >= -50:
        return "Muito bom"
    elif rssi >= -60:
        return "Bom"
    elif rssi >= -67:
        return "Razoável"
    elif rssi >= -75:
        return "Fraco"
    return "Muito fraco"

# =========================
# HISTÓRICO PERSISTENTE LOCAL
# =========================
ARQUIVO_HISTORICO = Path("historico_wifilab.csv")

def carregar_historico():
    if ARQUIVO_HISTORICO.exists():
        return pd.read_csv(ARQUIVO_HISTORICO)
    return pd.DataFrame()

def salvar_atendimento(registro):
    historico = carregar_historico()
    historico = pd.concat(
        [historico, pd.DataFrame([registro])],
        ignore_index=True
    )
    historico.to_csv(
        ARQUIVO_HISTORICO,
        index=False,
        encoding="utf-8-sig"
    )

# =========================
# MENU
# =========================
pagina = st.sidebar.radio(
    "MENU PRINCIPAL",
    [
        "Painel inicial",
        "Cobertura residencial",
        "Suporte N1/N2",
        "Histórico"
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab | Protótipo acadêmico")

# =========================
# PAINEL INICIAL
# =========================
if pagina == "Painel inicial":
    st.subheader("Painel inicial")
    historico = carregar_historico()

    c1, c2, c3 = st.columns(3)
    c1.metric("Atendimentos registrados", len(historico))
    c2.metric("Modelo", "RNA + Python")
    c3.metric("Módulos", "4")

    st.markdown("### Bem-vinda ao WiFiLab")
    st.write(
        "Este sistema reúne uma simulação inicial de cobertura Wi-Fi "
        "e ferramentas de apoio ao registro e análise de atendimentos técnicos."
    )

    st.info(
        "A RNA usa apenas sete pontos de referência. "
        "As estimativas são experimentais e precisam ser validadas "
        "com medições reais antes de uso profissional."
    )

    st.markdown("""
    **Use o menu lateral para acessar:**
    - 🏠 Cobertura residencial
    - 🛠️ Suporte N1/N2
    - 📋 Histórico de atendimentos
    """)

# =========================
# COBERTURA RESIDENCIAL
# =========================
elif pagina == "Cobertura residencial":
    st.subheader("🏠 Simulador de cobertura residencial")

    col1, col2, col3 = st.columns(3)
    with col1:
        largura = st.number_input(
            "Largura da residência (m)",
            min_value=2.0, max_value=50.0, value=10.0, step=0.5
        )
    with col2:
        comprimento = st.number_input(
            "Comprimento da residência (m)",
            min_value=2.0, max_value=50.0, value=8.0, step=0.5
        )
    with col3:
        obstaculos = st.number_input(
            "Obstáculos considerados",
            min_value=0, max_value=10, value=1, step=1
        )

    col4, col5 = st.columns(2)
    with col4:
        router_x = st.slider(
            "Posição X do roteador (m)",
            0.0, float(largura), float(largura / 2), 0.5
        )
    with col5:
        router_y = st.slider(
            "Posição Y do roteador (m)",
            0.0, float(comprimento), float(comprimento / 2), 0.5
        )

    xs = np.linspace(0, largura, 35)
    ys = np.linspace(0, comprimento, 35)
    XX, YY = np.meshgrid(xs, ys)
    distancias = np.sqrt(
        (XX - router_x)**2 + (YY - router_y)**2
    )

    Z = np.zeros_like(distancias)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = estimar_rssi(distancias[i, j], obstaculos)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mapa = ax.imshow(
        Z,
        origin="lower",
        extent=[0, largura, 0, comprimento],
        aspect="auto",
        cmap="RdYlGn",
        vmin=-90,
        vmax=-40
    )
    ax.scatter(
        [router_x], [router_y],
        marker="*", s=260, color="blue",
        edgecolors="white", label="Roteador"
    )
    ax.set_xlabel("Largura (m)")
    ax.set_ylabel("Comprimento (m)")
    ax.set_title("Mapa estimado de RSSI")
    ax.legend()
    barra = fig.colorbar(mapa, ax=ax)
    barra.set_label("RSSI estimado (dBm)")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Modelo preliminar: os obstáculos são aplicados uniformemente. "
        "Paredes individuais, materiais e interferência ainda não são modelados."
    )

# =========================
# SUPORTE N1/N2
# =========================
elif pagina == "Suporte N1/N2":
    st.subheader("🛠️ Registro e análise de atendimento")

    with st.form("form_atendimento"):
        c1, c2 = st.columns(2)

        with c1:
            cliente = st.text_input("Cliente ou identificador")
            protocolo = st.text_input("Protocolo / O.S.")
            tecnico = st.text_input("Técnico responsável")
            ambiente = st.selectbox(
                "Ambiente",
                ["Sala", "Quarto", "Cozinha", "Escritório",
                 "Corredor", "Outro"]
            )

        with c2:
            queixa = st.selectbox(
                "Queixa principal",
                ["Sinal fraco", "Lentidão", "Oscilação",
                 "Queda de conexão", "Outro"]
            )
            distancia = st.number_input(
                "Distância do roteador (m)",
                min_value=0.1, max_value=100.0,
                value=5.0, step=0.5
            )
            obstaculos = st.number_input(
                "Quantidade de obstáculos",
                min_value=0, max_value=20, value=1, step=1
            )
            rssi_medido = st.number_input(
                "RSSI medido (dBm)",
                min_value=-100.0, max_value=0.0,
                value=-70.0, step=1.0
            )

        observacoes = st.text_area("Observações técnicas")
        enviar = st.form_submit_button("Analisar e registrar atendimento")

    if enviar:
        if not cliente.strip():
            st.error("Informe o cliente ou um identificador.")
        else:
            estimado = estimar_rssi(distancia, obstaculos)
            diferenca = rssi_medido - estimado

            registro = {
                "Data/hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Cliente/ID": cliente.strip(),
                "Protocolo/O.S.": protocolo.strip(),
                "Técnico": tecnico.strip(),
                "Ambiente": ambiente,
                "Queixa": queixa,
                "Distância (m)": distancia,
                "Obstáculos": obstaculos,
                "RSSI medido (dBm)": rssi_medido,
                "RSSI estimado (dBm)": round(estimado, 2),
                "Diferença (dB)": round(diferenca, 2),
                "Classificação do RSSI medido": classificar_rssi(rssi_medido),
                "Observações": observacoes
            }

            st.markdown("### Resultado da análise")
            a, b, c = st.columns(3)
            a.metric("RSSI medido", f"{rssi_medido:.1f} dBm")
            b.metric("RSSI estimado", f"{estimado:.1f} dBm")
            c.metric("Diferença", f"{diferenca:+.1f} dB")

            st.write(
                f"Classificação indicativa do sinal medido: "
                f"**{classificar_rssi(rssi_medido)}**."
            )
            st.warning(
                "A diferença não identifica sozinha a causa da falha. "
                "Correlacione com testes de velocidade, latência, perda de "
                "pacotes, frequência, canal e condições do ambiente."
            )

            salvar_atendimento(registro)
            st.success("Atendimento registrado no arquivo de histórico.")

# =========================
# HISTÓRICO
# =========================
elif pagina == "Histórico":
    st.subheader("📋 Histórico de atendimentos")

    historico = carregar_historico()

    if historico.empty:
        st.info("Nenhum atendimento registrado ainda.")
    else:
        st.metric("Total de registros", len(historico))
        st.dataframe(historico, use_container_width=True)

        csv = historico.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Baixar histórico em CSV",
            data=csv,
            file_name="historico_wifilab.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "WiFiLab — protótipo acadêmico. "
    "Não substitui medições técnicas nem planejamento profissional de RF."
)
