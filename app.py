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

st.title("📶 WiFiLab - Gestão e Diagnóstico de Wi-Fi")
st.caption("Sistema Inteligente de Cobertura Residencial e Suporte Técnico N1/N2")

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
mse_modelo = mean_squared_error(Y_REF, y_pred_ref)
mae_modelo = mean_absolute_error(Y_REF, y_pred_ref)

def estimar_rssi(distancia, obstaculos):
    entrada = np.array([[
        max(float(distancia), 0.1),
        max(int(obstaculos), 0)
    ]])
    return float(rna.predict(entrada)[0])

def classificar_rssi(rssi):
    if rssi >= -50:
        return "Excelente (Muito bom)"
    elif rssi >= -60:
        return "Bom (Estável para streaming)"
    elif rssi >= -67:
        return "Razoável (Navegação básica)"
    elif rssi >= -75:
        return "Fraco (Possíveis quedas)"
    return "Muito fraco (Sem conexão útil)"

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
        "Planta Baixa & Dispositivos (N1)",
        "Suporte N1/N2",
        "Histórico"
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab | Protótipo TCC")

# =========================
# PAINEL INICIAL
# =========================
if pagina == "Painel inicial":
    st.subheader("Painel inicial")
    historico = carregar_historico()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Atendimentos", len(historico))
    c2.metric("MSE do Modelo", f"{mse_modelo:.2f} dB²")
    c3.metric("MAE do Modelo", f"{mae_modelo:.2f} dB")
    c4.metric("Versão", "2.0 TCC")

    st.markdown("### Diferencial Competitivo frente ao Mercado")
    st.write(
        "Este aplicativo vai além de simples medidores de sinal (como o WiFiman). "
        "Ele cruza **Inteligência Artificial (RNA)** com uma **planta baixa interativa e mapeamento de dispositivos por cômodo**, "
        "permitindo que o suporte N1 identifique exatamente onde há perda excessiva de sinal e qual dispositivo está sofrendo interferência."
    )

    st.info(
        f"A RNA integrada possui erro absoluto médio de calibração de **{mae_modelo:.2f} dB**, "
        "garantindo estimativas consistentes para o dimensionamento residencial."
    )

# =========================
# COBERTURA RESIDENCIAL
# =========================
elif pagina == "Cobertura residencial":
    st.subheader("🏠 Simulador de cobertura residencial")

    col1, col2, col3 = st.columns(3)
    with col1:
        largura = st.number_input("Largura da residência (m)", min_value=2.0, max_value=50.0, value=10.0, step=0.5)
    with col2:
        comprimento = st.number_input("Comprimento da residência (m)", min_value=2.0, max_value=50.0, value=8.0, step=0.5)
    with col3:
        obstaculos = st.number_input("Paredes/Obstáculos médios", min_value=0, max_value=10, value=1, step=1)

    col4, col5 = st.columns(2)
    with col4:
        router_x = st.slider("Posição X do Roteador (m)", 0.0, float(largura), float(largura / 2), 0.5)
    with col5:
        router_y = st.slider("Posição Y do Roteador (m)", 0.0, float(comprimento), float(comprimento / 2), 0.5)

    xs = np.linspace(0, largura, 35)
    ys = np.linspace(0, comprimento, 35)
    XX, YY = np.meshgrid(xs, ys)
    distancias = np.sqrt((XX - router_x)**2 + (YY - router_y)**2)

    Z = np.zeros_like(distancias)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = estimar_rssi(distancias[i, j], obstaculos)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mapa = ax.imshow(
        Z, origin="lower", extent=[0, largura, 0, comprimento],
        aspect="auto", cmap="RdYlGn", vmin=-90, vmax=-40
    )
    ax.scatter([router_x], [router_y], marker="*", s=260, color="blue", edgecolors="white", label="Roteador")
    ax.set_xlabel("Largura (m)")
    ax.set_ylabel("Comprimento (m)")
    ax.set_title("Mapa Térmico de RSSI da Residência")
    ax.legend()
    barra = fig.colorbar(mapa, ax=ax)
    barra.set_label("RSSI estimado (dBm)")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# PLANTA BAIXA & DISPOSITIVOS (N1)
# =========================
elif pagina == "Planta Baixa & Dispositivos (N1)":
    st.subheader("🗺️ Mapeamento de Cômodos e Dispositivos por Atendimento")
    st.write(
        "Esta ferramenta permite ao suporte técnico N1 mapear onde estão os aparelhos do cliente "
        "e calcular a perda de sinal exata com base na distância ao roteador e barreiras."
    )

    with st.form("form_dispositivos"):
        c1, c2 = st.columns(2)
        with c1:
            cliente_id = st.text_input("Nome do Cliente / OS", value="Cliente Exemplo")
            comodo_alvo = st.selectbox("Cômodo do Dispositivo com Problema", ["Sala de Estar", "Quarto Casal", "Quarto Solteiro", "Cozinha", "Escritório / Home Office", "Área Externa"])
            dispositivo = st.selectbox("Tipo de Aparelho", ["Smartphone", "Smart TV", "Notebook / PC", "Console de Videogame", "Câmera Wi-Fi"])
        with c2:
            dist_comodo = st.slider("Distância estimada até o Roteador (metros)", 1.0, 30.0, 6.0, 0.5)
            paredes_comodo = st.slider("Quantidade de Paredes entre o Roteador e o Cômodo", 0, 5, 2, 1)
            rssi_referencia_livre = -40.0  # RSSI base junto ao roteador

        analisar_dispositivo = st.form_submit_button("Analisar Perda e Qualidade do Dispositivo")

    if analisar_dispositivo:
        rssi_estimado_dispositivo = estimar_rssi(dist_comodo, paredes_comodo)
        perda_total_db = abs(rssi_referencia_livre - rssi_estimado_dispositivo)
        
        st.markdown("---")
        st.markdown(f"### 📊 Relatório de Diagnóstico para: **{dispositivo}** ({comodo_alvo})")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sinal Estimado (RSSI)", f"{rssi_estimado_dispositivo:.1f} dBm")
        col_b.metric("Perda de Sinal Calculada", f"{perda_total_db:.1f} dB")
        col_c.metric("Qualidade do Link", classificar_rssi(rssi_estimado_dispositivo))

        st.warning(
            f"**Diagnóstico para o Suporte N1:** O dispositivo **{dispositivo}** localizado no cômodo **{comodo_alvo}** "
            f"encontra-se a **{dist_comodo} metros** do roteador, atravessando **{paredes_comodo} paredes**. "
            f"A atenuação estimada é de **{perda_total_db:.1f} dB**. "
            "Se o cliente relatar lentidão ou quedas, recomenda-se reposicionar o roteador para o centro da casa ou adicionar um repetidor/mesh."
        )

# =========================
# SUPORTE N1/N2
# =========================
elif pagina == "Suporte N1/N2":
    st.subheader("🛠️ Registro Completo de Atendimento Técnico")

    with st.form("form_atendimento"):
        c1, c2 = st.columns(2)
        with c1:
            cliente = st.text_input("Cliente ou identificador")
            protocolo = st.text_input("Protocolo / O.S.")
            tecnico = st.text_input("Técnico responsável")
            ambiente = st.selectbox("Ambiente", ["Sala", "Quarto", "Cozinha", "Escritório", "Corredor", "Outro"])
        with c2:
            queixa = st.selectbox("Queixa principal", ["Sinal fraco", "Lentidão", "Oscilação", "Queda de conexão", "Outro"])
            distancia = st.number_input("Distância do roteador (m)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
            obstaculos = st.number_input("Quantidade de obstáculos", min_value=0, max_value=20, value=1, step=1)
            rssi_medido = st.number_input("RSSI medido (dBm)", min_value=-100.0, max_value=0.0, value=-70.0, step=1.0)

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

            salvar_atendimento(registro)
            st.success("Atendimento registrado no arquivo de histórico com sucesso!")

# =========================
# HISTÓRICO
# =========================
elif pagina == "Histórico":
    st.subheader("📋 Histórico de atendimentos")

    historico = carregar_historico()

    if historico.empty:
        st.info("Nenhum atendimento registrado ainda.")
    else:
        st.markdown("### 🔍 Filtros avançados")
        f_col1, f_col2 = st.columns(2)

        queixas_disponiveis = ["Todas"] + list(historico["Queixa"].unique())
        filtro_queixa = f_col1.selectbox("Filtrar por queixa principal", queixas_disponiveis)

        tecnicos_disponiveis = ["Todos"] + list(historico["Técnico"].astype(str).unique())
        filtro_tecnico = f_col2.selectbox("Filtrar por técnico", tecnicos_disponiveis)

        df_filtrado = historico.copy()
        if filtro_queixa != "Todas":
            df_filtrado = df_filtrado[df_filtrado["Queixa"] == filtro_queixa]
        if filtro_tecnico != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Técnico"] == filtro_tecnico]

        st.divider()
        st.metric("Registros exibidos / Total", f"{len(df_filtrado)} / {len(historico)}")
        st.dataframe(df_filtrado, use_container_width=True)

        csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Baixar histórico filtrado em CSV",
            data=csv,
            file_name="historico_wifilab_filtrado.csv",
            mime="text/csv"
        )

st.divider()
st.caption("WiFiLab — protótipo acadêmico desenvolvido para TCC.")
        
