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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# CONFIGURAÇÃO DO APLICATIVO
# =========================
st.set_page_config(
    page_title="WiFiLab Pro - Wave Propagation Edition",
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
    .commercial-card {
        background: linear-gradient(135deg, #1b212c 0%, #10243a 100%);
        border: 2px solid #58a6ff;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# IDENTIDADE & CRÉDITOS NA SIDEBAR
# =========================
st.sidebar.markdown("""
    <div style="text-align: center; padding: 12px; background-color: #1b212c; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 10px;">
        <h2 style="color: #58a6ff; margin: 0;">📶 WiFiLab</h2>
        <p style="font-size: 13px; color: #e6edf3; margin: 4px 0 2px 0; font-weight: bold;">Desenvolvido por Dayrlane Sousa</p>
        <p style="font-size: 11px; color: #8b949e; margin: 0;">Simulador de Ondas & ISP Pro</p>
    </div>
""", unsafe_allow_html=True)

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
        "Simulação 3D Interativa (Casa Tradicional)",
        "Consultoria Comercial & Upgrade Mesh (ISP)",
        "Simulador Térmico Geral",
        "Suporte N1/N2 & Histórico"
    ]
)

st.sidebar.divider()
st.sidebar.caption("WiFiLab | Protótipo Académico TCC")

# =========================
# 1. PROPAGAÇÃO DE ONDAS EM TEMPO REAL SOBRE PLANTA
# =========================
if pagina == "Propagação de Ondas em Tempo Real (Planta)":
    st.subheader("🌊 Simulação Dinâmica de Ondas Wi-Fi sobre Planta Residencial Real")
    st.write(
        "Arraste os cursores para posicionar o roteador na planta da casa. "
        "As frentes de onda eletromagnética propagam-se através das paredes divisórias (Sala, Quartos, Cozinha e Banheiro)."
    )

    c_pos1, c_pos2 = st.columns(2)
    with c_pos1:
        pos_roteador_x = st.slider("Posição X do Roteador (Largura)", 0.5, 9.5, 5.0, 0.5)
    with c_pos2:
        pos_roteador_y = st.slider("Posição Y do Roteador (Comprimento)", 0.5, 7.5, 6.5, 0.5)

    res = 70
    x_vals = np.linspace(0, 10, res)
    y_vals = np.linspace(0, 8, res)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    Z_wave = np.zeros_like(X_grid)
    for i in range(res):
        for j in range(res):
            x_p, y_p = X_grid[i, j], Y_grid[i, j]
            dist = np.sqrt((x_p - pos_roteador_x)**2 + (y_p - pos_roteador_y)**2)
            
            # Simulação física de barreiras estruturais da planta (Paredes divisórias)
            paredes_travessia = 0
            # Parede horizontal central (Y = 4.0)
            if (4.0 <= y_p <= 4.3) and not (3.5 <= x_p <= 6.5): 
                paredes_travessia += 1
            # Parede vertical divisória de quartos (X = 5.0, Y > 4.0)
            if (4.8 <= x_p <= 5.2) and (y_p >= 4.0):
                paredes_travessia += 1

            onda_senoidal = np.sin(dist * 7.0) * 3.5 
            rssi_base = estimar_rssi(dist, obstaculos=paredes_travessia + (2 if dist > 4 else 0))
            Z_wave[i, j] = rssi_base + onda_senoidal

    # Gráfico de Contorno (Ondas)
    fig_wave = go.Figure(data=go.Contour(
        z=Z_wave,
        x=x_vals,
        y=y_vals,
        colorscale='Viridis',
        ncontours=35,
        colorbar=dict(title="RSSI (dBm)")
    ))

    # Desenho das Paredes Estruturais da Planta Baixa (Linhas Brancas)
    # Contorno Externo da Casa
    fig_wave.add_shape(type="rect", x0=0, y0=0, x1=10, y1=8, line=dict(color="white", width=3))
    # Parede Horizontal Central (Corredor / Divisão Social x Íntima)
    fig_wave.add_shape(type="line", x0=0, y0=4, x1=3.5, y1=4, line=dict(color="white", width=2))
    fig_wave.add_shape(type="line", x0=6.5, y0=4, x1=10, y1=4, line=dict(color="white", width=2))
    # Parede Vertical (Divisão entre Quartos)
    fig_wave.add_shape(type="line", x0=5, y0=4, x1=5, y1=8, line=dict(color="white", width=2))
    # Parede da Cozinha / Banheiro
    fig_wave.add_shape(type="line", x0=3.5, y0=0, x1=3.5, y1=2.5, line=dict(color="white", width=2))

    # Rótulos dos Cômodos na Planta
    fig_wave.add_annotation(x=2.0, y=6.0, text="<b>Sala de Estar</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=7.5, y=6.0, text="<b>Cozinha / Área</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=2.5, y=2.0, text="<b>Quarto 1</b>", showarrow=False, font=dict(color="white", size=11))
    fig_wave.add_annotation(x=7.5, y=2.0, text="<b>Quarto 2 (Suíte)</b>", showarrow=False, font=dict(color="white", size=11))

    # Ponto do Roteador Wi-Fi
    fig_wave.add_trace(go.Scatter(
        x=[pos_roteador_x],
        y=[pos_roteador_y],
        mode='markers+text',
        marker=dict(size=16, color='red', symbol='star'),
        text=["Roteador Wi-Fi"],
        textposition="top center"
    ))

    fig_wave.update_layout(
        title="Planta Baixa Residencial & Propagação de Ondas em Tempo Real",
        xaxis_title="Largura (m)",
        yaxis_title="Comprimento (m)",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1b212c",
        font=dict(color="white"),
        height=550,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_wave, use_container_width=True)

    st.success(
        "🏠 **Simulação Realista:** As linhas brancas delimitam os cômodos da residência (Salas, Quartos e Cozinha). "
        "À medida que alteras a posição do roteador (estrela vermelha), as frentes de onda sofrem atenuação e desvio "
        "ao colidirem com as paredes estruturais, ilustrando perfeitamente o comportamento físico real de propagação em campo."
    )

# =========================
# 2. GERADOR DE PLANTA BAIXA AUTOMATIZADO
# =========================
elif pagina == "Gerador de Planta Baixa (500Mb)":
    st.subheader("🏠 Planta Baixa Automatizada por Tipologia & Plano de 500 Mbps")
    st.write(
        "Selecione a configuração da residência e onde o roteador está instalado. "
        "O sistema mapeia automaticamente os cômodos, calculando a atenuação de sinal e a taxa útil estimada."
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

        gerar_planta = st.form_submit_button("Gerar Mapeamento da Planta e Análise")

    if gerar_planta:
        st.markdown("---")
        st.markdown(f"### 📐 Relatório Estrutural para: *{tipo_casa}*")
        st.info(f"Roteador posicionado na **{local_roteador}** | Plano de referência: **{banda_contratada} Mbps** ({frequencia_wi_fi})")

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
        else:
            comodos = [
                {"nome": "Ambiente Único / Estúdio", "distancia": 2.0, "paredes": 0}
            ]

        cols = st.columns(2)
        for idx, comodo in enumerate(comodos):
            fator_freq = 0.0 if "5 GHz" in frequencia_wi_fi else -4.0
            rssi_calculado = estimar_rssi(comodo["distancia"], comodo["paredes"]) + fator_freq
            velocidade_estimada = estimar_velocidade_500mb(rssi_calculado)
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
            "**Diagnóstico para o Suporte Técnico N1:** Os cômodos com mais de 2 barreiras físicas "
            "apresentam atenuação relevante no plano de 500Mb. Recomenda-se o reposicionamento central do roteador "
            "ou a aplicação de nós repetidores mesh para eliminar áreas de sombra."
        )

# =========================
# 3. SIMULAÇÃO 3D INTERATIVA
# =========================
elif pagina == "Simulação 3D Interativa (Casa Tradicional)":
    st.subheader("🧊 Simulação 3D Didática: Propagação de Ondas em Casa Tradicional")
    st.write(
        "Visualize em formato tridimensional interativo como o sinal de Wi-Fi se propaga e sofre atenuação "
        "ao atravessar as divisões físicas de uma habitação tradicional."
    )

    c_obs1, c_obs2 = st.columns(2)
    with c_obs1:
        obstaculos_3d = st.slider("Barreiras Estruturais / Paredes Interceptadas", 0, 5, 2, 1, key="slider_3d_obst")
    with c_obs2:
        potencia_tx = st.selectbox("Potência de Transmissão do Roteador", ["Padrão (20 dBm)", "Alta Potência (23 dBm)"])

    n = 25
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 8, n)
    X3D, Y3D = np.meshgrid(x, y)
    rx, ry = 3.0, 4.0
    
    Z3D = np.zeros_like(X3D, dtype=float)
    for i in range(n):
        for j in range(n):
            dist = np.sqrt((X3D[i,j] - rx)**2 + (Y3D[i,j] - ry)**2)
            fator_pot = 3.0 if "Alta Potência" in potencia_tx else 0.0
            Z3D[i,j] = estimar_rssi(dist, obstaculos_3d) + fator_pot

    fig3d = go.Figure(data=[go.Surface(
        z=Z3D,
        x=X3D,
        y=Y3D,
        colorscale='RdYlGn',
        cmin=-90,
        cmax=-40,
        colorbar=dict(title="RSSI (dBm)")
    )])

    fig3d.add_trace(go.Scatter3d(
        x=[rx], y=[ry], z=[-40],
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='diamond'),
        text=["Roteador Principal"],
        textposition="top center"
    ))

    fig3d.update_layout(
        title="Modelo Volumétrico 3D de Atenuação de Sinal",
        scene=dict(
            xaxis_title="Largura da Residência (m)",
            yaxis_title="Comprimento da Residência (m)",
            zaxis_title="Potência do Sinal (dBm)",
            xaxis=dict(backgroundcolor="#1b212c", color="white"),
            yaxis=dict(backgroundcolor="#1b212c", color="white"),
            zaxis=dict(backgroundcolor="#1b212c", color="white"),
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        height=550
    )

    st.plotly_chart(fig3d, use_container_width=True)

# =========================
# 4. CONSULTORIA COMERCIAL & UPGRADE MESH (ISP)
# =========================
elif pagina == "Consultoria Comercial & Upgrade Mesh (ISP)":
    st.subheader("💼 Módulo Comercial ISP: Monetização de Rede & Upsell Mesh")
    st.write(
        "Ferramenta de inteligência de negócios para provedores de internet. "
        "Com base na reclamação de lentidão do cliente, o sistema calcula a necessidade estrutural de nós Mesh "
        "e gera uma **Proposta Comercial automatizada** para aumentar o ticket médio e eliminar o cancelamento (Churn)."
    )

    with st.form("form_comercial"):
        c1, c2 = st.columns(2)
        with c1:
            cliente_comercial = st.text_input("Nome do Assinante", value="Cliente Residencial")
            plano_atual = st.selectbox("Plano Contratado", [200, 300, 500, 700, 1000], index=2)
            comodo_critico = st.selectbox("Cômodo com Relato de Sombra/Lentidão", ["Quarto dos Fundos", "Área Gourmet / Edícula", "Escritório Superior", "Cozinha"])
        with c2:
            dist_critica = st.slider("Distância estimada do cômodo crítico (m)", 4.0, 25.0, 9.0, 0.5)
            paredes_criticas = st.slider("Paredes de alvenaria no trajeto", 2, 6, 3, 1)
            valor_mensalidade_mesh = st.number_input("Valor da Mensalidade do Nó Mesh Extra (R$)", value=19.90, step=5.0)

        gerar_proposta = st.form_submit_button("Gerar Proposta Comercial & Análise de Viabilidade Mesh")

    if gerar_proposta:
        rssi_critico = estimar_rssi(dist_critica, paredes_criticas)
        velocidade_atual_critica = estimar_velocidade_500mb(rssi_critico)
        
        nos_recomendados = 1 if paredes_criticas <= 3 else 2
        viabilidade_churn = "ALTO RISCO DE CANCELAMENTO (Cliente insatisfeito com Wi-Fi nos fundos)" if rssi_critico < -70 else "Moderado"

        st.markdown("---")
        st.markdown("### 📊 Relatório de Viabilidade e Proposta para o Provedor")

        st.markdown(f"""
        <div class="commercial-card">
            <h3 style="color: #58a6ff; margin-top: 0;">🎯 Diagnóstico Técnico-Comercial para: {cliente_comercial}</h3>
            <p><b>Status do Assinante:</b> <span style="color: #f85149;">{viabilidade_churn}</span></p>
            <hr style="border-color: #30363d;">
            <p><b>Plano Contratado:</b> {plano_atual} Mbps | <b>Velocidade Real Entregue no Cômodo Crítico ({comodo_critico}):</b> <span style="color: #f85149; font-size: 16px;"><b>{velocidade_atual_critica} Mbps</b></span></p>
            <p><b>Causa Raiz (IA):</b> Atenuação severa de <b>{abs(-40 - rssi_critico):.1f} dB</b> devido a {paredes_criticas} paredes e distância de {dist_critica}m.</p>
            <hr style="border-color: #30363d;">
            <h4 style="color: #3fb950; margin-bottom: 5px;">💡 Solução Recomendada para o ISP:</h4>
            <p>Instalação de <b>{nos_recomendados} Roteador(es) Wi-Fi Mesh adicional(is)</b> para entrega integral dos {plano_atual} Mbps contratados em 100% da residência.</p>
            <h4 style="color: #58a6ff; margin-bottom: 5px; margin-top: 15px;">💰 Proposta de Upsell Gerada Automaticamente:</h4>
            <p style="font-size: 18px; color: #ffffff;"><b>Adicional de R$ {valor_mensalidade_mesh:.2f} / mês</b> no plano do cliente (Serviço de Cobertura Total Garantida).</p>
            <p style="font-size: 12px; color: #8b949e; margin-bottom: 0;">Retorno financeiro estimado para o provedor: Zero Churn + Fidelização de Longo Prazo.</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("Proposta comercial calculada e formatada com sucesso!")

# =========================
# 5. SIMULADOR TÉRMICO GERAL
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
    ax.scatter([largur
