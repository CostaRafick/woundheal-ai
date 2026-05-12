import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="WoundHeal-AI Pro", layout="wide")

st.title("🐍LATOX_IA: WoundHeal analyse")
st.markdown("Ferramenta do projeto LATOX para automatizar as análises. Faça upload, ajuste a sensibilidade ou desenhe a área manualmente quando necessário.")

st.sidebar.header("Configurações do Ensaio")
escala = st.sidebar.number_input("Escala (pixels por µm)", value=0.5, step=0.1)

arquivos_upados = st.file_uploader("Selecione as fotos (Ordem cronológica)", type=['png', 'jpg', 'jpeg', 'tif'], accept_multiple_files=True)

if arquivos_upados:
    st.success(f"{len(arquivos_upados)} imagens carregadas. Iniciando processamento...")
    resultados = []
    aba_imagens, aba_graficos = st.tabs(["Visualização e Ajustes", "Resultados e Gráficos"])
    
    with aba_imagens:
        for arquivo in arquivos_upados:
            with st.container():
                st.subheader(f"Arquivo: {arquivo.name}")
                
                # Escolha do método POR IMAGEM
                modo = st.radio(
                    "Método para esta imagem:",
                    ["Automático (Textura/Bordas)", "Automático (Intensidade)", "✍️ Desenho Manual (Polígono)"],
                    key=f"modo_{arquivo.name}",
                    horizontal=True
                )
                
                imagem_pil = Image.open(arquivo).convert('L')
                img_array = np.array(imagem_pil)
                orig_h, orig_w = img_array.shape[:2]
                
                area_um2 = 0
                
                if "Desenho Manual" in modo:
                    st.info("🔹 **Como usar:** Clique nas bordas da ferida para criar os pontos do polígono. Dê um **duplo-clique** no último ponto para fechar o desenho da área.")
                    
                    canvas_w = 800
                    canvas_h = int(orig_h * (canvas_w / orig_w))
                    
                    imagem_fundo = Image.open(arquivo).convert('RGB')
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.4)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=imagem_fundo,
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="polygon",
                        key=f"canvas_{arquivo.name}",
                    )
                    
                    if canvas_result.image_data is not None:
                        mascara_desenho = canvas_result.image_data[:, :, 3] > 0
                        area_canvas_pixels = np.sum(mascara_desenho)
                        
                        fator_escala_area = (orig_w / canvas_w) ** 2
                        area_real_pixels = area_canvas_pixels * fator_escala_area
                        area_um2 = area_real_pixels * (1 / escala)**2
                    
                    if area_um2 > 0:
                        st.success(f"Área Desenhada Identificada: {area_um2:.2f} µm²")

                else:
                    # MÉTODOS AUTOMÁTICOS
                    limiar_manual = st.slider(
                        "Ajuste Fino da IA", 
                        min_value=0, max_value=255, value=0, 
                        help="0 = Automático. Use para guiar a IA se ela errar um pouco a borda.",
                        key=f"slider_{arquivo.name}" 
                    )
                    
                    # 1. Máscara Microscópio
                    img_blur_forte = cv2.GaussianBlur(img_array, (31, 31), 0)
                    _, mascara_campo = cv2.threshold(img_blur_forte, 40, 255, cv2.THRESH_BINARY)
                    contornos_campo, _ = cv2.findContours(mascara_campo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    mascara_campo_limpo = np.zeros_like(img_array)
                    if contornos_campo:
                        maior_campo = max(contornos_campo, key=cv2.contourArea)
                        cv2.drawContours(mascara_campo_limpo, [maior_campo], -1, 255, thickness=cv2.FILLED)
                        kernel_encolher = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
                        mascara_campo_limpo = cv2.erode(mascara_campo_limpo, kernel_encolher, iterations=1)
                    
                    # 2. IA de Detecção
                    img_blur_celulas = cv2.GaussianBlur(img_array, (5, 5), 0)
                    
                    if "Textura" in modo:
                        limiar_inf = limiar_manual if limiar_manual > 0 else 15
                        bordas = cv2.Canny(img_blur_celulas, limiar_inf, limiar_inf * 3)
                        kernel_textura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        massa_celulas = cv2.dilate(bordas, kernel_textura, iterations=2)
                        massa_celulas = cv2.morphologyEx(massa_celulas, cv2.MORPH_CLOSE, kernel_textura, iterations=2)
                        mascara_ferida = cv2.bitwise_not(massa_celulas)
                    else:
                        if limiar_manual == 0:
                            _, threshold = cv2.threshold(img_blur_celulas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        else:
                            _, threshold = cv2.threshold(img_blur_celulas, limiar_manual, 255, cv2.THRESH_BINARY)
                        kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        fechamento = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel_morf, iterations=3)
                        mascara_ferida = cv2.bitwise_not(fechamento)
                    
                    # 3. Cruzamento e Área
                    mascara_ferida_final = cv2.bitwise_and(mascara_ferida, mascara_campo_limpo)
                    contornos, _ = cv2.findContours(mascara_ferida_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_colorida = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
                    if contornos:
                        maior_contorno = max(contornos, key=cv2.contourArea)
                        area_pixels = cv2.contourArea(maior_contorno)
                        area_um2 = area_pixels * (1 / escala)**2 
                        cv2.drawContours(img_colorida, [maior_contorno], -1, (255, 0, 0), 3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # CORRIGIDO AQUI: use_column_width no lugar de use_container_width
                        st.image(img_array, caption="Original", use_column_width=True)
                    with col2:
                        # CORRIGIDO AQUI: use_column_width no lugar de use_container_width
                        st.image(img_colorida, caption=f"Área Identificada: {area_um2:.2f} µm²", use_column_width=True)
                
                resultados.append({'Arquivo': arquivo.name, 'Área da Ferida (µm²)': round(area_um2, 2)})
                st.divider()

    with aba_graficos:
        st.subheader("Tabela de Resultados Consolidada")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados) # Removido o argumento aqui também por segurança
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar Tabela (CSV)", csv, "resultados_wound.csv", "text/csv")
        st.subheader("Gráfico")
        st.line_chart(df_resultados.set_index('Arquivo')['Área da Ferida (µm²)'])
else:
    st.info("Aguardando o upload das imagens...")
