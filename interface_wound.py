import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import gc # Importado para gestão manual de memória RAM

# Configuração da página
st.set_page_config(page_title="WoundHeal-AI Pro", layout="wide")

# Branding LATOX
st.title("🐍 LATOX_IA: WoundHeal analyse")
st.markdown("Ferramenta do projeto LATOX para automatizar as análises. Faça upload, ajuste a sensibilidade ou desenhe a área manualmente quando necessário.")

# Menu lateral
st.sidebar.header("Configurações do Ensaio")
escala = st.sidebar.number_input("Escala (pixels por µm)", value=0.5, step=0.1)

arquivos_upados = st.file_uploader("Selecione as fotos (Ordem cronológica)", type=['png', 'jpg', 'jpeg', 'tif'], accept_multiple_files=True)

if arquivos_upados:
    st.success(f"{len(arquivos_upados)} imagens carregadas. Iniciando processamento otimizado...")
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
                
                # OTIMIZAÇÃO DE MEMÓRIA: Carregamento com redimensionamento inteligente
                imagem_pil_original = Image.open(arquivo).convert('L')
                orig_w_real, orig_h_real = imagem_pil_original.size
                
                max_dim = 1000 # Limite de segurança para evitar estouro de RAM
                fator_redim = 1.0
                
                if orig_w_real > max_dim or orig_h_real > max_dim:
                    if orig_w_real > orig_h_real:
                        novo_w = max_dim
                        novo_h = int(orig_h_real * (max_dim / orig_w_real))
                        fator_redim = max_dim / orig_w_real
                    else:
                        novo_h = max_dim
                        novo_w = int(orig_w_real * (max_dim / orig_h_real))
                        fator_redim = max_dim / orig_h_real
                        
                    imagem_pil = imagem_pil_original.resize((novo_w, novo_h), Image.Resampling.LANCZOS)
                else:
                    imagem_pil = imagem_pil_original

                img_array = np.array(imagem_pil)
                orig_h, orig_w = img_array.shape[:2]
                
                area_um2 = 0
                
                if "Desenho Manual" in modo:
                    st.info("🔹 **Como usar:** Clique nas bordas da ferida para criar os pontos do polígono. Dê um **duplo-clique** no último ponto para fechar o desenho da área.")
                    
                    canvas_w = 800
                    canvas_h = int(orig_h * (canvas_w / orig_w))
                    
                    # Fundo para o canvas usando a imagem otimizada
                    imagem_fundo = Image.fromarray(img_array).convert('RGB')
                    
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
                        
                        # Compensação matemática: Canvas -> Imagem Otimizada -> Escala Real
                        fator_escala_canvas = (orig_w / canvas_w) ** 2
                        area_pixels_img_otimizada = area_canvas_pixels * fator_escala_canvas
                        area_um2 = area_pixels_img_otimizada * (1 / (escala * fator_redim))**2
                    
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
                    
                    # 1. Máscara Microscópio (Filtro de vinheta)
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
                    
                    # 3. Cruzamento e Cálculo de Área com fator de correção
                    mascara_ferida_final = cv2.bitwise_and(mascara_ferida, mascara_campo_limpo)
                    contornos, _ = cv2.findContours(mascara_ferida_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_colorida = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    
                    if contornos:
                        maior_contorno = max(contornos, key=cv2.contourArea)
                        area_pixels_otimizada = cv2.contourArea(maior_contorno)
                        # Cálculo final corrigido pela escala e pelo fator de redimensionamento
                        area_um2 = area_pixels_otimizada * (1 / (escala * fator_redim))**2 
                        cv2.drawContours(img_colorida, [maior_contorno], -1, (255, 0, 0), 3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_array, caption="Original (Otimizada)", use_column_width=True)
                    with col2:
                        st.image(img_colorida, caption=f"Área Identificada: {area_um2:.2f} µm²", use_column_width=True)
                
                resultados.append({'Arquivo': arquivo.name, 'Área da Ferida (µm²)': round(area_um2, 2)})
                
                # Limpeza agressiva de memória após cada imagem processada
                del imagem_pil_original, imagem_pil, img_array
                gc.collect()
                
                st.divider()

    with aba_graficos:
        st.subheader("Tabela de Resultados Consolidada")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados)
        
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button("Baixar Tabela (CSV)", csv, "resultados_latox.csv", "text/csv")
        
        st.subheader("Gráfico de Fechamento")
        st.line_chart(df_resultados.set_index('Arquivo')['Área da Ferida (µm²)'])
else:
    st.info("Aguardando o upload das imagens...")
