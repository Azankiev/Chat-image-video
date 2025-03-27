import streamlit as st
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os
from openai import OpenAI

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave API do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração inicial do Streamlit
st.title("Análise de Imagens com OpenAI Multimodal")
st.write("Faça upload de uma imagem e escolha o tipo de análise que deseja!")

# Função para converter imagem para base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Função para analisar imagem com OpenAI
def analyze_image_with_openai(image_base64, analysis_type):
    try:
        if not OPENAI_API_KEY:
            return "Erro: Chave API da OpenAI não encontrada. Configure o arquivo .env corretamente."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Definir o prompt com base no tipo de análise
        if analysis_type == "Profissional":
            prompt = "Forneça uma análise profissional e detalhada desta imagem, descrevendo elementos visuais, composição, cores e possíveis contextos ou significados."
        else:  # Brincadeira
            prompt = "Analise esta imagem de forma humorística, zuando o conteúdo, inventando histórias engraçadas ou fazendo comentários sarcásticos sobre o que está acontecendo."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use o modelo que você tem acesso
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Erro ao analisar a imagem: {str(e)}"
        st.error(error_msg)
        return error_msg

# Interface do Streamlit
def main():
    # Upload da imagem
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Abrir e exibir a imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem carregada", use_container_width=True)
        
        # Seleção do tipo de análise
        analysis_type = st.selectbox(
            "Escolha o tipo de análise:",
            ("Profissional", "Brincadeira")
        )
        
        # Botão para analisar
        if st.button("Analisar Imagem"):
            with st.spinner("Analisando a imagem..."):
                # Converter imagem para base64
                img_base64 = image_to_base64(image)
                
                # Obter análise do modelo com base na escolha
                result = analyze_image_with_openai(img_base64, analysis_type)
                
                # Exibir resultado
                st.subheader(f"Análise da Imagem ({analysis_type})")
                st.write(result)

if __name__ == "__main__":
    main()