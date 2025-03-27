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
def analyze_image_with_openai(image_base64, analysis_type, analysis_subtype):
    try:
        if not OPENAI_API_KEY:
            return "Erro: Chave API da OpenAI não encontrada. Configure o arquivo .env corretamente."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Dicionário de prompts para diferentes tipos de análise
        prompts = {
            "Profissional": {
                "Técnica": "Forneça uma análise técnica detalhada desta imagem, focando em aspectos técnicos como composição, iluminação, enquadramento, qualidade técnica e elementos visuais.",
                "Narrativa": "Analise esta imagem sob uma perspectiva narrativa, descrevendo a história, personagens, contexto e desenvolvimento da narrativa visual.",
                "Estética": "Forneça uma análise estética desta imagem, explorando elementos visuais como cores, texturas, padrões e a beleza artística da composição.",
                "Cinematográfica": "Realize uma análise cinematográfica desta imagem, discutindo elementos como direção, fotografia, edição e técnicas cinematográficas utilizadas."
            },
            "Humorística": {
                "Sarcástica": "Analise esta imagem de forma sarcástica, fazendo comentários irônicos e bem-humorados sobre o conteúdo.",
                "Memes": "Transforme esta imagem em meme, criando descrições engraçadas e relacionando com memes populares da internet.",
                "Paródia": "Crie uma paródia humorística desta imagem, inventando uma história engraçada e exagerada baseada no conteúdo.",
                "Comédia": "Faça uma análise cômica desta imagem, adicionando elementos de humor e piadas relacionadas ao conteúdo."
            }
        }
        
        # Selecionar o prompt baseado no tipo de análise
        prompt = prompts[analysis_type][analysis_subtype]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        analysis_type = st.selectbox("Escolha o tipo de análise:", ("Profissional", "Humorística"))
        
        # Seleção do subtipo de análise baseado no tipo principal
        if analysis_type == "Profissional":
            analysis_subtype = st.selectbox(
                "Escolha o estilo de análise profissional:",
                ("Técnica", "Narrativa", "Estética", "Cinematográfica")
            )
        else:
            analysis_subtype = st.selectbox(
                "Escolha o estilo de análise humorística:",
                ("Sarcástica", "Memes", "Paródia", "Comédia")
            )
        
        # Botão para analisar
        if st.button("Analisar Imagem"):
            with st.spinner("Analisando a imagem..."):
                # Converter imagem para base64
                img_base64 = image_to_base64(image)
                
                # Obter análise do modelo com base na escolha
                result = analyze_image_with_openai(img_base64, analysis_type, analysis_subtype)
                
                # Exibir resultado
                st.subheader(f"Análise {analysis_type} - {analysis_subtype}")
                st.write(result)

if __name__ == "__main__":
    main()