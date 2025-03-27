import streamlit as st
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os
from openai import OpenAI
import cv2
import tempfile

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave API do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração inicial do Streamlit
st.title("Análise de Vídeos com OpenAI Multimodal")
st.write("Faça upload de um vídeo e escolha o tipo de análise que deseja!")

# Função para converter frame (imagem) para base64
def frame_to_base64(frame):
    # Converter frame do OpenCV (BGR) para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Converter para objeto PIL
    image = Image.fromarray(frame_rgb)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Função para extrair frames de um vídeo
def extract_frames(video_path, max_frames=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)  # Pegar frames espaçados
    
    for i in range(max_frames):
        frame_pos = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

# Função para analisar frames com OpenAI
def analyze_video_with_openai(frames, analysis_type):
    try:
        if not OPENAI_API_KEY:
            return "Erro: Chave API da OpenAI não encontrada. Configure o arquivo .env corretamente."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Definir o prompt com base no tipo de análise
        if analysis_type == "Profissional":
            prompt = "Forneça uma análise profissional e detalhada destes frames de vídeo, descrevendo elementos visuais, composição, movimento, cores e possíveis contextos ou narrativas."
        else:  # Brincadeira
            prompt = "Analise estes frames de vídeo de forma humorística, zuando o conteúdo, inventando histórias engraçadas ou fazendo comentários sarcásticos sobre o que está acontecendo."
        
        # Converter frames para base64
        frame_base64_list = [frame_to_base64(frame) for frame in frames]
        
        # Preparar conteúdo para a API (enviar múltiplos frames)
        content = [{"type": "text", "text": prompt}]
        for frame_base64 in frame_base64_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_base64}"}})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use o modelo que você tem acesso
            messages=[{"role": "user", "content": content}],
            max_tokens=1000  # Aumentei por causa de múltiplos frames
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Erro ao analisar o vídeo: {str(e)}"
        st.error(error_msg)
        return error_msg

# Interface do Streamlit
def main():
    # Upload do vídeo
    uploaded_file = st.file_uploader("Escolha um vídeo...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Salvar o vídeo temporariamente para processamento
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Exibir o vídeo
        st.video(video_path)
        
        # Seleção do tipo de análise
        analysis_type = st.selectbox(
            "Escolha o tipo de análise:",
            ("Profissional", "Brincadeira")
        )
        
        # Botão para analisar
        if st.button("Analisar Vídeo"):
            with st.spinner("Extraindo frames e analisando o vídeo..."):
                # Extrair frames
                frames = extract_frames(video_path)
                if not frames:
                    st.error("Erro ao processar o vídeo.")
                    return
                
                # Mostrar os frames extraídos
                st.subheader("Frames Extraídos")
                for i, frame in enumerate(frames):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {i+1}", use_container_width=True)
                
                # Analisar os frames
                result = analyze_video_with_openai(frames, analysis_type)
                
                # Exibir resultado
                st.subheader(f"Análise do Vídeo ({analysis_type})")
                st.write(result)
        
        # Remover o arquivo temporário após uso
        os.unlink(video_path)

if __name__ == "__main__":
    main()