from pathlib import Path
import os
from os import path as osp
import json
import cv2
import webvtt
import whisper
import moviepy
from moviepy.editor import VideoFileClip
from PIL import Image
import base64
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import io

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a chave API do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração inicial do Streamlit
st.title("Análise Multimodal de Vídeos")
st.write("Faça upload de um vídeo para análise visual e transcrição de áudio!")

# Função para converter frame para base64
def frame_to_base64(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Função para extrair todos os frames de um vídeo
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Erro: Não foi possível abrir o vídeo em {video_path}. Verifique o formato ou o arquivo.")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

# Função para transcrever áudio com Whisper
def transcribe_audio(video_path):
    try:
        model = whisper.load_model("base")
        video = VideoFileClip(video_path)
        audio_path = osp.join(tempfile.gettempdir(), "temp_audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        return result["text"]
    except Exception as e:
        return f"Erro ao transcrever áudio: {str(e)}"

# Função para criar legendas WebVTT a partir da transcrição
def create_vtt(transcription, video_duration):
    vtt = webvtt.WebVTT()
    words = transcription.split()
    chunk_size = max(1, len(words) // 5)
    for i in range(0, len(words), chunk_size):
        start_time = (i / len(words)) * video_duration
        end_time = min(((i + chunk_size) / len(words)) * video_duration, video_duration)
        caption = vtt.Caption(
            start=f"{start_time:.2f}",
            end=f"{end_time:.2f}",
            text=" ".join(words[i:i + chunk_size])
        )
        vtt.captions.append(caption)
    vtt_file = osp.join(tempfile.gettempdir(), "captions.vtt")
    vtt.save(vtt_file)
    return vtt_file

# Função para analisar frames com OpenAI
def analyze_video_with_openai(video_path, analysis_type, analysis_subtype, max_frames_to_analyze=10):
    try:
        if not OPENAI_API_KEY:
            return "Erro: Chave API da OpenAI não encontrada."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Dicionário de prompts para diferentes tipos de análise
        prompts = {
            "Profissional": {
                "Técnica": "Forneça uma análise técnica detalhada destes frames de vídeo, focando em aspectos técnicos como composição, iluminação, enquadramento, movimento de câmera e qualidade técnica.",
                "Narrativa": "Analise estes frames de vídeo sob uma perspectiva narrativa, descrevendo a história, personagens, contexto e desenvolvimento da narrativa visual.",
                "Estética": "Forneça uma análise estética destes frames de vídeo, explorando elementos visuais como cores, texturas, padrões e a beleza artística da composição.",
                "Cinematográfica": "Realize uma análise cinematográfica destes frames, discutindo elementos como direção, fotografia, edição e técnicas cinematográficas utilizadas."
            },
            "Humorística": {
                "Sarcástica": "Analise estes frames de vídeo de forma sarcástica, fazendo comentários irônicos e bem-humorados sobre o conteúdo.",
                "Memes": "Transforme estes frames em memes, criando descrições engraçadas e relacionando com memes populares da internet.",
                "Paródia": "Crie uma paródia humorística destes frames, inventando uma história engraçada e exagerada baseada no conteúdo.",
                "Comédia": "Faça uma análise cômica destes frames, adicionando elementos de humor e piadas relacionadas ao conteúdo."
            }
        }
        
        # Selecionar o prompt baseado no tipo de análise
        prompt = prompts[analysis_type][analysis_subtype]
        
        frames = extract_frames(video_path)
        if frames is None:  # Verificação explícita de None
            return "Erro: Não foi possível extrair frames do vídeo."
        
        # Limitar o número de frames para análise
        frames_to_analyze = frames[:max_frames_to_analyze] if len(frames) > max_frames_to_analyze else frames
        frame_base64_list = [frame_to_base64(frame) for frame in frames_to_analyze]
        
        content = [{"type": "text", "text": f"{prompt} (Analisando {len(frames_to_analyze)} de {len(frames)} frames)"}]
        for frame_base64 in frame_base64_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_base64}"}})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=1000
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
        # Salvar o vídeo temporariamente
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Exibir o vídeo
        st.video(uploaded_file)
        
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
        
        # Seleção do número máximo de frames
        max_frames_to_analyze = st.slider(
            "Número máximo de frames para análise:",
            min_value=1,
            max_value=50,
            value=10
        )
        
        # Botão para analisar
        if st.button("Analisar Vídeo"):
            with st.spinner("Processando vídeo..."):
                # Extrair frames
                frames = extract_frames(temp_path)
                if frames is None:  # Verificação explícita de None
                    return  # Para o processamento se os frames não forem extraídos
                
                st.subheader("Frames Extraídos")
                st.write(f"Total de frames extraídos: {len(frames)}")
                
                # Exibir os primeiros 5 frames
                for i, frame in enumerate(frames[:5]):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {i+1}", use_container_width=True)
                
                # Análise visual
                result = analyze_video_with_openai(temp_path, analysis_type, analysis_subtype, max_frames_to_analyze)
                
                # Exibir resultado
                st.subheader(f"Análise {analysis_type} - {analysis_subtype}")
                st.write(result)
                
                # Limpar arquivo temporário
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()