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
def analyze_video_with_openai(frames, analysis_type, max_frames_to_analyze=10):
    try:
        if not OPENAI_API_KEY:
            return "Erro: Chave API da OpenAI não encontrada."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        if analysis_type == "Profissional":
            prompt = "Forneça uma análise profissional e detalhada destes frames de vídeo, descrevendo elementos visuais, composição, movimento, cores e possíveis contextos ou narrativas."
        else:
            prompt = "Analise estes frames de vídeo de forma humorística, zuando o conteúdo, inventando histórias engraçadas ou fazendo comentários sarcásticos."
        
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
    uploaded_file = st.file_uploader("Escolha um vídeo...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Salvar o vídeo temporariamente
        video_path = Path(tempfile.gettempdir()) / "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Exibir o vídeo
        st.video(str(video_path))
        
        # Seleção do tipo de análise
        analysis_type = st.selectbox("Escolha o tipo de análise visual:", ("Profissional", "Brincadeira"))
        max_frames_to_analyze = st.slider(
            "Número máximo de frames para análise:",
            min_value=1,
            max_value=50,
            value=10
        )
        
        # Checkbox para transcrição
        transcribe = st.checkbox("Incluir transcrição de áudio")
        
        if st.button("Analisar Vídeo"):
            with st.spinner("Processando vídeo..."):
                # Extrair frames
                frames = extract_frames(str(video_path))
                if frames is None:  # Verificação explícita de None
                    return  # Para o processamento se os frames não forem extraídos
                
                st.subheader("Frames Extraídos")
                st.write(f"Total de frames extraídos: {len(frames)}")
                for i, frame in enumerate(frames[:5]):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame {i+1}", use_container_width=True)
                
                # Análise visual
                visual_result = analyze_video_with_openai(frames, analysis_type, max_frames_to_analyze)
                st.subheader(f"Análise Visual ({analysis_type})")
                st.write(visual_result)
                
                # Transcrição de áudio (se selecionada)
                if transcribe:
                    transcription = transcribe_audio(str(video_path))
                    st.subheader("Transcrição de Áudio")
                    st.write(transcription)
                    
                    # Criar e exibir legendas
                    video = VideoFileClip(str(video_path))
                    vtt_file = create_vtt(transcription, video.duration)
                    with open(vtt_file, "r") as f:
                        st.download_button("Baixar legendas (VTT)", f.read(), file_name="captions.vtt")
                    os.remove(vtt_file)
        
        # Limpar arquivo temporário
        if os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == "__main__":
    main()