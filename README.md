# Análise Multimodal de Vídeos

Este é um aplicativo Streamlit que permite analisar vídeos de forma multimodal, combinando análise visual com transcrição de áudio.

## Funcionalidades

- Upload de vídeos (formatos: MP4, AVI, MOV)
- Análise visual dos frames do vídeo usando GPT-4
- Transcrição de áudio usando Whisper
- Geração de legendas em formato WebVTT
- Dois modos de análise: Profissional e Brincadeira

## Requisitos

- Python 3.8+
- OpenAI API Key
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/analise-videos.git
cd analise-videos
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Crie um arquivo `.env` na raiz do projeto e adicione sua chave API da OpenAI:
```
OPENAI_API_KEY=sua-chave-api-aqui
```

## Como Usar

1. Execute o aplicativo:
```bash
streamlit run video_rag.py
```

2. Acesse o aplicativo no navegador (geralmente em http://localhost:8501)

3. Faça upload de um vídeo e escolha as opções de análise

## Estrutura do Projeto

- `video_rag.py`: Arquivo principal do aplicativo
- `requirements.txt`: Lista de dependências
- `.env`: Arquivo de configuração para chaves API (não incluído no repositório)
- `.gitignore`: Configuração de arquivos ignorados pelo Git

## Contribuindo

Sinta-se à vontade para contribuir com o projeto! Abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes. 