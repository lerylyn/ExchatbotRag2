import streamlit as st
import pandas as pd
from pypdf import PdfReader
from openai import AzureOpenAI

# Primeira Versão

st.set_page_config(
    page_title="Suporte de TI - Chatbot RAG",
    page_icon="🛠️",
    layout="wide"
)

# Inicialização do estado de sessão (State)
if "chat_iniciado" not in st.session_state:
    st.session_state.chat_iniciado = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conteudo_base" not in st.session_state:
    st.session_state.conteudo_base = ""


def extrair_texto_arquivo(uploaded_file) -> str:
    """Extrai texto dos formatos .txt, .csv ou .pdf."""
    nome = uploaded_file.name.lower()
    try:
        if nome.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif nome.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)
        elif nome.endswith(".pdf"):
            leitor = PdfReader(uploaded_file)
            texto = ""
            for pagina in leitor.pages:
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto += texto_pagina + "\n"
            return texto
    except Exception as e:
        st.sidebar.error(f"Erro ao processar o arquivo: {e}")
        return ""
    return ""


# ==========================================
# BARRA LATERAL: Configurações e Conexão
# ==========================================
with st.sidebar:
    st.title("⚙️ Configurações da IA")
    st.caption("Microsoft Foundry / Azure OpenAI Service")

    # Campos de Conexão (Persistem entre chats via session_state)
    endpoint = st.text_input(
        "Endpoint Azure:",
        value=st.session_state.get("saved_endpoint", ""),
        placeholder="https://seu-recurso.openai.azure.com/",
        type="default"
    )
    api_key = st.text_input(
        "API Key:",
        value=st.session_state.get("saved_key", ""),
        placeholder="Cole sua API Key",
        type="password"
    )
    modelo = st.text_input(
        "Nome do Deployment / Modelo:",
        value=st.session_state.get("saved_model", "gpt-4o-mini"),
        placeholder="ex: gpt-4o-mini"
    )

    st.divider()

    # Persona do Assistente
    persona_padrao = (
        "Você é um Especialista de Suporte Técnico de TI nível 2/3. "
        "Seu tom é profissional, analítico, objetivo e didático. "
        "Utilize a base de conhecimento anexada para resolver os chamados e incidentes "
        "com passos estruturados de troubleshooting. Se a informação não estiver na base nem for "
        "resolvível com boas práticas de TI, avise claramente ao solicitante."
    )
    persona = st.text_area(
        "Definição da Persona (Prompt do Sistema):",
        value=persona_padrao,
        height=140
    )

    # Upload da Base de Conhecimento (RAG)
    st.divider()
    arquivo_rag = st.file_uploader(
        "Base de Conhecimento (RAG):",
        type=["pdf", "txt", "csv"],
        help="Envie documentações de procedimentos operacionais, manuais ou FAQs."
    )

    # Processamento do arquivo
    if arquivo_rag is not None:
        st.session_state.conteudo_base = extrair_texto_arquivo(arquivo_rag)
        st.success(f"Base carregada: {arquivo_rag.name}")
    else:
        st.session_state.conteudo_base = ""

    # Botão para Inicializar o Chat
    if st.button("🚀 Iniciar / Atualizar Chat", use_container_width=True, type="primary"):
        if not endpoint or not api_key or not modelo:
            st.error("Preencha Endpoint, API Key e Nome do Modelo.")
        else:
            # Salva credenciais na sessão
            st.session_state.saved_endpoint = endpoint
            st.session_state.saved_key = api_key
            st.session_state.saved_model = modelo
            st.session_state.persona = persona
            st.session_state.chat_iniciado = True
            st.rerun()

    # Botão Novo Chat: preserva credenciais, limpa histórico
    if st.session_state.chat_iniciado:
        if st.button("🔄 Novo Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==========================================
# PAINEL PRINCIPAL: Área do Chatbot
# ==========================================
st.subheader("🛠️ Suporte de TI — Central de Atendimento Inteligente")

if not st.session_state.chat_iniciado:
    st.info("👈 Preencha os parâmetros de conexão na barra lateral e clique em **'Iniciar / Atualizar Chat'** para começar.")
else:
    # Exibe o histórico de mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada do Usuário
    if prompt_usuario := st.chat_input("Descreva o incidente ou problema técnico..."):
        # Adiciona e exibe mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        # Montagem do contexto do sistema com RAG (in-context injection)
        system_instruction = st.session_state.persona
        if st.session_state.conteudo_base:
            system_instruction += (
                f"\n\n--- DOCUMENTAÇÃO DE SUPORTE (RAG) ---\n"
                f"{st.session_state.conteudo_base}\n"
                f"--- FIM DA DOCUMENTAÇÃO ---"
            )

        mensagens_api = [{"role": "system", "content": system_instruction}]
        for m in st.session_state.messages:
            mensagens_api.append({"role": m["role"], "content": m["content"]})

        # Chamada ao Azure OpenAI
        with st.chat_message("assistant"):
            try:
                client = AzureOpenAI(
                    azure_endpoint=st.session_state.saved_endpoint.strip(),
                    api_key=st.session_state.saved_key.strip(),
                    api_version="2024-06-01"
                )

                response = client.chat.completions.create(
                    model=st.session_state.saved_model.strip(),
                    messages=mensagens_api,
                    temperature=0.2
                )
                resposta_ia = response.choices[0].message.content
                st.markdown(resposta_ia)

                # Salva a resposta no histórico da sessão
                st.session_state.messages.append({"role": "assistant", "content": resposta_ia})

            except Exception as erro:
                st.error(f"Falha na comunicação com a Azure: {erro}")
