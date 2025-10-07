import streamlit as st
import pandas as pd

import io
import matplotlib.pyplot as plt

# Importações do LangChain atualizadas
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler


# Criação do agente com uso de cache
# O cache evita recriar o agente a cada interação, a menos que o DataFrame mude.

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def setup_agent(df: pd.DataFrame):
    """
    Inicializa a LLM DeepSeek e o Agente Pandas DataFrame.
    """
    if not DEEPSEEK_API_KEY or not DEEPSEEK_BASE_URL:
        st.error(
            "ERRO: As variáveis DEEPSEEK_API_KEY e DEEPSEEK_BASE_URL não estão configuradas."
        )
        st.stop()

    try:
        llm_deepseek = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        AGENT_PREFIX = """
    Você é um assistente de análise de dados muito experiente.
    Você tem acesso a um DataFrame pandas chamado `df`.
    Sua única ferramenta é o `python_repl` para executar código Python.

    - Para responder a perguntas, escreva e execute o código Python necessário para obter a resposta.
    - Se for solicitada uma **visualização** (ex: "histograma", "gráfico de barras", "plot"), 
      **gere o código usando a biblioteca `matplotlib.pyplot`**.
    - **NÃO** inclua `st.pyplot()` no código que você gera. Apenas gere o gráfico com Matplotlib.
    - Sempre retorne a resposta final em português.
    """

        agent = create_pandas_dataframe_agent(
            llm=llm_deepseek,
            df=df,
            verbose=False,
            agent_type="openai-tools",
            memory=memory,
            allow_dangerous_code=True,
            agent_kwargs={"prefix": AGENT_PREFIX},
        )
        return agent

    except Exception as e:
        st.error(
            f"Erro ao inicializar a LLM DeepSeek. Verifique sua chave de API. Detalhe: {e}"
        )
        st.stop()


# Configuração da página em Streamlit

st.set_page_config(page_title="Analista Arquivo CSV", layout="wide")
st.title("📊 Analista de CSV com suporte a gráficos.")
st.markdown(
    "Peça análises e **gráficos** em linguagem natural."
)

# Inicializa o histórico de chat e o agente na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None


# Solicitar a chave de API.
def input_key():
    text_input_container = st.empty()
    st_deepseek_api_key = text_input_container.text_input(
        "# Insira sua chave de API DeepSeek. #",
        key="name_input",
        type="password",
        help="Você pode obter sua chave de API em https://deepseek.ai",
    )

    if st_deepseek_api_key != "":
        text_input_container.empty()
        return st_deepseek_api_key


# Definição das variáveis globais para a chave e URL da API DeepSeek
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"
DEEPSEEK_API_KEY = input_key()


# Upload do arquivo
uploaded_file = st.sidebar.file_uploader("Upload do Arquivo CSV", type=["csv"])

if uploaded_file:
    try:
        # Decodifica o arquivo e carrega no DataFrame
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

        st.sidebar.success(
            f"Arquivo **{uploaded_file.name}** carregado! ({len(df)} linhas)"
        )
        st.sidebar.dataframe(df.head(), use_container_width=True)

        # Cria o agente para o DataFrame carregado
        st.session_state.agent = setup_agent(df)

        # Mensagem inicial de boas-vindas
        if not st.session_state.messages:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Olá! Seu arquivo foi carregado. Pergunte algo como: 'Quais os tipos de dados?' ou 'Crie um gráfico de barras dispersão das colunas ....'.",
                }
            )
    except Exception as e:
        st.sidebar.error(f"Erro ao ler o arquivo CSV: {e}")
        st.session_state.agent = None
else:
    st.info(
        "Por favor, carregue um arquivo CSV na barra lateral para começar a análise."
    )

# --- 5. LÓGICA DO CHAT E PLOTAGEM ---

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Processa nova entrada do usuário
if prompt := st.chat_input("Digite sua pergunta de análise ou plotagem..."):
    if st.session_state.agent is None:
        st.warning("Carregue um arquivo CSV antes de fazer uma pergunta.")
        st.stop()

    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoca o Agente e processa a resposta
    with st.chat_message("assistant"):
        #st_callback = StreamlitCallbackHandler(st.container())

        with st.spinner("DeepSeek está pensando e gerando a resposta..."):
            try:
                # O agente invoca a LLM para gerar e executar o código Python
                result = st.session_state.agent.invoke(
                    {"input": prompt} #, {"callbacks": [st_callback]}
                )

                response_text = result.get(
                    "output", "Não foi possível gerar uma resposta clara."
                )

                # --- LÓGICA DE EXIBIÇÃO DO GRÁFICO (PONTO CENTRAL DA REFATORAÇÃO) ---

                # 1. Captura a figura atual do Matplotlib que o agente criou em background
                fig = plt.gcf()

                # 2. Verifica se a figura tem algum conteúdo (eixos) para ser exibido
                if fig.get_axes():
                    st.pyplot(fig)  # 3. Renderiza a figura no Streamlit

                # Exibe a resposta textual do agente
                st.markdown(response_text)

                # Adiciona a resposta de texto ao histórico
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

            except Exception as e:
                error_message = f"Ocorreu um erro durante a análise. Tente reformular a pergunta. (Detalhe: {e})"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
            finally:
                # Limpa todas as figuras do Matplotlib para a próxima execução
                plt.close("all")
