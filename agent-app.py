import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

# ================= 1. 页面基本配置 =================
st.set_page_config(page_title="轨道交通教学助理", page_icon="🚆", layout="wide")

# ================= 2. 核心 AI 逻辑 =================
@st.cache_resource
def init_ai_agent():
    # 1. 环境变量配置
# 1. 环境变量配置 (从 Streamlit 加密保险箱读取)
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    except KeyError:
        st.error("⚠️ 未检测到 API Key，请在 Streamlit Cloud 的 Secrets 中配置！")
        st.stop()
        
    os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
    
    # 如果使用了本地代理，请保留你的代理配置，例如：
    # os.environ["http_proxy"] = "http://127.0.0.1:7890" 
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    
    # 2. 初始化核心组件 (⚠️ 就是这里之前漏掉了！)
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    search_tool = TavilySearchResults(k=4)
    
    # 3. 定义文件夹路径并自动创建
    pdf_dir = './materials'
    md_dir = './md_materials'

    for folder in [pdf_dir, md_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 4. 批量读取不同格式文件
    pdf_loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    
    md_loader = DirectoryLoader(md_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    md_docs = md_loader.load()
    
    # 合并所有文档
    all_docs = pdf_docs + md_docs
    
    # 5. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    splits = text_splitter.split_documents(all_docs)
    
    print(f"📊 加载完成：PDF {len(pdf_docs)}页, Markdown {len(md_docs)}个文件")

    # 6. 构建本地向量数据库
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    return llm, retriever, search_tool

llm, retriever, search_tool = init_ai_agent()

# ================= 3. 布局与侧边栏 =================
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=train", width=80)
    st.title("控制面板")
    
    # 注入姓名和学号
    st.markdown("---")
    st.markdown("### 👨‍💻 开发者信息")
    st.info(f"**姓名：** 蔡和倬\n\n**学号：** 23211119")
    st.markdown("---")
    
    st.success("✅ 系统环境已就绪")
    st.write("当前模式：混合检索 (RAG + Web)")

st.title("🚆 轨道交通移动通信系统 - 智能教学助理")

# 会话记录初始化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "同学你好！我是助教。有什么专业问题需要解答？"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 4. 交互逻辑 =================
if prompt := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 使用 st.status 展示思考过程
        with st.status("🔍 正在处理您的请求...", expanded=True) as status:
            st.write("正在检索本地 课件+Markdown 库...")
            context_docs = retriever.invoke(prompt)
            
            local_found = False
            final_answer = ""
            sources = []
            source_type = ""

            if context_docs:
                st.write("正在对比课程原文...")
                context_text = "\n\n".join([doc.page_content for doc in context_docs])
                qa_prompt = f"你是一个轨道交通助教。基于以下课件和markdown格式讲义回答：\n{context_text}\n\n问题：{prompt}\n注意：若课件未提及，请回答'【LOCAL_NOT_FOUND】'。"
                response = llm.invoke(qa_prompt).content
                
                if "【LOCAL_NOT_FOUND】" not in response:
                    final_answer = response
                    sources = list(set([os.path.basename(doc.metadata.get('source', '未知文件')) for doc in context_docs]))
                    source_type = "📚 本地知识库"
                    local_found = True

            if not local_found:
                # 给出回应，告知用户正在转向联网
                st.warning("⚠️ 本地 课件+Markdown 库未找到相关准确解答，正在为您联网搜索全网资料...")
                st.write("正在调用 Tavily 搜索引擎...")
                
                search_results = search_tool.invoke(prompt)
                web_context = "\n".join([r['content'] for r in search_results])
                web_prompt = f"你是通信工程助教。校内资料库未涉及，请根据联网搜索信息回答：\n{web_context}\n\n问题：{prompt}"
                final_answer = llm.invoke(web_prompt).content
                sources = ["Tavily 网络搜索"]
                source_type = "🌐 联网实时搜索"
            
            status.update(label="✅ 解答完成！", state="complete", expanded=False)

        # 展示最终结果
        full_response = f"{final_answer}\n\n---\n**来源：** {source_type} | **参考：** `{', '.join(sources)}`"
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})