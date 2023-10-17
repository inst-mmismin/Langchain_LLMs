import time 
import tempfile
import streamlit as st

from utils import key_LLMName
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub

def set_LLMs(prompt, temp=0.5, max_length=96, **submit_info):
    LLMS = {}
    
    # openAI LLM 
    if submit_info['openai_api_key'] : 
        from langchain.llms import OpenAI
        openai_api_key = submit_info['openai_api_key'] 
        llm = OpenAI(openai_api_key=openai_api_key, temperature=temp, max_tokens=max_length) 
        LLMS['openai'] = LLMChain(prompt=prompt, llm=llm) 
    
    # cohere LLM
    if submit_info['cohere_api_key'] : 
        from langchain.llms import Cohere
        cohere_api_key = submit_info['cohere_api_key'] 
        llm = Cohere(cohere_api_key=cohere_api_key, temperature=temp, max_tokens=max_length) 
        LLMS['cohere'] = LLMChain(prompt=prompt, llm=llm) 
    
    # google vertexAI LLM
    if submit_info['google_application_credentials_file_path'] : 
        import os 
        from langchain.llms import VertexAI
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = submit_info['google_application_credentials_file_path']
        llm = VertexAI(temperature=temp, max_output_tokens=max_length) 
        LLMS['google'] = LLMChain(prompt=prompt, llm=llm)

    # huggingface LLM
    if submit_info['huggingface_api_key'] : 
        huggingface_api_key = submit_info['huggingface_api_key'] 
        for repo_id in submit_info['huggingface_repo_ids'] :
            print(repo_id)
            llm = HuggingFaceHub(
                repo_id=repo_id, model_kwargs={"temperature": temp, "max_length": max_length},
                huggingfacehub_api_token = huggingface_api_key
            )
            LLMS['huggingface_'+repo_id] = LLMChain(prompt=prompt, llm=llm)
    return LLMS

def run_LLMs(LLMS, question):
    RESULTS = {} 
    for key, llm_chain in LLMS.items() : 
        s = time.time()
        model_name = key_LLMName.get(key, key)
        res = llm_chain.run(question)
        duration = time.time() - s
        
        RESULTS.update({model_name : {'res': res, 'duration': duration}})
    return RESULTS
        
def show_results(results): 
    for model_name, value in results.items() :
        st.markdown(f"""
            <h5 style='text-align: left; color: Gray;'>
                {model_name}  
                <span style='font-size: 0.8em;'>(🕒 {value['duration']:.2f} sec)</span>
            </h5>
            """, unsafe_allow_html=True)
        st.code(value['res'])
        st.write('')
    

st.set_page_config(page_title='🦜🔗 Compare LLMs')
st.title("👍 랭체인으로 LLM 성능 비교")
st.caption("LLM이 좋다며?? 근데 왜이렇게 많아!! 나는 한번에 보고싶은데 🤩 feat.🦜🔗")

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'submit_info' not in st.session_state:
    st.session_state.submit_info = {} 

with st.sidebar:
    st.header("API 도우미")
    st.markdown("1. OpenAI API : [링크](https://platform.openai.com/account/api-keys)")
    st.markdown("2. Cohere API : [링크](https://dashboard.cohere.com/api-keys)")
    st.markdown("3. Google Application Credentials : [링크](https://console.cloud.google.com)")
    st.markdown("4. HugginFace API : [링크](https://huggingface.co/settings/tokens)")
    st.markdown("5. HugginFace repo 찾기 : [링크](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)")

    
    st.header("API 정보 입력") 
    
    if not st.session_state.submitted:
        with st.form(key='api_form'):
            openai_api_key = st.text_input("1. OpenAI API Key ", type="password")
            cohere_api_key = st.text_input("2. Cohere API Key", type="password")
            uploaded_file = st.file_uploader("3. 📁 GCP Credential JSON 파일 업로드", type=["json"])

            huggingface_api_key = st.text_input("4. HuggingFace API Key", type="password")
            huggingface_repo_ids = st.text_input("5. HuggingFace Repo ID(,로 구분하기)", value="tiiuae/falcon-7b-instruct")
            
            
            submit_button = st.form_submit_button("제출하기!🙌")
            
            if submit_button:
                st.session_state.submitted = True

                tfile_name = None
                if uploaded_file is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    tfile_name = tfile.name

                print('aaaa')

                st.session_state.submit_info = {
                    'openai_api_key': openai_api_key,
                    'cohere_api_key': cohere_api_key,
                    'google_application_credentials_file_path': tfile_name,
                    'huggingface_api_key': huggingface_api_key,
                    'huggingface_repo_ids': [r_id.strip() for r_id in huggingface_repo_ids.split(',')]
                }
                
                print('bbb')
                st.rerun()  
    else:
        st.markdown("<h4 style='text-align: center; color: black;'>제출됨!🤟</h4>", unsafe_allow_html=True)

    st.header("만든 사람")
    st.markdown("😄 정정민")
    st.markdown("📗 블로그 : [링크](https://blog.naver.com/mmismin/223008610969)")
    

st.markdown("<h4 style='text-align: center; color: black;'>아래 프롬프트를 입력하세요 👇</h4>", unsafe_allow_html=True)
question = st.text_input("", value="which team is the winner of the 2002 world cup? ")

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

start_button = st.button('시작하기!')
if start_button : 
    LLMS = set_LLMs(prompt, 0.1, 512, **st.session_state.submit_info)
    RESULTS = run_LLMs(LLMS, question)
    show_results(RESULTS)

    st.write("👍")