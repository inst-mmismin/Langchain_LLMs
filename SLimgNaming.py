import time 
import tempfile
import requests
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import ImageCaptionLoader

def translate(text, deepl_key, target='KO'): 
    url = 'https://api-free.deepl.com/v2/translate'
    authorization = f'DeepL-Auth-Key {deepl_key}'
    headers = {'Authorization': authorization}
    data = {'text': text, 'target_lang': target}
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        translated_text = response.json()['translations'][0]['text']
        return translated_text
    else:
        return "Translation failed"
    
def load_openaiLLM(prompt, temp, max_length, **submit_info):
    from langchain.llms import OpenAI
    openai_api_key = submit_info['openai_api_key'] 
    llm = OpenAI(openai_api_key=openai_api_key, temperature=temp, max_tokens=max_length) 
    openaiLLM = LLMChain(prompt=prompt, llm=llm) 
    return openaiLLM

def run_LLMs(openaiLLM, question, deepl_key):
    s = time.time()
    res = openaiLLM.run(question)
    duration = time.time() - s

    results = {'res': res, 'res_ko': translate(res, deepl_key), 'duration': duration} 
    return results
        
def show_results(results): 
    results_ko = results['res_ko'].replace('.', '').replace('"', '')
    st.markdown(f"""
        <h3 style='text-align: center; color: Black;'>
            "{results_ko}"
        </h3>
        """, unsafe_allow_html=True)
    

st.set_page_config(page_title="제목학원 LM Ver.")
st.title("돌아온 제목학원")
st.caption("이제 센스는 LLM이 챙겨주겠지! feat.🦜🔗")

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'submit_info' not in st.session_state:
    st.session_state.submit_info = {} 

with st.sidebar:
    st.header("API 도우미")
    st.markdown("1. OpenAI API : [링크](https://platform.openai.com/account/api-keys)")
    st.markdown("2. DeepL API : [링크](https://www.deepl.com/ko/account/summary)")

    st.header("API 정보 및 세팅 입력") 
    
    if not st.session_state.submitted:
        with st.form(key='api_form'):
            openai_api_key = st.text_input("1. OpenAI API Key ", type="password")
            deepl_api_key = st.text_input("2. DeepL API Key", type="password")

            tmp = st.slider('temperature', min_value=0.0, max_value=2.0, value=1.2, step=0.1)
            max_len = st.slider('최대 토큰 길이', min_value=4, max_value=20, value=16, step=1)
            
            submit_button = st.form_submit_button("제출하기!🙌")
            
            if submit_button:
                st.session_state.submitted = True

                st.session_state.submit_info = {
                    'temp': tmp,
                    'max_length': max_len,
                    'openai_api_key': openai_api_key,
                    'deepl_api_key': deepl_api_key,
                }
                
                st.rerun()  
    else:
        st.markdown("<h4 style='text-align: left; color: black;'>제출됨!🤟</h4>", unsafe_allow_html=True)

    st.header("만든 사람")
    st.markdown("😄 정정민")
    st.markdown("📗 블로그 : [링크](https://blog.naver.com/mmismin/223008610969)")
    

uploaded_image = st.file_uploader("🏞️ 이미지를 올려주세요", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_image.read())
    
    st.image(tfile.name, caption="다양한 사진을 올려주세요~!")

    loader = ImageCaptionLoader(images=[tfile.name])
    question = loader.load()[0].page_content.replace(' [SEP]', '.')
    question_ko = translate(question, st.session_state.submit_info['deepl_api_key'])

    st.markdown("<h4 style='text-align: left; color: black;'>이미지 설명 👇</h4>", unsafe_allow_html=True)
    st.markdown(f"{question_ko}")

    template = """Question: You're a comedian with a great sense of humor, 
    especially when it comes to reading photo descriptions 
    and coming up with very witty 10-character titles to go with them. 
    People should laugh out loud when they hear your answer.
    For example, for an image of Santa Claus sprawled out on a bed, 
    you might title it: "OMG, I had overslept... it's December 26th." 
    As an another example, an image of an athlete lying on the ground with his arms up, 
    you might title it: "please, Get me the remote controller..". 
    Now, can you come up with a title for this kind of photo description? {question}
    Answer: I'll make a witty title. My title is """

    prompt = PromptTemplate(template=template, input_variables=["question"])

start_button = st.button('시작하기!')
if start_button and uploaded_image is not None: 
    openaiLLM = load_openaiLLM(prompt, **st.session_state.submit_info)
    res = run_LLMs(openaiLLM, question, st.session_state.submit_info['deepl_api_key'])
    show_results(res)

