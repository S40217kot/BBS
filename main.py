import streamlit as st

st.set_page_config(page_title="Home")

if st.sidebar.button("AR そろばん"):
    st.switch_page("pages/soroban.py")
if st.sidebar.button("AR テルミン"):
    st.switch_page("pages/terumin.py")
if st.sidebar.button("AR パレット"):
    st.switch_page("pages/paretto.py")
if st.sidebar.button("AR 人体模型"):
    st.switch_page("pages/jintai.py")
if st.sidebar.button("AR スクワット"):
    st.switch_page("pages/sukutest.py")
if st.sidebar.button("Home"):
    pass

# titleの描画
st.title("Border Break Studies")

with st.spinner("必要なモジュールを読み込んでいます\nしばらくお待ちください"):
    import time as ti
    import os

try:
    with st.spinner('写真の読み込み中です\nしばらくお待ちください'):
        # 画像を読み込む

        current_dir = os.path.dirname(os.path.abspath(__file__))

        main = os.path.join(current_dir, 'Images', 'main.png')
        
    st.write("<b>道具がない、教室に通えない。そんな見えない'格差の境界線'を壊すARコンテンツ<b>", unsafe_allow_html=True)
    st.image(main)
    st.markdown("<b>コンテンツ一覧<b>", unsafe_allow_html=True)
    if st.button("AR そろばん"):
        st.switch_page("pages/soroban.py")
    if st.button("AR テルミン"):
        st.switch_page("pages/terumin.py")
    if st.button("AR パレット"):
        st.switch_page("pages/paretto.py")
    if st.button("AR 人体模型"):
        st.switch_page("pages/jintai.py")
    if st.button("AR スクワット"):
        st.switch_page("pages/sukutest.py")
except Exception as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
except RuntimeError as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")


