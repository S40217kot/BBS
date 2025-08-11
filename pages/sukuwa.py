import streamlit as st

write = True

# titleの描画
st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/soroban.py")
if st.sidebar.button("AR テルミン"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/terumin.py")
if st.sidebar.button("AR パレット"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/paretto.py")
if st.sidebar.button("AR 人体模型"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/jintai.py")
if st.sidebar.button("AR スクワット"):
    pass
if st.sidebar.button("Home"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("main.py")

with st.spinner("必要なモジュールを読み込んでいます\nしばらくお待ちください"):
    import time as ti

# ページの移動がある場合
# urlを取得
url = st.query_params
out = url.get('situation', [None])[0]
if out == "f":
    mokuhyo = url.get('mokuhyo', [None])
    write = False
    st.write(f'お疲れ様です。\n{mokuhyo}回達成できました。\n次回からも頑張ってください。')
    if st.button("もう一度する"):
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
            ti.sleep(1)
            st.switch_page("pages/sukuwa.py")
    if st.button("ホームに戻る"):
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
            ti.sleep(1)
            st.switch_page("main.py")

if write == True:
    # ページの描画
    st.title("ARスクワット")
    mokuhyo = st.number_input("目標回数を入力してください", 0)
    if st.button("スクワットへ"):
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
            ti.sleep(1)
            st.session_state["mokuhyo"] = f"{mokuhyo}"
            st.switch_page("pages/sukuwatto.py")

