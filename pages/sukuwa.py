import streamlit as st

write = True

# titleの描画
st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=s'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("AR テルミン"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=t'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("AR パレット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=p'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("AR 人体模型"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=j'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("AR スクワット"):
    pass
if st.sidebar.button("Home"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=h'>", unsafe_allow_html=True)
        exit()

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
            st.write(f"<meta http-equiv='refresh' content='0;url=sukuwa'>", unsafe_allow_html=True)
    if st.button("ホームに戻る"):
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
            ti.sleep(1)
            st.write(f"<meta http-equiv='refresh' content='0;url=?page=h'>", unsafe_allow_html=True)

if write == True:
    # ページの描画
    st.title("ARスクワット")
    mokuhyo = st.number_input("目標回数を入力してください", 0)
    if st.button("スクワットへ"):
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
            ti.sleep(1)

            st.write(f"<meta http-equiv='refresh' content='0;url=sukuwatto?number={mokuhyo}'>", unsafe_allow_html=True)
