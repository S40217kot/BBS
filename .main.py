import streamlit as st

# titleの描画
st.title("Border Break Studies")

with st.spinner("必要なモジュールを読み込んでいます\nしばらくお待ちください"):
    import time as ti
    import os

# ページを描画(run)させるかを決める
# 別のurlに飛ぶときはrunさせない
run = "run"

# ページの移動がある場合
# urlを取得
url = st.query_params
out = url.get('page', [None])[0]
# ページに飛ばす
if out != None:
    run = "not"
    if out == "h":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url=/'>", unsafe_allow_html=True)
        pass
    if out == "s":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url={'.soroban'}'>", unsafe_allow_html=True)
        pass
    if out == "t":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url={'.terumin'}'>", unsafe_allow_html=True)
        pass
    if out == "p":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url={'.paretto'}'>", unsafe_allow_html=True)
        pass
    if out == "j":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url={'.jintai'}'>", unsafe_allow_html=True)
        pass
    if out == "c":
        with st.spinner('リダイレクト中です\nしばらくお待ちください'):
             ti.sleep(1)
             st.write(f"<meta http-equiv='refresh' content='0;url={'.sukuwa'}'>", unsafe_allow_html=True)
        pass

# runさせる場合だけページを描画する
if run == "run":
    try:
        with st.spinner('写真の読み込み中です\nしばらくお待ちください'):
            # 画像を読み込む

            current_dir = os.path.dirname(os.path.abspath(__file__))

            main = os.path.join(current_dir, 'Images', 'main.png')
            soroba = os.path.join(current_dir, 'Images', 'soroba.png')
            terumi = os.path.join(current_dir, 'Images', 'terumi.png')
            oeka = os.path.join(current_dir, 'Images', 'oeka.png')
            jinta = os.path.join(current_dir, 'Images', 'jinta.png')
            sukuwa = os.path.join(current_dir, 'Images', 'sukuwa.png')
        if st.sidebar.button("AR そろばん"):
            with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                ti.sleep(1)
                st.write(f"<meta http-equiv='refresh' content='0;url=?page=s'>", unsafe_allow_html=True)
                exit()
        if st.sidebar.button("AR テルミン"):
            with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                ti.sleep(1)
                st.write(f"<meta http-equiv='refresh' content='0;url=?page=t'>", unsafe_allow_html=True)
                exit()
        if st.sidebar.button("AR パレット"):
            with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                ti.sleep(1)
                st.write(f"<meta http-equiv='refresh' content='0;url=?page=p'>", unsafe_allow_html=True)
                exit()
        if st.sidebar.button("AR 人体模型"):
            with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                ti.sleep(1)
                st.write(f"<meta http-equiv='refresh' content='0;url=?page=j'>", unsafe_allow_html=True)
                exit()
        if st.sidebar.button("AR スクワット"):
            with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                ti.sleep(1)
                st.write(f"<meta http-equiv='refresh' content='0;url=?page=c'>", unsafe_allow_html=True)
                exit()
        if st.sidebar.button("Home"):
            pass
        st.write("<b>道具がない、教室に通えない。そんな見えない'格差の境界線'を壊すARコンテンツ<b>", unsafe_allow_html=True)
        st.image(main)
        st.markdown("<b>コンテンツ一覧<b>", unsafe_allow_html=True)
        if st.button("AR そろばん"):
            st.write(f"<meta http-equiv='refresh' content='0;url={'?page=s'}'>", unsafe_allow_html=True)
        st.image(soroba)
        if st.button("AR テルミン"):
            st.write(f"<meta http-equiv='refresh' content='0;url={'?page=t'}'>", unsafe_allow_html=True)
        st.image(terumi)
        if st.button("AR パレット"):
            st.write(f"<meta http-equiv='refresh' content='0;url={'?page=p'}'>", unsafe_allow_html=True)
        st.image(oeka)
        if st.button("AR 人体模型"):
            st.write(f"<meta http-equiv='refresh' content='0;url={'?page=j'}'>", unsafe_allow_html=True)
        st.image(jinta)
        if st.button("AR スクワット"):
            st.write(f"<meta http-equiv='refresh' content='0;url={'?page=c'}'>", unsafe_allow_html=True)
        st.image(sukuwa)

    except Exception as e:
        st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
    except RuntimeError as e:
        st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")