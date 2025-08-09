import streamlit as st

st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=s'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("AR テルミン"):
    pass
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
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=c'>", unsafe_allow_html=True)
        exit()
if st.sidebar.button("Home"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=/?page=h'>", unsafe_allow_html=True)
        exit()

with st.spinner('読み込み中です\nしばらくお待ちください'):

    with st.spinner("モジュールの読み込み中です\nしばらくお待ちください"):
       import mediapipe as mp
       import time as ti
       import cv2
       import sounddevice as sd
       import numpy as np

    with st.spinner("カメラ映像の取得中です\nしばらくお待ちください"):
        # カメラを起動（0番はデフォルトカメラ）
        cap = cv2.VideoCapture(0)

    with st.spinner("MediapipeHandsの初期化中です\nしばらくお待ちください"):
        # MediaPipe Handsの初期設定
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils

    with st.spinner("関数の定義中です\nしばらくお待ちください"):
        # 音を鳴らす
        # 音声信号生成関数
        def generate_tone(frequency, volume=0.5, sample_rate=44100, duration=0.1):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return volume * np.sin(2 * np.pi * frequency * t)

        # +と-を逆転させる関数
        def reverse_sign(n):
            return -n

# ページの配置
st.title('AR テルミン')
placeholder = st.empty()
if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
         ti.sleep(1)
         st.write(f"<meta http-equiv='refresh' content='0;url=/'>", unsafe_allow_html=True)
st.markdown("<b>使い方<b>", unsafe_allow_html=True)
st.write("上に行けば行くほど音が小くなり")
st.write("下に行けば行くほど音が大きくなります。")
st.write("右に行けば行くほど音が高くなり")
st.write("左に行けば行くほど音が低くくなります。")
st.write("上記の方法で音を制御します。")

# メインループ
try:
    while True:
        ret, frame = cap.read()  # カメラから1フレーム取得
        if not ret:
            break

        # OpenCVはBGR形式なので、まずRGBに変換
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 画像を水平方向に反転
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.flip(image_rgb, 1)

        # MediaPipeで手を検出
        results = hands.process(image_rgb)  # ★ここでRGB画像を渡す

        # ランドマークを元のBGR画像に描画（OpenCVの画像はBGR形式）
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS 
                )

                hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                
                # 現在の指先位置を表示
                cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)

                # 音を流す
                freq = int(hand_x * 10)
                bolm = int(hand_y / 150)
                tone = generate_tone(frequency=freq, duration=0.2, volume=bolm)
                sd.play(tone, samplerate=44100)
                sd.wait()

        cv2.rectangle(frame, (10, 65), (20, 400), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 400), (570, 440), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(frame, (570, 400), (600, 410), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(frame, (570, 430), (600, 440), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(frame, (600, 410), (600, 440), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)

        # 画像をRGBに変換してStreamlitで表示（StreamlitはRGB形式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB")  # ★ランドマーク描画後の画像を表示

except Exception as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
    import traceback as tr
    tr.print_exc()
except RuntimeError as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
    import traceback as tr
    tr.print_exc()
finally:
    # リソースの解放
    cap.release()