import streamlit as st

st.set_page_config(page_title="テルミン")
st.title("Border Break Studies")
st.title('AR テルミン')

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/soroban.py")
        exit()
if st.sidebar.button("AR テルミン"):
    pass
if st.sidebar.button("AR パレット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/paretto.py")
        exit()
if st.sidebar.button("AR 人体模型"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/jintai.py")
        exit()
if st.sidebar.button("AR スクワット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/sukuwa.py")
        exit()
if st.sidebar.button("Home"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("main.py")
        exit()

with st.spinner('読み込み中です\nしばらくお待ちください'):

    with st.spinner("モジュールの読み込み中です\nしばらくお待ちください"):
       import mediapipe as mp
       import time as ti
       import cv2
       import sounddevice as sd
       import numpy as np
       from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
       import mediapipe as mp
       import cv2
       import av
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

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


        return av.VideoFrame.from_ndarray(frame, format="bgr24")

ctx = webrtc_streamer(
    key="camera",
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandProcessor
)

st.markdown("<b>使い方<b>", unsafe_allow_html=True)
st.write("上に行けば行くほど音が小くなり")
st.write("下に行けば行くほど音が大きくなります。")
st.write("右に行けば行くほど音が高くなり")
st.write("左に行けば行くほど音が低くくなります。")
st.write("上記の方法で音を制御します。")

