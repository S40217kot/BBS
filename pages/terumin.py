import streamlit as st

st.set_page_config(page_title="テルミン")
st.title("Border Break Studies")
st.title("ARテルミン")

#st.title("現在このページはご利用いただけません\n申し訳ありません。")
#import time as ti
#ti.sleep(2)
#with st.spinner('リダイレクト中です\nしばらくお待ちください'):
#    st.switch_page("main.py")

if st.sidebar.button("AR そろばん"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
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
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/sukutest.py")
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
      import numpy as np
      import streamlit.components.v1 as components
      from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
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

        if st.button("音が出るようにする"):
            components.html("""
            <script>
            let ctx = new (window.AudioContext || window.webkitAudioContext)();
            let osc = ctx.createOscillator();
            let gainNode = ctx.createGain();
            osc.type = 'sine';
            osc.connect(gainNode);
            gainNode.connect(ctx.destination);
            gainNode.gain.value = 0;  // 最初は音量0
            osc.start();

            function setTone(freq, vol){
                ctx.resume();
                osc.frequency.setValueAtTime(freq, ctx.currentTime);
                gainNode.gain.setValueAtTime(vol, ctx.currentTime);
            }

            function stopTone(){
                gainNode.gain.setValueAtTime(0, ctx.currentTime);
            }
            </script>
            """, height=0, width=0)
    with st.spinner("関数の定義中です\nしばらくお待ちください"):
        # 音を鳴らす
        def generate_tone(freq, vol):
            components.html(f"<script>setTone({freq},{vol});</script>", height=0, width=0)

        def stop_tone():
            components.html("<script>stopTone();</script>", height=0, width=0)


        # +と-を逆転させる関数
        def reverse_sign(n):
            return -n
    
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

components.html(f"<script>setTone({100},{1});</script>", height=0, width=0)

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        try:
            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)

                    # 音量と周波数のマッピング
                    freq = max(100, hand_x * 5)
                    vol = min(1.0, max(0.0, 1 - hand_y / frame.shape[0]))
                    generate_tone(freq, vol)
            else:
                components.html("<script>stopTone();</script>", height=0, width=0)

            # --- UI用の描画 ---
            cv2.rectangle(frame, (10, 65), (20, 400), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
            cv2.rectangle(frame, (10, 400), (570, 440), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
            cv2.rectangle(frame, (570, 400), (600, 410), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
            cv2.rectangle(frame, (570, 430), (600, 440), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)
            cv2.rectangle(frame, (600, 410), (600, 440), (180, 180, 180), cv2.FILLED, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(frame, format="bgr24")
        except Exception as e:
            print("Error in recv:", e)
            return av.VideoFrame.from_ndarray(frame, format="bgr24")

ctx = webrtc_streamer(
    key="camera",
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandProcessor
)

if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("main.py")
        exit()
st.write("使い方")
st.text("...")



