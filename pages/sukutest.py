import streamlit as st

st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/soroban.py")
        exit()
if st.sidebar.button("AR テルミン"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/terumin.py")
        exit()
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
    pass
if st.sidebar.button("Home"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("main.py")
        exit()

with st.spinner('読み込み中です\nしばらくお待ちください'):

    with st.spinner("モジュールの読み込み中です\nしばらくお待ちください"):
       from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
       import mediapipe as mp
       import time as ti
       import cv2
       import math
       import av
       import numpy as np
       from PIL import ImageFont, ImageDraw, Image

    with st.spinner("MediaPipiPoseの初期化中です\nしばらくお待ちください"):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
    
    with st.spinner("関数の定義中です\nしばらくお待ちください"):
        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
        
            radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
            angle = abs(radians * 180.0 / math.pi)
            if angle > 180.0:
                angle = 360 - angle

            return angle

        def putText_japanese(img, text, point, size, color):
            font = ImageFont.truetype('BIZ-UDGothicR.ttc', size)

            #imgをndarrayからPILに変換
            img_pil = Image.fromarray(img)

            #drawインスタンス生成
            draw = ImageDraw.Draw(img_pil)

            #テキスト描画
            draw.text(point, text, fill=color, font=font)

            #PILからndarrayに変換して返す
            return np.array(img_pil)

    with st.spinner("変数の定義中です\nしばらくお待ちください"):
        stage = None  # "up" or "down"
        squat_counter = 0

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_pose = mp.solutions.pose
mp_hands_instance = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        
        self.pose = mp_hands_instance

    def recv(self, frame):
        global stage, squat_counter
        try:
            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(image_rgb)

            try:
                landmarks = results.pose_landmarks.landmark
                # ランドマークを取得（左足）
                hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee_lm = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle_lm = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                # 可視性が低ければスキップ（映っていないと判断）
                if hip_lm.visibility < 0.5 or knee_lm.visibility < 0.5 or ankle_lm.visibility < 0.5:
                    frame = putText_japanese(frame, "膝が検出されませんでした", (0, 50), 25, (0, 0, 255))
                    pass  # カウントや角度計算を行わない
                else:
                    # 角度を計算
                    hip = [hip_lm.x, hip_lm.y]
                    knee = [knee_lm.x, knee_lm.y]
                    ankle = [ankle_lm.x, ankle_lm.y]
                    angle = calculate_angle(hip, knee, ankle)
                    # 角度表示
                    frame = putText_japanese(frame, f"膝の角度を80度以下にしてください。", (0, 50), 25, (255, 255, 255))
                    frame = putText_japanese(frame, f"膝の角度 : {int(angle)}°", (0, 80), 25, (255, 255, 255))
                    # スクワット判定
                    if angle < 80:
                        stage = "down"
                    if angle > 100 and stage == "down":
                        stage = "up"
                        squat_counter += 1
            except:
                pass
            # 骨格を描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            # 回数表示
            frame = putText_japanese(frame, f'回数 : {squat_counter}', (0, 0), 50, (0, 255, 0))

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

# ページの配置
st.title('AR スクワット')
if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("main.py")
        exit()
st.write("使い方")

st.text("...")
