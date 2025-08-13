import streamlit as st

st.set_page_config(page_title="パレット")
st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/soroban.py")
if st.sidebar.button("AR テルミン"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/terumin.py")
if st.sidebar.button("AR パレット"):
    pass
if st.sidebar.button("AR 人体模型"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/jintai.py")
if st.sidebar.button("AR スクワット"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/sukutest.py")
if st.sidebar.button("Home"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("main.py")

with st.spinner('読み込み中です\nしばらくお待ちください'):

    with st.spinner("モジュールの読み込み中です\nしばらくお待ちください"):
        import mediapipe as mp
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
        import time as ti
        import cv2
        import av
        import math

    with st.spinner("MediaPipeHandsの初期化中です\nしばらくお待ちください"):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

    with st.spinner("関数の定義中です\nしばらくお待ちください"):
        def is_hand_touching_button(hand_x, hand_y, circle_x, circle_y, circle_radius):
            distance = math.sqrt((hand_x - circle_x) ** 2 + (hand_y - circle_y) ** 2)
            return distance < circle_radius

        def function_button(cllor):
            global pen_clor
            pen_clor = cllor
            return pen_clor

    with st.spinner("リングバッファクラス定義中です\nしばらくお待ちください"):
        class RingBuffer:
            def __init__(self, size):
                self.size = size
                self.data = [None] * size
                self.index = 0
                self.count = 0

            def append(self, point):
                self.data[self.index] = point
                self.index = (self.index + 1) % self.size
                if self.count < self.size:
                    self.count += 1

            def get_points(self):
                if self.count < self.size:
                    return self.data[:self.count]
                return self.data[self.index:] + self.data[:self.index]

    with st.spinner("変数の定義中です\nしばらくお待ちください"):
        MAX_POINTS = 500

        kaku_points_red = RingBuffer(MAX_POINTS)
        kaku_points_blue = RingBuffer(MAX_POINTS)
        kaku_points_green = RingBuffer(MAX_POINTS)
        kaku_points_yellow = RingBuffer(MAX_POINTS)

        kaku = False
        pen_clor = "red"

        button_rect_green = (100, 450, 20)
        button_rect_blue = (200, 450, 20)
        button_rect_red = (300, 450, 20)
        button_rect_yellow = (400, 450, 20)

        last_pressed_time = ti.time()

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# MediaPipe Handsはグローバルで1つだけ生成
mp_hands_instance = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        global kaku, pen_clor, last_pressed_time
        self.hands = mp_hands_instance

    def recv(self, frame):
        global kaku, pen_clor, last_pressed_time
        try:
            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.flip(frame, 1)  # 反転はここだけに
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(image_rgb)

            hand_touching = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_touching = True

                    hand_x_hito = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    hand_y_hito = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    hand_x_oya = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                    hand_y_oya = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

                    distance = math.sqrt((hand_x_hito - hand_x_oya) ** 2 + (hand_y_hito - hand_y_oya) ** 2)

                    if distance > 30 and not kaku:
                        kaku = True

                    if kaku:
                        if pen_clor == "red":
                            kaku_points_red.append((hand_x_hito, hand_y_hito))
                        elif pen_clor == "blue":
                            kaku_points_blue.append((hand_x_hito, hand_y_hito))
                        elif pen_clor == "green":
                            kaku_points_green.append((hand_x_hito, hand_y_hito))
                        elif pen_clor == "yellow":
                            kaku_points_yellow.append((hand_x_hito, hand_y_hito))

                    if distance <= 30 and kaku:
                        kaku = False

                    # ボタン処理
                    if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_green[0], button_rect_green[1], button_rect_green[2]):
                        if ti.time() - last_pressed_time > 3:
                            function_button("green")
                            last_pressed_time = ti.time()
                    if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_blue[0], button_rect_blue[1], button_rect_blue[2]):
                        if ti.time() - last_pressed_time > 3:
                            function_button("blue")
                            last_pressed_time = ti.time()
                    if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_red[0], button_rect_red[1], button_rect_red[2]):
                        if ti.time() - last_pressed_time > 3:
                            function_button("red")
                            last_pressed_time = ti.time()
                    if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_yellow[0], button_rect_yellow[1], button_rect_yellow[2]):
                        if ti.time() - last_pressed_time > 3:
                            function_button("yellow")
                            last_pressed_time = ti.time()

            # 軌跡描画（線は太さ2にして少し軽く）
            for points, color in [
                (kaku_points_red.get_points(), (0, 0, 255)),
                (kaku_points_blue.get_points(), (255, 0, 0)),
                (kaku_points_green.get_points(), (0, 255, 0)),
                (kaku_points_yellow.get_points(), (0, 255, 255)),
            ]:
                if len(points) > 1:
                    for i in range(1, len(points)):
                        if points[i-1] is not None and points[i] is not None:
                            cv2.line(frame, points[i-1], points[i], color, 2)

            # ボタンの背景と描画
            cv2.rectangle(frame, (0, 400), (800, 500), (255, 255, 255), -1)
            cv2.circle(frame, (button_rect_green[0], button_rect_green[1]), button_rect_green[2], (0, 255, 0), -1)
            cv2.circle(frame, (button_rect_blue[0], button_rect_blue[1]), button_rect_blue[2], (255, 0, 0), -1)
            cv2.circle(frame, (button_rect_red[0], button_rect_red[1]), button_rect_red[2], (0, 0, 255), -1)
            cv2.circle(frame, (button_rect_yellow[0], button_rect_yellow[1]), button_rect_yellow[2], (0, 255, 255), -1)

            # 指の軌道を描画
            if hand_touching:
                cv2.circle(frame, (hand_x_hito, hand_y_hito), 5, (0, 255, 0), -1)
                cv2.circle(frame, (hand_x_oya, hand_y_oya), 5, (0, 255, 0), -1)

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
        st.switch_page("main.py")

st.write("使い方")
st.text("...")

