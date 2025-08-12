import streamlit as st

st.title("Border Break Studies")

st.set_page_config(page_title="パレット")

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
        st.switch_page("pages/sukuwa.py")
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
        # ボタンを押したか判別する関数
        def is_hand_touching_button(hand_x, hand_y, circle_x, circle_y, circle_radius):
            distance = math.sqrt((hand_x - circle_x) ** 2 + (hand_y - circle_y) ** 2)
            return distance < circle_radius
        
        # ボタンが押された際に色を変えるプログラム
        def function_button(cllor):
            global pen_clor

            pen_clor = cllor

            return pen_clor
    with st.spinner("変数の定義中です\nしばらくお待ちください"):
        # 座標指定（リストに変更）
        kaku_points_red = []
        kaku_points_blue = []
        kaku_points_green = []
        kaku_points_yellow = []

        # 書き始めているかを確認
        kaku = False

        # ペンの色を定義
        pen_clor = "red"

        # ボタンの位置（x, y, 幅, 高さ）
        # green
        button_rect_green = (100, 450, 20)  # (x, y, size)
        # blue
        button_rect_blue = (200, 450, 20)  # (x, y, size)
        # red
        button_rect_red = (300, 450, 20)  # (x, y, size)
        # yellow
        button_rect_yellow = (400, 450, 20)  # (x, y, size)
        # ボタンが押されたかどうかを追跡
        last_pressed_time = ti.time()

        # 手が検出されたかを調べる関数
        hand_touching = False

# ページの配置
st.title('AR パレット')

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

        # 手が検出されたかを調べる関数
        hand_touching = False

        # 画像の反転
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.flip(image_rgb, 1)

        # MediaPipeで手を検出
        results = self.hands.process(image_rgb)

        # ランドマークを元のBGR画像に描画（OpenCVの画像はBGR形式）
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                # landmarkが取得されていることを確認
                hand_touching = True

                # ランドマークの座標を取得
                hand_x_hito = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                hand_y_hito = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                hand_x_oya = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                hand_y_oya = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

                # 親指と人差し指の距離を計算
                distance = math.sqrt((hand_x_hito - hand_x_oya) ** 2 + (hand_y_hito - hand_y_oya) ** 2)

                # 距離が一定以上ある場合、描画を開始
                if distance > 30 and kaku == False:
                    # 書き始めている場合、kakuをTrueにする
                    kaku = True

                # 書き始めている場合、座標を追加
                if kaku == True:
                    # kaku_points に座標を追加する処理
                    if pen_clor == "red":
                        kaku_points_red.append((hand_x_hito, hand_y_hito))
                    elif pen_clor == "blue":
                        kaku_points_blue.append((hand_x_hito, hand_y_hito))
                    elif pen_clor == "green":
                        kaku_points_green.append((hand_x_hito, hand_y_hito))
                    elif pen_clor == "yellow":
                        kaku_points_yellow.append((hand_x_hito, hand_y_hito))
                    else:
                        Exception("無効な色が選択されました")
                        break

                # 距離が一定以下になった場合、描画を終了
                if distance <= 30 and kaku == True:
                    kaku = False

                # ボタンが押されたら色を変える
                # green
                if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_green[0], button_rect_green[1], button_rect_green[2]):
                    if ti.time() - last_pressed_time > 3:
                        function_button("green")
                        last_pressed_time = ti.time()
                # blue
                if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_blue[0], button_rect_blue[1], button_rect_blue[2]):
                    if ti.time() - last_pressed_time > 3:
                        function_button("blue")
                        last_pressed_time = ti.time()
                # red
                if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_red[0], button_rect_red[1], button_rect_red[2]):
                    if ti.time() - last_pressed_time > 3:
                        function_button("red")
                        last_pressed_time = ti.time()
                # yellow
                if is_hand_touching_button(hand_x_hito, hand_y_hito, button_rect_yellow[0], button_rect_yellow[1], button_rect_yellow[2]):
                    if ti.time() - last_pressed_time > 3:
                        function_button("yellow")
                        last_pressed_time = ti.time()

        # 描画された軌跡を表示
        # red
        if len(kaku_points_red) > 1:
            for i in range(1, len(kaku_points_red)):
                cv2.line(frame, kaku_points_red[i-1], kaku_points_red[i], (0, 0, 255), 3)
        # blue
        if len(kaku_points_blue) > 1:
            for i in range(1, len(kaku_points_blue)):
                cv2.line(frame, kaku_points_blue[i-1], kaku_points_blue[i], (255, 0, 0), 3)
        # green
        if len(kaku_points_green) > 1:
            for i in range(1, len(kaku_points_green)):
                cv2.line(frame, kaku_points_green[i-1], kaku_points_green[i], (0, 255, 0), 3)
        # yellow
        if len(kaku_points_yellow) > 1:
            for i in range(1, len(kaku_points_yellow)):
                cv2.line(frame, kaku_points_yellow[i-1], kaku_points_yellow[i], (0, 255, 255), 3)
        
        # 色変えボタンの背景を描画
        cv2.rectangle(frame, (0, 400), (800, 500), (255, 255, 255), -1)

        # ボタン描画
        # green
        cv2.circle(frame, (button_rect_green[0], button_rect_green[1]),
                    button_rect_green[2],
                    (0, 255, 0), -1)
        # blue
        cv2.circle(frame, (button_rect_blue[0], button_rect_blue[1]),
                   button_rect_blue[2],
                   (255, 0, 0), -1)
        # red
        cv2.circle(frame, (button_rect_red[0], button_rect_red[1]),
                   button_rect_red[2],
                   (0, 0, 255), -1)
        # yellow
        cv2.circle(frame, (button_rect_yellow[0], button_rect_yellow[1]),
                   button_rect_yellow[2],
                   (0, 255, 255), -1)

        # 指の軌道を描画
        if hand_touching == True:
            cv2.circle(frame, (hand_x_hito, hand_y_hito), 5, (0, 255, 0), -1)
            cv2.circle(frame, (hand_x_oya, hand_y_oya), 5, (0, 255, 0), -1)

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

