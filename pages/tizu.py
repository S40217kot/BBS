import streamlit as st

st.title("Border Break Studies")
st.title('AR 地図')

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/terumin.py")
if st.sidebar.button("AR 地図"):
    pass
if st.sidebar.button("AR パレット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/paretto.py")
if st.sidebar.button("AR 人体模型"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/jintai.py")
if st.sidebar.button("AR スクワット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/sukuwa.py")
if st.sidebar.button("Home"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("main.py")

with st.spinner('読み込み中です\nしばらくお待ちください'):
    with st.spinner("モジュールのロード中です\nしばらくお待ちください"):
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
        import mediapipe as mp
        import cv2
        import av
        import os
        import math
        import numpy as np
        from PIL import ImageFont, ImageDraw, Image
    
    with st.spinner("画像の読み込み中です\nしばらくお待ちください"):
        # 透過PNG画像の読み込み
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_image_path = os.path.join(current_dir, '..', 'Images', 'tizu.png')
        target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
        target_size = 400  # 画像の直径
        try:
            target_image = cv2.resize(target_image, (target_size, target_size))
        except Exception as e:
            st.error(e)

        target_image_path_meron = os.path.join(current_dir, '..', 'Images', 'meron.png')
        target_image_meron = cv2.imread(target_image_path_meron, cv2.IMREAD_UNCHANGED)
        target_size_meron = 100  # 画像の直径
        try:
            target_image_meron = cv2.resize(target_image_meron, (target_size_meron, target_size_meron))
        except Exception as e:
            st.error(e)

        target_image_path_maturi = os.path.join(current_dir, '..', 'Images', 'maturi.png')
        target_image_maturi = cv2.imread(target_image_path_maturi, cv2.IMREAD_UNCHANGED)
        target_size_maturi = 200  # 画像の直径
        try:
            target_image_maturi = cv2.resize(target_image_maturi, (target_size_maturi, target_size_maturi))
        except Exception as e:
            st.error(e)

        target_image_path_edozusi = os.path.join(current_dir, '..', 'Images', 'edozusi.png')
        target_image_edozusi = cv2.imread(target_image_path_edozusi, cv2.IMREAD_UNCHANGED)
        target_size_edozusi = 200  # 画像の直径
        try:
            target_image_edozusi = cv2.resize(target_image_edozusi, (target_size_edozusi, target_size_edozusi))
        except Exception as e:
            st.error(e)

        target_image_path_huzi = os.path.join(current_dir, '..', 'Images', 'huzi.png')
        target_image_huzi = cv2.imread(target_image_path_huzi, cv2.IMREAD_UNCHANGED)
        target_size_huzi = 200  # 画像の直径
        try:
            target_image_huzi = cv2.resize(target_image_huzi, (target_size_huzi, target_size_huzi))
        except Exception as e:
            st.error(e)

        target_image_path_biwako = os.path.join(current_dir, '..', 'Images', 'biwako.png')
        target_image_biwako = cv2.imread(target_image_path_biwako, cv2.IMREAD_UNCHANGED)
        target_size_biwako = 200  # 画像の直径
        try:
            target_image_biwako = cv2.resize(target_image_biwako, (target_size_biwako, target_size_biwako))
        except Exception as e:
            st.error(e)

        target_image_path_genbaku = os.path.join(current_dir, '..', 'Images', 'genbaku.png')
        target_image_genbaku = cv2.imread(target_image_path_biwako, cv2.IMREAD_UNCHANGED)
        target_size_genbaku = 200  # 画像の直径
        try:
            target_image_genbaku = cv2.resize(target_image_genbaku, (target_size_genbaku, target_size_genbaku))
        except Exception as e:
            st.error(e)

        target_image_path_onsen = os.path.join(current_dir, '..', 'Images', 'onse.png')
        target_image_onsen = cv2.imread(target_image_path_onsen, cv2.IMREAD_UNCHANGED)
        target_size_onsen = 200  # 画像の直径
        try:
            target_image_onsen = cv2.resize(target_image_onsen, (target_size_onsen, target_size_onsen))
        except Exception as e:
            st.error(e)

        target_image_path_ramen = os.path.join(current_dir, '..', 'Images', 'ra-me.png')
        target_image_ramen = cv2.imread(target_image_path_ramen, cv2.IMREAD_UNCHANGED)
        target_size_ramen = 200  # 画像の直径
        try:
            target_image_ramen = cv2.resize(target_image_ramen, (target_size_ramen, target_size_ramen))
        except Exception as e:
            st.error(e)

        target_image_path_satou = os.path.join(current_dir, '..', 'Images', 'satou.png')
        target_image_satou = cv2.imread(target_image_path_satou, cv2.IMREAD_UNCHANGED)
        target_size_satou = 200  # 画像の直径
        try:
            target_image_satou = cv2.resize(target_image_satou, (target_size_satou, target_size_satou))
        except Exception as e:
            st.error(e)
    
    with st.spinner("定数の定義中です\nしばらくお待ちください"):
        def is_hand_touching_gazou(hand_x, hand_y, gazou_x, gazou_y, gazou_radius):
            distance = math.sqrt((hand_x - gazou_x) ** 2 + (hand_y - gazou_y) ** 2)
            return distance < gazou_radius
        
        def putText_japanese(img, text, point, size, color):
            try:
                font = ImageFont.truetype('BIZ-UDGothicR.ttc', size)
            except OSError:
                font = ImageFont.load_default()

            #imgをndarrayからPILに変換
            img_pil = Image.fromarray(img)

            #drawインスタンス生成
            draw = ImageDraw.Draw(img_pil)

            #テキスト描画
            draw.text(point, text, fill=color, font=font)

            #PILからndarrayに変換して返す
            return np.array(img_pil)

    with st.spinner("MediaPipeの初期化中です\nしばらくお待ちください"):
        # MediaPipe Handsの初期設定
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils

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
        try:
            frame = frame.to_ndarray(format="bgr24")
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.flip(image_rgb, 1)

            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)

                    if is_hand_touching_gazou(hand_x, hand_y, 500, 100, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"北海道地方:メロンが有名です", (0, 10), 25, (0, 0, 0))
                        if target_image_meron is not None:
                            h, w = target_image_meron.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 400 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_meron[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 400, 200, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"東北地方:青森県のねぶた祭が", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_maturi is not None:
                            h, w = target_image_maturi.shape[:2]
                            x_offset = max(0, 500 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_maturi[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 400, 300, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"関東地方:東京の江戸前寿司が", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_edozusi is not None:
                            h, w = target_image_edozusi.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_edozusi[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 350, 350, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"関東地方:山梨と静岡の富士山", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"が有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_huzi is not None:
                            h, w = target_image_huzi.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_huzi[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 300, 350, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"近畿地方:滋賀の琵琶湖が有名", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"です", (0, 50), 25, (0, 0, 0))
                        if target_image_biwako is not None:
                            h, w = target_image_biwako.shape[:2]
                            x_offset = max(0, 500 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_biwako[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 200, 350, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"中国地方:広島の原爆ドームが", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_genbaku is not None:
                            h, w = target_image_genbaku.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_genbaku[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 250, 350, 30):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"四国地方:愛知の道後温泉が有", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"名です", (0, 50), 25, (0, 0, 0))
                        if target_image_onsen is not None:
                            h, w = target_image_onsen.shape[:2]
                            x_offset = max(0, 500 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_onsen[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 150, 350, 50):
                        cv2.rectangle(frame, (0, 0), (350, 40),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"九州地方:福岡の博多ラーメンが", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_ramen is not None:
                            h, w = target_image_ramen.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_ramen[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)
                    if is_hand_touching_gazou(hand_x, hand_y, 100, 500, 50):
                        cv2.rectangle(frame, (0, 0), (700, 80),(255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"沖縄地方:特産品はゴーヤー", (0, 10), 25, (0, 0, 0))
                        cv2.rectangle(frame, (0, 40), (350, 80), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                        frame = putText_japanese(frame, f"有名です", (0, 50), 25, (0, 0, 0))
                        if target_image_satou is not None:
                            h, w = target_image_satou.shape[:2]
                            x_offset = max(0, 550 - w // 2)
                            y_offset = max(0, 350 - h // 2)

                            # 貼り付け可能な領域を制限
                            h_end = min(frame.shape[0], y_offset + h)
                            w_end = min(frame.shape[1], x_offset + w)

                            roi = frame[y_offset:h_end, x_offset:w_end]

                            overlay_resized = target_image_satou[0:(h_end-y_offset), 0:(w_end-x_offset)]

                            try:
                                if overlay_resized.shape[2] == 4:
                                    alpha = overlay_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_resized[:, :, c] * alpha
                                else:
                                    roi[:, :, :] = overlay_resized
                            except Exception as e:
                                st.error(e)


            if target_image is not None:
                h, w = target_image.shape[:2]
                x_offset = (frame.shape[1] - w) // 2
                y_offset = (frame.shape[0] - h) // 2
                if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                    if target_image.shape[2] == 4:
                        alpha_channel = target_image[:, :, 3] / 255.0
                        overlay_colors = target_image[:, :, :3]
                        for c in range(3):
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                                frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                                overlay_colors[:, :, c] * alpha_channel
                    else:
                        frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image



            return av.VideoFrame.from_ndarray(frame, format="bgr24")
        except Exception as e:
            st.error(e)
            return av.VideoFrame.from_ndarray(frame, format="bgr24")

ctx = webrtc_streamer(
    key="camera",
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=HandProcessor
)

st.markdown("<b>使い方<b>", unsafe_allow_html=True)
