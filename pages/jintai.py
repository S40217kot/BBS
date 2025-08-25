import streamlit as st

st.title("Border Break Studies")
st.set_page_config(page_title="人体模型")
st.title('AR 人体模型')

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/soroban.py")
        exit()
if st.sidebar.button("AR 地図"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
        ti.sleep(1)
        st.switch_page("pages/tizu.py")
        exit()
if st.sidebar.button("AR パレット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/paretto.py")
        exit()
if st.sidebar.button("AR 人体模型"):
    pass
if st.sidebar.button("AR スクワット"):
    import time as ti
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
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
       from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
       import av
       import mediapipe as mp
       import time as ti
       import cv2
       import os
       import math
       import numpy as np
       from PIL import ImageFont, ImageDraw, Image

    with st.spinner("画像の読み込み中です\nしばらくお待ちください"):
        
        # 透過PNG画像の読み込み
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_image_path = os.path.join(current_dir, '..', 'Images', 'nou.png')
        target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
        target_size = 40  # 画像の直径
        target_image = cv2.resize(target_image, (target_size, target_size))

        # 透過PNG画像の読み込み
        target_image_path_hai = os.path.join(current_dir, '..', 'Images','hai.png')
        target_image_hai = cv2.imread(target_image_path_hai, cv2.IMREAD_UNCHANGED)
        target_size_hai = 80  # 画像の直径
        target_image_hai = cv2.resize(target_image_hai, (target_size_hai, target_size_hai))

        # 透過PNG画像の読み込み
        target_image_path_i = os.path.join(current_dir, '..', 'Images','i.png')
        target_image_i = cv2.imread(target_image_path_i, cv2.IMREAD_UNCHANGED)
        target_size_i = 40  # 画像の直径
        target_image_i = cv2.resize(target_image_i, (target_size_i, target_size_i))

    with st.spinner("MediaPipePoseの初期化中です\nしばらくお待ちください"):
        # MediaPipe poseのセットアップ
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def is_hand_touching_zouki(hand_x, hand_y, zouki_x, zouki_y, zouki_radius):
        distance = math.sqrt((hand_x - zouki_x) ** 2 + (hand_y - zouki_y) ** 2)
        return distance < zouki_radius
    
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

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_pose = mp.solutions.pose

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        try:

            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(image_rgb)

             # 検出結果がある場合、ランドマークを描画
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                twenty_one = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
                twenty_two = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
                twenty_one_x = int(twenty_one.x * frame.shape[1])
                twenty_one_y = int(twenty_one.y * frame.shape[0])


                # MediaPipeでlandmarksを描画
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if target_image is not None and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                # one = 左目の内側（1番）
                one = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER]
                # four = 右目の内側（4番）
                four = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER]

                # フレームサイズに合わせてスケーリング
                one_x = int(one.x * frame.shape[1])
                one_y = int(one.y * frame.shape[0])
                four_x = int(four.x * frame.shape[1])
                four_y = int(four.y * frame.shape[0])

                # 中心座標（画像の中心をここに合わせる）
                # 座標保存用にnou_x,nou_yを作成
                center_x = nou_x = (one_x + four_x) // 2
                center_y = nou_y = (one_y + four_y - 100) // 2

                # 目と目の距離を使って拡大縮小の倍率を決定
                dx = one_x - four_x
                dy = one_y - four_y
                eye_distance = int((dx ** 2 + dy ** 2) ** 0.5)

                # スケール倍率（10pxのとき1倍）
                scale = eye_distance / 10.0
                scale = max(0.3, min(scale, 3.0))  # サイズ制限（任意）

                # target_image をスケールに応じてリサイズ
                h, w = target_image.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_image = cv2.resize(target_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 貼り付け開始座標（画像の中心をcenterに合わせる）
                x_offset = center_x - new_w // 2
                y_offset = center_y - new_h // 2

                # フレーム範囲に合わせて表示範囲を制限
                x1 = max(x_offset, 0)
                y1 = max(y_offset, 0)
                x2 = min(x_offset + new_w, frame.shape[1])
                y2 = min(y_offset + new_h, frame.shape[0])

                if x2 > x1 and y2 > y1:
                    # リサイズ画像の切り取り範囲を算出
                    crop = resized_image[y1 - y_offset:y2 - y_offset,
                                         x1 - x_offset:x2 - x_offset]

                    # アルファチャンネルありの場合（透過画像対応）
                    if crop.shape[2] == 4:
                        alpha = crop[:, :, 3] / 255.0
                        overlay = crop[:, :, :3]
                        for c in range(3):
                            frame[y1:y2, x1:x2, c] = \
                                frame[y1:y2, x1:x2, c] * (1 - alpha) + overlay[:, :, c] * alpha
                    else:
                        frame[y1:y2, x1:x2] = crop

                if target_image is not None and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                # eleven = 左の肩
                eleven = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                # four = 右の肩
                twelve = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # フレームサイズに合わせてスケーリング
                eleven_x = int(eleven.x * frame.shape[1])
                eleven_y = int(eleven.y * frame.shape[0])
                twelve_x = int(twelve.x * frame.shape[1])
                twelve_y = int(twelve.y * frame.shape[0])

                # 中心座標（画像の中心をここに合わせる）
                # 座標保存用にhai_x,hai_yを作成
                center_x_hai = hai_x = (eleven_x + twelve_x) // 2
                center_y_hai = hai_y = (eleven_y + twelve_y + 100) // 2

                # 距離を使って拡大縮小の倍率を決定
                dx_hai = eleven_x - twelve_x
                dy_hai = eleven_y - twelve_y
                eye_distance_hai = int((dx_hai ** 2 + dy_hai ** 2) ** 0.5)

                # スケール倍率（10pxのとき1倍）
                scale_hai = eye_distance_hai / 10.0
                scale_hai = max(0.3, min(scale_hai, 3.0))  # サイズ制限（任意）

                # target_image_hai をスケールに応じてリサイズ
                h, w = target_image_hai.shape[:2]
                new_w_hai = int(w * scale_hai)
                new_h_hai = int(h * scale_hai)
                resized_image_hai = cv2.resize(target_image_hai, (new_w_hai, new_h_hai), interpolation=cv2.INTER_AREA)

                # 貼り付け開始座標（画像の中心をcenterに合わせる）
                x_offset_hai = center_x_hai - new_w_hai // 2
                y_offset_hai = center_y_hai- new_h_hai // 2

                # フレーム範囲に合わせて表示範囲を制限
                x1_hai = max(x_offset_hai, 0)
                y1_hai = max(y_offset_hai, 0)
                x2_hai = min(x_offset_hai + new_w_hai, frame.shape[1])
                y2_hai = min(y_offset_hai + new_h_hai, frame.shape[0])

                if x2 > x1 and y2 > y1:
                    # リサイズ画像の切り取り範囲を算出
                    crop = resized_image[y1 - y_offset:y2 - y_offset,
                                         x1 - x_offset:x2 - x_offset]

                if x2_hai > x1_hai and y2_hai > y1_hai:

                    # リサイズ画像の切り取り範囲を算出
                    crop = resized_image_hai[y1_hai - y_offset_hai:y2_hai - y_offset_hai,
                                         x1_hai - x_offset_hai:x2_hai - x_offset_hai]
                    
                    # アルファチャンネルありの場合（透過画像対応）
                    if crop.shape[2] == 4:
                        alpha = crop[:, :, 3] / 255.0
                        overlay = crop[:, :, :3]
                        for c in range(3):
                            frame[y1_hai:y2_hai, x1_hai:x2_hai, c] = \
                                frame[y1_hai:y2_hai, x1_hai:x2_hai, c] * (1 - alpha) + overlay[:, :, c] * alpha
                    else:
                        frame[y1_hai:y2_hai, x1_hai:x2_hai] = crop


                if target_image is not None and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # eleven = 左の肩
                    eleven = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    # twenty_four = 右の腰
                    twenty_four = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                    # フレームサイズに合わせてスケーリング
                    eleven_x = int(eleven.x * frame.shape[1])
                    eleven_y = int(eleven.y * frame.shape[0])
                    twenty_four_x = int(twenty_four.x * frame.shape[1])
                    twenty_four_y = int(twenty_four.y * frame.shape[0])

                    # 中心座標（画像の中心をここに合わせる）
                    # 座標保存用にi_x,i_yを作成
                    center_x_i = i_x = (eleven_x + twenty_four_x - 50) // 2
                    center_y_i = i_y = ((eleven_y + twenty_four_y - 100) // 2) + 50

                    # 目と目の距離を使って拡大縮小の倍率を決定
                    dx_i = eleven_x - twenty_four_x
                    dy_i = eleven_y - twenty_four_y
                    eye_distance_i = int((dx_i ** 2 + dy_i ** 2) ** 0.5)

                    # スケール倍率（2.5pxのとき1倍）
                    scale_i = eye_distance_i / 2.5
                    scale_i = max(0.3, min(scale_i, 3.0))  # サイズ制限（任意）

                    # target_image をスケールに応じてリサイズ
                    h_i, w_i = target_image_i.shape[:2]
                    new_w_i = int(w_i * scale_i)
                    new_h_i = int(h_i * scale_i)
                    resized_image_i = cv2.resize(target_image_i, (new_w_i, new_h_i), interpolation=cv2.INTER_AREA)

                    # 貼り付け開始座標（画像の中心をcenterに合わせる）
                    x_offset_i = center_x_i - new_w_i // 2
                    y_offset_i = center_y_i - new_h_i // 2

                    # フレーム範囲に合わせて表示範囲を制限
                    x1_i = max(x_offset_i, 0)
                    y1_i = max(y_offset_i, 0)
                    x2_i = min(x_offset_i + new_w_i, frame.shape[1])
                    y2_i = min(y_offset_i + new_h_i, frame.shape[0])

                    if x2_i > x1_i and y2_i > y1_i:
                        # リサイズ画像の切り取り範囲を算出
                        crop = resized_image_i[y1_i - y_offset_i:y2_i - y_offset_i,
                                             x1_i - x_offset_i:x2_i - x_offset_i]

                        # アルファチャンネルありの場合（透過画像対応）
                        if crop.shape[2] == 4:
                            alpha = crop[:, :, 3] / 255.0
                            overlay = crop[:, :, :3]
                            for c in range(3):
                                frame[y1_i:y2_i, x1_i:x2_i, c] = \
                                    frame[y1_i:y2_i, x1_i:x2_i, c] * (1 - alpha) + overlay[:, :, c] * alpha
                        else:
                            frame[y1_i:y2_i, x1_i:x2_i] = crop

                
                if is_hand_touching_zouki(twenty_one_x, twenty_one_y, nou_x, nou_y, 100):
                    cv2.rectangle(frame, (0, 0), (700, 80), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
                    frame = putText_japanese(frame, f"脳 : 頭蓋骨の中にあり、思考や行動、記憶、感情などを\nつかさどる臓器です。", (0, 10), 25, (255, 255, 255))
                if is_hand_touching_zouki(twenty_one_x, twenty_one_y, hai_x, hai_y, 100):
                    cv2.rectangle(frame, (0, 0), (700, 80), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
                    frame = putText_japanese(frame, f"肺 : 肋骨の中にあり呼吸によって空気中の酸素を体内に\n取り込み、二酸化炭素を排出する役割の臓器です。", (0, 10), 25, (255, 255, 255))
                if is_hand_touching_zouki(twenty_one_x, twenty_one_y, i_x, i_y, 100):
                    cv2.rectangle(frame, (0, 0), (700, 80), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
                    frame = putText_japanese(frame, f"胃 : 食道の中にあり食べたものを一時的に貯蔵し、消化\nを助ける役割の臓器です。", (0, 10), 25, (255, 255, 255))
            return av.VideoFrame.from_ndarray(frame, format="bgr24")
        except Exception as e:
            import traceback
            st.error(e)
            traceback.print_exc()
            return frame
        
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

