import streamlit as st

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
    pass
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
       import os

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

    with st.spinner("カメラ映像の取得中です\nしばらくお待ちください"):
        # カメラを起動（0番はデフォルトカメラ）
        cap = cv2.VideoCapture(0)

    with st.spinner("MediaPipePoseの初期化中です\nしばらくお待ちください"):
        # MediaPipe poseのセットアップ
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ページの配置
st.title('AR 人体模型')
placeholder = st.empty()
if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
         ti.sleep(1)
         st.write(f"<meta http-equiv='refresh' content='0;url=/'>", unsafe_allow_html=True)
st.write("使い方")
st.text("...")

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

        # ランドマークの検出
        results = pose.process(frame)

         # 検出結果がある場合、ランドマークを描画
        if results.pose_landmarks:
            for idx, landmarks in enumerate(results.pose_landmarks.landmark):

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
                center_x = (one_x + four_x) // 2
                center_y = (one_y + four_y - 100) // 2

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
                center_x_hai = (eleven_x + twelve_x) // 2
                center_y_hai = (eleven_y + twelve_y + 100) // 2

                # 距離を使って拡大縮小の倍率を決定
                dx_hai = eleven_x - twelve_x
                dy_hai = eleven_y - twelve_y
                eye_distance_hai = int((dx_hai ** 2 + dy_hai ** 2) ** 0.5)

                # スケール倍率（5pxのとき1倍）
                scale_hai = eye_distance_hai / 5.0
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
            center_x_i = (eleven_x + twenty_four_x - 50) // 2
            center_y_i = ((eleven_y + twenty_four_y - 100) // 2) + 50

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
        
        # 画像をRGBに変換してStreamlitで表示（StreamlitはRGB形式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB")

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