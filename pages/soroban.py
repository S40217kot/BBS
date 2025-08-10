import streamlit as st

st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    pass
if st.sidebar.button("AR テルミン"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        import time as ti
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
    with st.spinner("モジュールのロード中です\nしばらくお待ちください"):
        import cv2
        import mediapipe as mp
        import time as ti
        import os
        import math

    with st.spinner("画像の読み込み中です\nしばらくお待ちください"):
        # 透過PNG画像の読み込み
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_image_path_tama = os.path.join(current_dir, '..', 'Images', 'tama1.png')
        target_image_tama = cv2.imread(target_image_path_tama, cv2.IMREAD_UNCHANGED)
        target_size_tama = 40  # 画像の直径
        target_image_tama = cv2.resize(target_image_tama, (target_size_tama, target_size_tama))

    with st.spinner("変数の定義中です\nしばらくお待ちください"):
        # 何列目はすべて右から数えます
        # 何行目はすべて上から数えます

        # 1列目x座標
        gazou_x_tama_1_1 = 239
        # 2列目x座標
        gazou_x_tama_1_2 = 309
        # 3列目x座標
        gazou_x_tama_1_3 = 379
        # 4列目x座標
        gazou_x_tama_1_4 = 449
        # 5列目x座標
        gazou_x_tama_1_5 = 519
        # 6列目x座標
        gazou_x_tama_1_6 = 589

        # 1行目y座標
        gazou_y_tama_1_1 = gazou_y_tama_2_1 = gazou_y_tama_3_1 = gazou_y_tama_4_1 = gazou_y_tama_5_1 = gazou_y_tama_6_1 = 127
        # 2行目y座標
        gazou_y_tama_1_2 = gazou_y_tama_2_2 = gazou_y_tama_3_2 = gazou_y_tama_4_2 = gazou_y_tama_5_2 = gazou_y_tama_6_2 = 258
        # 3行目y座標
        gazou_y_tama_1_3 = gazou_y_tama_2_3 = gazou_y_tama_3_3 = gazou_y_tama_4_3 = gazou_y_tama_5_3 = gazou_y_tama_6_3 = 293
        # 4行目y座標
        gazou_y_tama_1_4 = gazou_y_tama_2_4 = gazou_y_tama_3_4 = gazou_y_tama_4_4 = gazou_y_tama_5_4 = gazou_y_tama_6_4 = 328
        # 5行目y座標
        gazou_y_tama_1_5 = gazou_y_tama_2_5 = gazou_y_tama_3_5 = gazou_y_tama_4_5 = gazou_y_tama_5_5 = gazou_y_tama_6_5 = 363

        # 1桁目
        keta_1 = 0
        # 2桁目
        keta_2 = 0
        # 3桁目
        keta_3 = 0
        # 4桁目
        keta_4 = 0
        # 5桁目
        keta_5 = 0
        # 6桁目
        keta_6 = 0

    with st.spinner("カメラ映像の取得中です\nしばらくお待ちください"):
        # カメラを起動（0番はデフォルトカメラ）
        cap = cv2.VideoCapture(0)
        cap.isOpened(): 
            print("エラー: カメラを開けませんでした。")

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

    with st.spinner("定数の定義中です\nしばらくお待ちください"):
        def is_hand_touching_gazou(hand_x, hand_y, gazou_x, gazou_y, gazou_radius):
            distance = math.sqrt((hand_x - gazou_x) ** 2 + (hand_y - gazou_y) ** 2)
            return distance < gazou_radius
        
        ################### もっとスマートに書けるかも

        def mobe_tama(wich , column, line):
            global keta_1, keta_2, keta_3, keta_4, keta_5, keta_6, gazou_y_tama_1_1, gazou_y_tama_1_2, gazou_y_tama_1_3, gazou_y_tama_1_4, gazou_y_tama_1_5, gazou_y_tama_2_1, gazou_y_tama_2_2, gazou_y_tama_2_3, gazou_y_tama_2_4, gazou_y_tama_2_5, gazou_y_tama_3_1, gazou_y_tama_3_2, gazou_y_tama_3_3, gazou_y_tama_3_4, gazou_y_tama_3_5, gazou_y_tama_4_1, gazou_y_tama_4_2, gazou_y_tama_4_3, gazou_y_tama_4_4, gazou_y_tama_4_5, gazou_y_tama_5_1, gazou_y_tama_5_2, gazou_y_tama_5_3, gazou_y_tama_5_4, gazou_y_tama_5_5, gazou_y_tama_6_1, gazou_y_tama_6_2, gazou_y_tama_6_3, gazou_y_tama_6_4, gazou_y_tama_6_5, gazou_x_tama_1_1, gazou_x_tama_1_2, gazou_x_tama_1_3, gazou_x_tama_1_4, gazou_x_tama_1_5, gazou_x_tama_1_6
            if wich == 'u':
                if column == 1:
                    if line == 1:
                        if gazou_y_tama_1_1 == 127:
                            gazou_y_tama_1_1 = 89
                            keta_1 += 5
                    if line == 2:
                        if gazou_y_tama_1_2 == 258:
                            gazou_y_tama_1_2 = 190
                            keta_1 += 1
                    if line == 3:
                        if gazou_y_tama_1_3 == 293:
                            gazou_y_tama_1_3 = 225
                            keta_1 += 1
                    if line == 4:
                        if gazou_y_tama_1_4 == 328:
                            gazou_y_tama_1_4 = 260
                            keta_1 += 1
                    if line == 5:
                        if gazou_y_tama_1_5 == 363:
                            gazou_y_tama_1_5 = 295
                            keta_1 += 1
                if column == 2:
                    if line == 1:
                        if gazou_y_tama_2_1 == 127:
                            gazou_y_tama_2_1 = 89
                            keta_2 += 5
                    if line == 2:
                        if gazou_y_tama_2_2 == 258:
                            gazou_y_tama_2_2 = 190
                            keta_2 += 1
                    if line == 3:
                        if gazou_y_tama_2_3 == 293:
                            gazou_y_tama_2_3 = 225
                            keta_2 += 1
                    if line == 4:
                        if gazou_y_tama_2_4 == 328:
                            gazou_y_tama_2_4 = 260
                            keta_2 += 1
                    if line == 5:
                        if gazou_y_tama_2_5 == 363:
                            gazou_y_tama_2_5 = 295
                            keta_2 += 1
                if column == 3:
                    if line == 1:
                        if gazou_y_tama_3_1 == 127:
                            gazou_y_tama_3_1 = 89
                            keta_3 += 5
                    if line == 2:
                        if gazou_y_tama_3_2 == 258:
                            gazou_y_tama_3_2 = 190
                            keta_3 += 1
                    if line == 3:
                        if gazou_y_tama_3_3 == 293:
                            gazou_y_tama_3_3 = 225
                            keta_3 += 1
                    if line == 4:
                        if gazou_y_tama_3_4 == 328:
                            gazou_y_tama_3_4 = 260
                            keta_3 += 1
                    if line == 5:
                        if gazou_y_tama_3_5 == 363:
                            gazou_y_tama_3_5 = 295
                            keta_3 += 1
                if column == 4:
                    if line == 1:
                        if gazou_y_tama_4_1 == 127:
                            gazou_y_tama_4_1 = 89
                            keta_4 += 5
                    if line == 2:
                        if gazou_y_tama_4_2 == 258:
                            gazou_y_tama_4_2 = 190
                            keta_4 += 1
                    if line == 3:
                        if gazou_y_tama_4_3 == 293:
                            gazou_y_tama_4_3 = 225
                            keta_4 += 1
                    if line == 4:
                        if gazou_y_tama_4_4 == 328:
                            gazou_y_tama_4_4 = 260
                            keta_4 += 1
                    if line == 5:
                        if gazou_y_tama_4_5 == 363:
                            gazou_y_tama_4_5 = 295
                            keta_4 += 1
                if column == 5:
                    if line == 1:
                        if gazou_y_tama_5_1 == 127:
                            gazou_y_tama_5_1 = 89
                            keta_5 += 5
                    if line == 2:
                        if gazou_y_tama_5_2 == 258:
                            gazou_y_tama_5_2 = 190
                            keta_5 += 1
                    if line == 3:
                        if gazou_y_tama_5_3 == 293:
                            gazou_y_tama_5_3 = 225
                            keta_5 += 1
                    if line == 4:
                        if gazou_y_tama_5_4 == 328:
                            gazou_y_tama_5_4 = 260
                            keta_5 += 1
                    if line == 5:
                        if gazou_y_tama_5_5 == 363:
                            gazou_y_tama_5_5 = 295
                            keta_5 += 1
                if column == 6:
                    if line == 1:
                        if gazou_y_tama_6_1 == 127:
                            gazou_y_tama_6_1 = 89
                            keta_6 += 5
                    if line == 2:
                        if gazou_y_tama_6_2 == 258:
                            gazou_y_tama_6_2 = 190
                            keta_6 += 1
                    if line == 3:
                        if gazou_y_tama_6_3 == 293:
                            gazou_y_tama_6_3 = 225
                            keta_6 += 1
                    if line == 4:
                        if gazou_y_tama_6_4 == 328:
                            gazou_y_tama_6_4 = 260
                            keta_6 += 1
                    if line == 5:
                        if gazou_y_tama_6_5 == 363:
                            gazou_y_tama_6_5 = 295
                            keta_6 += 1
            if wich == 'd':
                if column == 1:
                    if line == 1:
                        if gazou_y_tama_1_1 == 89:
                            gazou_y_tama_1_1 = 127
                            keta_1 -= 5
                    if line == 2:
                        if gazou_y_tama_1_2 == 190:
                            gazou_y_tama_1_2 = 258
                            keta_1 -= 1
                    if line == 3:
                        if gazou_y_tama_1_3 == 225:
                            gazou_y_tama_1_3 = 293
                            keta_1 -= 1
                    if line == 4:
                        if gazou_y_tama_1_4 == 260:
                            gazou_y_tama_1_4 = 328
                            keta_1 -= 1
                    if line == 5:
                        if gazou_y_tama_1_5 == 295:
                            gazou_y_tama_1_5 = 363
                            keta_1 -= 1
                if column == 2:
                    if line == 1:
                        if gazou_y_tama_2_1 == 89:
                            gazou_y_tama_2_1 = 127
                            keta_2 -= 5
                    if line == 2:
                        if gazou_y_tama_2_2 == 190:
                            gazou_y_tama_2_2 = 258
                            keta_2 -= 1
                    if line == 3:
                        if gazou_y_tama_2_3 == 225:
                            gazou_y_tama_2_3 = 293
                            keta_2 -= 1
                    if line == 4:
                        if gazou_y_tama_2_4 == 260:
                            gazou_y_tama_2_4 = 328
                            keta_2 -= 1
                    if line == 5:
                        if gazou_y_tama_2_5 == 295:
                            gazou_y_tama_2_5 = 363
                            keta_2 -= 1
                if column == 3:
                    if line == 1:
                        if gazou_y_tama_3_1 == 89:
                            gazou_y_tama_3_1 = 127
                            keta_3 -= 5
                    if line == 2:
                        if gazou_y_tama_3_2 == 190:
                            gazou_y_tama_3_2 = 258
                            keta_3 -= 1
                    if line == 3:
                        if gazou_y_tama_3_3 == 225:
                            gazou_y_tama_3_3 = 293
                            keta_3 -= 1
                    if line == 4:
                        if gazou_y_tama_3_4 == 260:
                            gazou_y_tama_3_4 = 328
                            keta_3 -= 1
                    if line == 5:
                        if gazou_y_tama_3_5 == 295:
                            gazou_y_tama_3_5 = 363
                            keta_3 -= 1
                if column == 4:
                    if line == 1:
                        if gazou_y_tama_4_1 == 89:
                            gazou_y_tama_4_1 = 127
                            keta_4 -= 5
                    if line == 2:
                        if gazou_y_tama_4_2 == 190:
                            gazou_y_tama_4_2 = 258
                            keta_4 -= 1
                    if line == 3:
                        if gazou_y_tama_4_3 == 225:
                            gazou_y_tama_4_3 = 293
                            keta_4 -= 1
                    if line == 4:
                        if gazou_y_tama_4_4 == 260:
                            gazou_y_tama_4_4 = 328
                            keta_4 -= 1
                    if line == 5:
                        if gazou_y_tama_4_5 == 295:
                            gazou_y_tama_4_5 = 363
                            keta_4 -= 1
                if column == 5:
                    if line == 1:
                        if gazou_y_tama_5_1 == 89:
                            gazou_y_tama_5_1 = 127
                            keta_5 -= 5
                    if line == 2:
                        if gazou_y_tama_5_2 == 190:
                            gazou_y_tama_5_2 = 258
                            keta_5 -= 1
                    if line == 3:
                        if gazou_y_tama_5_3 == 225:
                            gazou_y_tama_5_3 = 293
                            keta_5 -= 1
                    if line == 4:
                        if gazou_y_tama_5_4 == 260:
                            gazou_y_tama_5_4 = 328
                            keta_5 -= 1
                    if line == 5:
                        if gazou_y_tama_5_5 == 295:
                            gazou_y_tama_5_5 = 363
                            keta_5 -= 1
                if column == 6:
                    if line == 1:
                        if gazou_y_tama_6_1 == 89:
                            gazou_y_tama_6_1 = 127
                            keta_6 -= 5
                    if line == 2:
                        if gazou_y_tama_6_2 == 190:
                            gazou_y_tama_6_2 = 258
                            keta_6 -= 1
                    if line == 3:
                        if gazou_y_tama_6_3 == 225:
                            gazou_y_tama_6_3 = 293
                            keta_6 -= 1
                    if line == 4:
                        if gazou_y_tama_6_4 == 260:
                            gazou_y_tama_6_4 = 328
                            keta_6 -= 1
                    if line == 5:
                        if gazou_y_tama_6_5 == 295:
                            gazou_y_tama_6_5 = 363
                            keta_6 -= 1
    
# ページの配置
st.title('AR そろばん')
placeholder = st.empty()
if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
         ti.sleep(1)
         st.write(f"<meta http-equiv='refresh' content='0;url=/?page=h'>", unsafe_allow_html=True)
st.markdown("<b>使い方<b>", unsafe_allow_html=True)
st.write("親指で球を上げます")
st.write("人差し指で球を下げます")
st.write("一番右が一の位で左に行くにつれ二の位三の位・・・となっていきます。")
st.write("一番上のたまが五でその下四つが一です")
st.write("上記の方法で球を上げ下げし計算します。")

# メインループ
try:
    while True:
        ret, frame = cap.read()  # カメラから1フレーム取得
        if not ret:
            break

        # OpenCVはRGB形式なので、RGBに変換
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
    
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 1, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 1, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 1, 2)
                    mobe_tama('d', 1, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 1, 2)
                    mobe_tama('d', 1, 3)
                    mobe_tama('d', 1, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 1, 2)
                    mobe_tama('d', 1, 3)
                    mobe_tama('d', 1, 4)
                    mobe_tama('d', 1, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 2, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 2, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 2, 2)
                    mobe_tama('d', 2, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 2, 2)
                    mobe_tama('d', 2, 3)
                    mobe_tama('d', 2, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 2, 2)
                    mobe_tama('d', 2, 3)
                    mobe_tama('d', 2, 4)
                    mobe_tama('d', 2, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 3, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 3, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 3, 2)
                    mobe_tama('d', 3, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 3, 2)
                    mobe_tama('d', 3, 3)
                    mobe_tama('d', 3, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 3, 2)
                    mobe_tama('d', 3, 3)
                    mobe_tama('d', 3, 4)
                    mobe_tama('d', 3, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 4, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 4, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 4, 2)
                    mobe_tama('d', 4, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 4, 2)
                    mobe_tama('d', 4, 3)
                    mobe_tama('d', 4, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 4, 2)
                    mobe_tama('d', 4, 3)
                    mobe_tama('d', 4, 4)
                    mobe_tama('d', 4, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 5, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 5, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 5, 2)
                    mobe_tama('d', 5, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 5, 2)
                    mobe_tama('d', 5, 3)
                    mobe_tama('d', 5, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 5, 2)
                    mobe_tama('d', 5, 3)
                    mobe_tama('d', 5, 4)
                    mobe_tama('d', 5, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_1_1, 20):
                    mobe_tama('d', 6, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_1_2, 20):
                    mobe_tama('d', 6, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_1_3, 20):
                    mobe_tama('d', 6, 2)
                    mobe_tama('d', 6, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_1_4, 20):
                    mobe_tama('d', 6, 2)
                    mobe_tama('d', 6, 3)
                    mobe_tama('d', 6, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_1_5, 20):
                    mobe_tama('d', 6, 2)
                    mobe_tama('d', 6, 3)
                    mobe_tama('d', 6, 4)
                    mobe_tama('d', 6, 5)
        
        # 何列目はすべて右から数えます
        # 何行目はすべて上から数えます
        # 帽を描画
        # 1列目
        cv2.rectangle(frame, (236, 65), (241, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 2列目
        cv2.rectangle(frame, (306, 65), (311, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 3列目
        cv2.rectangle(frame, (376, 65), (381, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 4列目
        cv2.rectangle(frame, (446, 65), (451, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 5列目
        cv2.rectangle(frame, (516, 65), (521, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 6列目
        cv2.rectangle(frame, (586, 65), (591, 380), (0, 65, 128), cv2.FILLED, cv2.LINE_AA)
        # 1行目
        cv2.rectangle(frame, (200, 45), (610, 70), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
        # 2行目
        cv2.rectangle(frame, (200, 145), (610, 170), (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        # 3行目
        cv2.rectangle(frame, (200, 380), (610, 410), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)

        # ランドマークを元のBGR画像に描画（OpenCVの画像はBGR形式）
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS 
                )

                hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                
                # 現在の指先位置を表示
                cv2.circle(frame, (hand_x, hand_y), 5, (0, 255, 0), -1)
    
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_1, 20):
                    mobe_tama('u', 1, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_2, 20):
                    mobe_tama('u', 1, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_3, 20):
                    mobe_tama('u', 1, 2)
                    mobe_tama('u', 1, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_4, 20):
                    mobe_tama('u', 1, 2)
                    mobe_tama('u', 1, 3)
                    mobe_tama('u', 1, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_1, gazou_y_tama_1_5, 20):
                    mobe_tama('u', 1, 2)
                    mobe_tama('u', 1, 3)
                    mobe_tama('u', 1, 4)
                    mobe_tama('u', 1, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_2_1, 20):
                    mobe_tama('u', 2, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_2_2, 20):
                    mobe_tama('u', 2, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_2_3, 20):
                    mobe_tama('u', 2, 2)
                    mobe_tama('u', 2, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_2_4, 20):
                    mobe_tama('u', 2, 2)
                    mobe_tama('u', 2, 3)
                    mobe_tama('u', 2, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_2, gazou_y_tama_2_5, 20):
                    mobe_tama('u', 2, 2)
                    mobe_tama('u', 2, 3)
                    mobe_tama('u', 2, 4)
                    mobe_tama('u', 2, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_3_1, 20):
                    mobe_tama('u', 3, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_3_2, 20):
                    mobe_tama('u', 3, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_3_3, 20):
                    mobe_tama('u', 3, 2)
                    mobe_tama('u', 3, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_3_4, 20):
                    mobe_tama('u', 3, 2)
                    mobe_tama('u', 3, 3)
                    mobe_tama('u', 3, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_3, gazou_y_tama_3_5, 20):
                    mobe_tama('u', 3, 2)
                    mobe_tama('u', 3, 3)
                    mobe_tama('u', 3, 4)
                    mobe_tama('u', 3, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_4_1, 20):
                    mobe_tama('u', 4, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_4_2, 20):
                    mobe_tama('u', 4, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_4_3, 20):
                    mobe_tama('u', 4, 2)
                    mobe_tama('u', 4, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_4_4, 20):
                    mobe_tama('u', 4, 2)
                    mobe_tama('u', 4, 3)
                    mobe_tama('u', 4, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_4, gazou_y_tama_4_5, 20):
                    mobe_tama('u', 4, 2)
                    mobe_tama('u', 4, 3)
                    mobe_tama('u', 4, 4)
                    mobe_tama('u', 4, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_5_1, 20):
                    mobe_tama('u', 5, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_5_2, 20):
                    mobe_tama('u', 5, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_5_3, 20):
                    mobe_tama('u', 5, 2)
                    mobe_tama('u', 5, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_5_4, 20):
                    mobe_tama('u', 5, 2)
                    mobe_tama('u', 5, 3)
                    mobe_tama('u', 5, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_5, gazou_y_tama_5_5, 20):
                    mobe_tama('u', 5, 2)
                    mobe_tama('u', 5, 3)
                    mobe_tama('u', 5, 4)
                    mobe_tama('u', 5, 5)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_6_1, 20):
                    mobe_tama('u', 6, 1)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_6_2, 20):
                    mobe_tama('u', 6, 2)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_6_3, 20):
                    mobe_tama('u', 6, 2)
                    mobe_tama('u', 6, 3)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_6_4, 20):
                    mobe_tama('u', 6, 2)
                    mobe_tama('u', 6, 3)
                    mobe_tama('u', 6, 4)
                if is_hand_touching_gazou(hand_x, hand_y, gazou_x_tama_1_6, gazou_y_tama_6_5, 20):
                    mobe_tama('u', 6, 2)
                    mobe_tama('u', 6, 3)
                    mobe_tama('u', 6, 4)
                    mobe_tama('u', 6, 5)

        # 何列目はすべて右から数えます
        # 何行目はすべて上から数えます
        # 画像を配置
        # 1列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_1 - w // 2
            y_offset = gazou_y_tama_1_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
            # 1列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_1 - w // 2
            y_offset = gazou_y_tama_1_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 1列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_1 - w // 2
            y_offset = gazou_y_tama_1_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 1列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_1 - w // 2
            y_offset = gazou_y_tama_1_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 1列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_1 - w // 2
            y_offset = gazou_y_tama_1_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 2列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_2 - w // 2
            y_offset = gazou_y_tama_2_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 2列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_2 - w // 2
            y_offset = gazou_y_tama_2_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 2列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_2 - w // 2
            y_offset = gazou_y_tama_2_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 2列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_2 - w // 2
            y_offset = gazou_y_tama_2_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 2列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_2 - w // 2
            y_offset = gazou_y_tama_2_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 3列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_3 - w // 2
            y_offset = gazou_y_tama_3_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 3列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_3 - w // 2
            y_offset = gazou_y_tama_3_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 3列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_3 - w // 2
            y_offset = gazou_y_tama_3_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 3列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_3 - w // 2
            y_offset = gazou_y_tama_3_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 3列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_3 - w // 2
            y_offset = gazou_y_tama_3_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 4列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_4 - w // 2
            y_offset = gazou_y_tama_4_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 4列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_4 - w // 2
            y_offset = gazou_y_tama_4_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 4列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_4 - w // 2
            y_offset = gazou_y_tama_4_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 4列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_4 - w // 2
            y_offset = gazou_y_tama_4_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 4列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_4 - w // 2
            y_offset = gazou_y_tama_4_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 5列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_5 - w // 2
            y_offset = gazou_y_tama_5_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 5列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_5 - w // 2
            y_offset = gazou_y_tama_5_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 5列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_5 - w // 2
            y_offset = gazou_y_tama_5_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 5列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_5 - w // 2
            y_offset = gazou_y_tama_5_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 5列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_5 - w // 2
            y_offset = gazou_y_tama_5_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 6列目1行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_6 - w // 2
            y_offset = gazou_y_tama_6_1 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 6列目2行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_6 - w // 2
            y_offset = gazou_y_tama_6_2 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 6列目3行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_6 - w // 2
            y_offset = gazou_y_tama_6_3 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 6列目4行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_6 - w // 2
            y_offset = gazou_y_tama_6_4 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
        # 6列目5行目
        if target_image_tama is not None:
            h, w = target_image_tama.shape[:2]
            x_offset = gazou_x_tama_1_6 - w // 2
            y_offset = gazou_y_tama_6_5 - h // 2
            if x_offset >= 0 and y_offset >= 0 and x_offset + w <= frame.shape[1] and y_offset + h <= frame.shape[0]:
                if target_image_tama.shape[2] == 4:
                    alpha_channel = target_image_tama[:, :, 3] / 255.0
                    overlay_colors = target_image_tama[:, :, :3]
                    for c in range(3):
                        frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
                            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha_channel) + \
                            overlay_colors[:, :, c] * alpha_channel
                else:
                    frame[y_offset:y_offset+h, x_offset:x_offset+w] = target_image_tama
    
        cv2.circle(frame, (589, 160), 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{keta_1}{keta_2}{keta_3}{keta_4}{keta_5}{keta_6}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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




