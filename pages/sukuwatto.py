import streamlit as st

st.title("Border Break Studies")

if st.sidebar.button("AR そろばん"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/soroban.py")
if st.sidebar.button("AR テルミン"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/terumin.py")
if st.sidebar.button("AR パレット"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/paretto.py")
if st.sidebar.button("AR 人体模型"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("pages/jintai.py")
if st.sidebar.button("AR スクワット"):
    pass
if st.sidebar.button("Home"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        st.switch_page("main.py")

# ページの移動がある場合
# urlを取得
out = st.session_state.get("mokuhyo", "None")
# ページに飛ばす
if out != None:
    mokuhyo = out
    pass
else:
    import time as ti
    st.error("申し訳ございません\nこちら側のページへの直接アクセスはできません。")
    st.warning("このメッセージが表示されてから2秒後に目標回数入力ページに自動的に転送されます。")
    ti.sleep(2)
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.write(f"<meta http-equiv='refresh' content='0;url=.sukuwa'>", unsafe_allow_html=True)
        exit()

# 入力された値を検査する
if mokuhyo.isdecimal():
    squat_counter = int(mokuhyo)
    pass
else:
    import time as ti
    st.error("数値以外が入力されました\n数値のみの受付となっております")
    st.warning("このメッセージが表示されてから2秒後に目標回数入力ページに自動的に転送されます。")
    ti.sleep(2)
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
        ti.sleep(1)
        st.switch_page("pages/sukuwa.py")
        exit()
    exit()

with st.spinner('読み込み中です\nしばらくお待ちください'):

    with st.spinner("モジュールの読み込み中です\nしばらくお待ちください"):
       import mediapipe as mp
       import time as ti
       import cv2
       import math
       import numpy as np
       from PIL import ImageFont, ImageDraw, Image

    with st.spinner("カメラ映像の取得中です\nしばらくお待ちください"):
        # カメラを起動（0番はデフォルトカメラ）
        cap = cv2.VideoCapture(0)

    with st.spinner("MediaPipiPoseの取得中です\nしばらくお待ちください"):
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
            font = ImageFont.truetype('C:\Windows\Fonts\BIZ-UDGothicR.ttc', size)

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

# ページの配置
st.title('AR スクワット')
placeholder = st.empty()
if st.button("ホームへ"):
    with st.spinner('リダイレクト中です\nしばらくお待ちください'):
         ti.sleep(1)
         st.write(f"<meta http-equiv='refresh' content='0;url=/'>", unsafe_allow_html=True)
st.write("使い方")
st.text("...")

# メインループ
try:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()  # カメラから1フレーム取得
            if not ret:
                break

            # OpenCVはBGR形式なので、まずRGBに変換
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            image_rgb = cv2.flip(image_rgb, 1)

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
                        squat_counter -= 1

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

            cv2.imshow('ARsquat', frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

            if squat_counter == 0:
                with st.spinner('リダイレクト中です\nしばらくお待ちください'):
                    ti.sleep(1)
                    st.write(f"<meta http-equiv='refresh' content='0;url=.sukuwa?situation=finish&mokuhyo={mokuhyo}'>", unsafe_allow_html=True)
                    exit()
                break

            # 画像をRGBに変換してStreamlitで表示（StreamlitはRGB形式）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB")  # ★ランドマーク描画後の画像を表示

except Exception as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
except RuntimeError as e:
    st.error(f"申し上げございません\nシステム内部で問題が発生しました：{e}")
finally:
    # リソースの解放

    cap.release()


