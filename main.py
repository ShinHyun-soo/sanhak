import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd # pandas 라이브러리 추가

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """두 점 사이의 유클리드 거리를 계산하는 함수"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def process_frame(frame, pose_model):
    """단일 프레임을 처리하여 보폭을 계산하고 랜드마크를 그리는 함수"""
    
    # 성능 향상을 위해 프레임 크기 조절 및 색상 변환
    h, w, _ = frame.shape
    frame_small = cv2.resize(frame, (int(w/2), int(h/2)))
    image = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    results = pose_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    stride_in_meters = None
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:
            stride_dist_pixels = calculate_distance(left_ankle, right_ankle)
            
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            hip_dist_pixels = calculate_distance(left_hip, right_hip)

            if hip_dist_pixels > 0.05:
                scale = 0.5 / hip_dist_pixels 
                stride_in_meters = stride_dist_pixels * scale
                
                cv2.putText(image, f"Stride: {stride_in_meters:.2f} m", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                            
    return image, stride_in_meters

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="보폭 비교 분석기", page_icon="📊", layout="wide")

st.title("📊 동영상 보폭 비교 분석기")
st.write("두 개의 걷는 동영상을 업로드하여 보폭 변화를 하나의 그래프에서 비교 분석합니다.")
st.info("정확한 분석을 위해 **측면에서 촬영되고, 전신이 잘 보이는** 동영상을 사용해 주세요.", icon="💡")

col1, col2 = st.columns(2)

with col1:
    st.header("동영상 1")
    uploaded_file1 = st.file_uploader("첫 번째 동영상을 업로드하세요.", type=["mp4", "mov", "avi"], key="file1")

with col2:
    st.header("동영상 2")
    uploaded_file2 = st.file_uploader("두 번째 동영상을 업로드하세요.", type=["mp4", "mov", "avi"], key="file2")

if st.button("두 영상 분석 시작하기", use_container_width=True):
    if uploaded_file1 and uploaded_file2:
        
        pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose2 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        tfile1 = tempfile.NamedTemporaryFile(delete=False)
        tfile1.write(uploaded_file1.read())
        video_path1 = tfile1.name

        tfile2 = tempfile.NamedTemporaryFile(delete=False)
        tfile2.write(uploaded_file2.read())
        video_path2 = tfile2.name

        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        
        with col1:
            stframe1 = st.empty()
        with col2:
            stframe2 = st.empty()
            
        # 통합 차트 영역을 컬럼 바깥에 생성
        st.header("보폭 변화 비교 그래프")
        chart_spot = st.empty()
            
        stride_data1, stride_data2 = [], []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        max_frames = max(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_count = 0

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 and not ret2:
                break
            
            frame_count +=1

            if ret1:
                processed_frame1, stride1 = process_frame(frame1, pose1)
                stframe1.image(processed_frame1, channels="BGR", use_column_width=True)
                stride_data1.append(stride1 if stride1 is not None else np.nan) # 데이터가 없을 경우 NaN 추가

            if ret2:
                processed_frame2, stride2 = process_frame(frame2, pose2)
                stframe2.image(processed_frame2, channels="BGR", use_column_width=True)
                stride_data2.append(stride2 if stride2 is not None else np.nan) # 데이터가 없을 경우 NaN 추가

            # --- 차트 업데이트 로직 변경 ---
            # 두 데이터의 길이를 맞추기 위해 짧은 쪽에 NaN 추가
            len1, len2 = len(stride_data1), len(stride_data2)
            max_len = max(len1, len2)
            
            # pandas 데이터프레임 생성
            chart_data = pd.DataFrame({
                '동영상 1': stride_data1 + [np.nan] * (max_len - len1),
                '동영상 2': stride_data2 + [np.nan] * (max_len - len2),
            })
            
            # 통합 차트 업데이트
            chart_spot.line_chart(chart_data)

            progress = frame_count / max_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"분석 진행률: {int(min(progress, 1.0) * 100)}%")

        cap1.release()
        cap2.release()
        pose1.close()
        pose2.close()
        
        status_text.success("분석 완료!")
        st.balloons()
        
    else:
        st.warning("⚠️ 두 개의 동영상을 모두 업로드해주세요.")