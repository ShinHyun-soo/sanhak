import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd
from collections import deque
import math

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_pixel_height(landmarks, frame_height):
    """키(픽셀) 계산"""
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    if nose.visibility > 0.5 and left_heel.visibility > 0.5 and right_heel.visibility > 0.5:
        avg_heel_y = (left_heel.y + right_heel.y) / 2
        return abs((avg_heel_y - nose.y) * frame_height)
    return 0

def determine_view_mode(landmarks):
    """
    어깨 너비와 몸통 높이의 비율을 이용해 정면/측면을 판단합니다.
    - Return: "FRONT" or "SIDE"
    """
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # 가시성이 너무 낮으면 판단 불가 (기본값 Side)
    if l_sh.visibility < 0.5 or r_sh.visibility < 0.5:
        return "SIDE"

    # 어깨 너비 (X축 거리)
    shoulder_width = abs(l_sh.x - r_sh.x)
    
    # 몸통 높이 (어깨 중점 ~ 골반 중점 Y축 거리)
    mid_sh_y = (l_sh.y + r_sh.y) / 2
    mid_hip_y = (l_hip.y + r_hip.y) / 2
    torso_height = abs(mid_sh_y - mid_hip_y)

    if torso_height == 0: return "SIDE"

    # 비율 계산 (어깨 너비 / 몸통 높이)
    # 일반적인 성인 기준 정면이면 이 비율이 0.5 ~ 0.6 이상 나옴
    ratio = shoulder_width / torso_height

    if ratio > 0.45: # 임계값 (조정 가능)
        return "FRONT"
    else:
        return "SIDE"

def calibrate_and_detect_view(video_path, pose_model):
    """
    캘리브레이션을 하면서 영상이 정면인지 측면인지도 함께 판단합니다.
    """
    cap = cv2.VideoCapture(video_path)
    max_pixel_height = 0
    frame_count = 0
    
    view_votes = [] # 프레임별 판정 결과 투표

    while frame_count < 60: # 첫 60프레임 분석
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(image)

        if results.pose_landmarks:
            # 1. 키 측정
            pixel_height = calculate_pixel_height(results.pose_landmarks.landmark, h)
            if pixel_height > max_pixel_height:
                max_pixel_height = pixel_height
            
            # 2. 뷰 모드 판별 (투표)
            view_mode = determine_view_mode(results.pose_landmarks.landmark)
            view_votes.append(view_mode)
        
        frame_count += 1
        
    cap.release()
    
    # 가장 많이 나온 판정 결과 선택 (Majority Vote)
    final_view = max(set(view_votes), key=view_votes.count) if view_votes else "SIDE"
    
    return max_pixel_height, final_view

def find_local_maxima_index(data, window_size=15):
    if len(data) < window_size: return -1
    center_index = len(data) - (window_size // 2) - 1
    if center_index < 0: return -1
    window = list(data)[center_index - (window_size//2) : center_index + (window_size//2) + 1]
    if max(window) == data[center_index]: return center_index
    return -1

def calculate_tilt(p1, p2):
    if p1.visibility < 0.5 or p2.visibility < 0.5: return None
    dy = p1.y - p2.y
    dx = p1.x - p2.x
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg > 90: angle_deg -= 180
    elif angle_deg < -90: angle_deg += 180
    return abs(angle_deg)

def process_frame_analysis(frame, pose_model, state):
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    step_length_in_meters = None 
    balance_metrics = {}
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # --- 균형 분석 (항상 계산) ---
        shoulder_tilt = calculate_tilt(l_sh, r_sh)
        hip_tilt = calculate_tilt(l_hip, r_hip)
        
        if shoulder_tilt is not None:
            balance_metrics['shoulder_tilt'] = shoulder_tilt
            cv2.line(image, (int(l_sh.x * w), int(l_sh.y * h)), (int(r_sh.x * w), int(r_sh.y * h)), (255, 0, 0), 2)
        
        if hip_tilt is not None:
            balance_metrics['hip_tilt'] = hip_tilt
            cv2.line(image, (int(l_hip.x * w), int(l_hip.y * h)), (int(r_hip.x * w), int(r_hip.y * h)), (0, 255, 0), 2)

        # --- 보폭 분석 (항상 계산) ---
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        curr_sep = abs(l_ankle.x - r_ankle.x) if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5 else 0
        state.setdefault('sep_history', deque(maxlen=15)).append(curr_sep)
        state.setdefault('frames_since_step', 0)
        state['frames_since_step'] += 1
        
        peak_idx = find_local_maxima_index(state['sep_history'], 15)
        if peak_idx != -1 and state['frames_since_step'] > 10:
            step_val = state['sep_history'][peak_idx]
            if state.get('scale') and step_val > 0:
                step_length_in_meters = step_val * w * state['scale']
                cv2.putText(image, f"Step: {step_length_in_meters:.2f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            state['frames_since_step'] = 0
            
    return image, step_length_in_meters, balance_metrics

# --- UI ---
st.set_page_config(page_title="AI 스마트 보행 분석기", layout="wide")
st.title("🎥 AI 스마트 보행 분석기 (자동 감지)")

st.sidebar.header("1. 설정")
person_actual_height = st.sidebar.number_input("피사체 키 (m)", 1.0, 2.0, 1.70)

st.sidebar.header("2. 영상 업로드")
f1 = st.sidebar.file_uploader("영상 1", type=["mp4", "mov", "avi"])
f2 = st.sidebar.file_uploader("영상 2", type=["mp4", "mov", "avi"])

col1, col2 = st.columns(2)

if st.button("분석 시작"):
    if (f1 or f2) and person_actual_height:
        pose1, pose2 = mp_pose.Pose(), mp_pose.Pose()
        cap1, cap2 = None, None
        stframe1, stframe2 = None, None
        state1, state2 = {}, {}
        step_data1, step_data2 = [], []
        bal_data1, bal_data2 = {'s':[], 'h':[]}, {'s':[], 'h':[]}
        
        view_mode1, view_mode2 = "SIDE", "SIDE" # 기본값

        with st.spinner("AI가 영상 구도(정면/측면)를 분석 중입니다..."):
            if f1:
                t1 = tempfile.NamedTemporaryFile(delete=False); t1.write(f1.read())
                ph1, vm1 = calibrate_and_detect_view(t1.name, pose1)
                state1['scale'] = person_actual_height / ph1 if ph1 > 0 else 0
                view_mode1 = vm1
                cap1 = cv2.VideoCapture(t1.name)
                with col1: 
                    stframe1 = st.empty()
                    st.info(f"영상 1 감지됨: **{vm1} (측면)**" if vm1=="SIDE" else f"영상 1 감지됨: **{vm1} (정면)**")

            if f2:
                t2 = tempfile.NamedTemporaryFile(delete=False); t2.write(f2.read())
                ph2, vm2 = calibrate_and_detect_view(t2.name, pose2)
                state2['scale'] = person_actual_height / ph2 if ph2 > 0 else 0
                view_mode2 = vm2
                cap2 = cv2.VideoCapture(t2.name)
                with col2: 
                    stframe2 = st.empty()
                    st.info(f"영상 2 감지됨: **{vm2} (측면)**" if vm2=="SIDE" else f"영상 2 감지됨: **{vm2} (정면)**")

        # 영상 처리 루프 (생략 없이 동일 로직 수행)
        proc1, proc2 = bool(cap1), bool(cap2)
        while proc1 or proc2:
            if proc1:
                ret, frame = cap1.read()
                if ret:
                    img, step, bal = process_frame_analysis(frame, pose1, state1)
                    stframe1.image(img, channels="BGR", use_container_width=True)
                    if step: step_data1.append(step)
                    if bal.get('shoulder_tilt'): bal_data1['s'].append(bal['shoulder_tilt'])
                    if bal.get('hip_tilt'): bal_data1['h'].append(bal['hip_tilt'])
                else: proc1 = False
            
            if proc2:
                ret, frame = cap2.read()
                if ret:
                    img, step, bal = process_frame_analysis(frame, pose2, state2)
                    stframe2.image(img, channels="BGR", use_container_width=True)
                    if step: step_data2.append(step)
                    if bal.get('shoulder_tilt'): bal_data2['s'].append(bal['shoulder_tilt'])
                    if bal.get('hip_tilt'): bal_data2['h'].append(bal['hip_tilt'])
                else: proc2 = False

        st.divider()
        st.header("📊 분석 결과 (자동 추천)")

        # 어떤 탭을 기본으로 보여줄지 결정
        # 하나라도 FRONT가 있으면 '균형' 탭을 중요하게 표시하거나 안내
        detected_front = (view_mode1 == "FRONT" or view_mode2 == "FRONT")
        
        tab1, tab2 = st.tabs(["🚶 보폭 분석 (측면 권장)", "⚖️ 균형 분석 (정면 권장)"])

        # 탭 내용 채우기
        with tab1:
            if detected_front: st.caption("⚠️ 정면 영상이 감지되었습니다. 보폭 데이터가 정확하지 않을 수 있습니다.")
            s1 = pd.Series(step_data1).dropna()
            s2 = pd.Series(step_data2).dropna()
            c1, c2 = st.columns(2)
            if not s1.empty: c1.metric("영상 1 보폭", f"{s1.mean():.2f}m"); c1.line_chart(s1)
            if not s2.empty: c2.metric("영상 2 보폭", f"{s2.mean():.2f}m"); c2.line_chart(s2)

        with tab2:
            if not detected_front: st.caption("⚠️ 측면 영상만 감지되었습니다. 좌우 균형 데이터가 의미 없을 수 있습니다.")
            b1, b2 = st.columns(2)
            with b1:
                if bal_data1['s']: 
                    st.metric("영상 1 어깨 기울기", f"{np.mean(bal_data1['s']):.1f}°")
                    st.line_chart(bal_data1['s'])
            with b2:
                if bal_data2['s']: 
                    st.metric("영상 2 어깨 기울기", f"{np.mean(bal_data2['s']):.1f}°")
                    st.line_chart(bal_data2['s'])
