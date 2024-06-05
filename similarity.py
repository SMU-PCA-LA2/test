import os
import numpy as np
import dlib
import joblib
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# dlib의 얼굴 탐지기와 랜드마크 추출기 초기화
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('C:\\Users\\LeeDongwon\\Desktop\\workspace\\opencv\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('C:\\Users\\LeeDongwon\\Desktop\\workspace\\opencv\\dlib_face_recognition_resnet_model_v1.dat')

def find_landmarks(img):
    # 이미지에서 얼굴 탐지
    dets = detector(img, 1)

    # 랜드마크 좌표를 저장할 배열 초기화
    landmarks = []
    for k, d in enumerate(dets):
        # 얼굴 영역에서 랜드마크 추출
        shape = sp(img, d)

        # dlib shape를 numpy 배열로 변환하여 저장
        landmark = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        landmarks.append(landmark)

    return landmarks

def encode_faces(img, landmarks):
    face_descriptors = []
    for landmark in landmarks:
        # 얼굴 랜드마크를 dlib.full_object_detection 형태로 변환
        shape = dlib.full_object_detection(
            dlib.rectangle(0, 0, img.shape[1], img.shape[0]),
            [dlib.point(pt[0], pt[1]) for pt in landmark]
        )
        # 얼굴 랜드마크를 사용하여 얼굴의 특징 벡터 계산
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return face_descriptors

def detect_faces_webcam(classifier, pca, threshold):
    cap = cv2.VideoCapture(0)

    fixed_box_position = (220, 170)  # 고정된 위치 좌표

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지를 수평으로 반전
        frame = cv2.flip(frame, 1)

        # 이미지 크기를 줄여서 처리
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # 얼굴 인식
        landmarks = find_landmarks(frame_small)

        for landmark in landmarks:
            # 얼굴 특징 벡터 추출
            face_descriptors = encode_faces(frame_small, [landmark])
            reduced_face_descriptor = pca.transform(face_descriptors)
            
            # 분류기로 예측
            prediction = classifier.predict(reduced_face_descriptor)
            
            # 웹캠에서 얻은 얼굴 벡터와 분류기로부터 얻은 얼굴 벡터의 유사도 계산
            similarity = cosine_similarity(reduced_face_descriptor, pca.transform([classifier.support_vectors_[0]]))[0][0]
            similarity_percentage = (similarity + 1) / 2 * 100  # -1 to 1 범위를 0 to 100%로 변환
            
            if similarity_percentage > threshold:
                initial = prediction[0]
            else:
                initial = "Unknown"

            # 얼굴에 대한 네모 박스 그리기
            x, y = landmark[0][0] * 2, landmark[0][1] * 2
            w, h = (landmark[16][0] - landmark[0][0]) * 2, (landmark[8][1] - landmark[24][1]) * 2  # 얼굴 크기에 맞게 조절
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 이니셜 및 유사도 표시
            similarity_text = f"{initial} ({similarity_percentage:.2f}%)"
            cv2.putText(frame, similarity_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 고정된 위치에 박스 그리기
        x, y = fixed_box_position
        w, h = 230, 230  # 박스의 크기를 100x100으로 설정
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 저장된 모델 로드
    classifier = joblib.load('C:\\Users\\LeeDongwon\\Desktop\\workspace\\opencv\\classifier.joblib')
    pca = joblib.load('C:\\Users\\LeeDongwon\\Desktop\\workspace\\opencv\\pca.joblib')
    threshold = 70  # 유사도 임계값 설정 (백분율 기준)

    # 웹캠으로부터 얼굴 인식하고 이니셜 및 유사도 표시
    detect_faces_webcam(classifier, pca, threshold)
