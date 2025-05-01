# -*- coding: utf-8 -*-
"""
이미지 내 얼굴을 감지하고, 얼굴 랜드마크(눈 중심) 또는 바운딩 박스 중심을 기준으로
3분할 법칙 또는 황금 비율에 맞춰 자동으로 크롭하는 스크립트 (명령줄 인수 사용).

DNN 모델(YuNet)을 사용하여 얼굴 및 랜드마크를 감지합니다.
버전: 1.1.0
"""
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any, Union

# tqdm import 시도 (없으면 기능 비활성화)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs): # tqdm이 없을 경우 대체 함수
        print("INFO: tqdm 라이브러리가 설치되지 않아 진행 표시줄을 생략합니다. (pip install tqdm)")
        return iterable

__version__ = "1.1.0"

# --- 기본 설정값 (명령줄 인수로 덮어쓸 수 있음) ---
DEFAULT_OUTPUT_DIR = "output_final"
DEFAULT_SELECTION_METHOD = 'largest' # 'largest' or 'center'
DEFAULT_REFERENCE_POINT = 'eye'     # 'eye' or 'box'
DEFAULT_ASPECT_RATIO = None        # 예: "16:9", "1.0", "4:3", None (원본 유지)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3

# DNN 얼굴 검출 모델 파일 경로 및 URL
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# 결과 파일명 접미사
OUTPUT_SUFFIX_THIRDS = '_thirds'
OUTPUT_SUFFIX_GOLDEN = '_golden'
# --- 설정값 끝 ---

# --- 유틸리티 함수 ---

def download_model(url: str, file_path: str) -> bool:
    """지정된 URL에서 모델 파일을 다운로드합니다."""
    if not os.path.exists(file_path):
        print(f"INFO: 모델 파일 다운로드 중... ({os.path.basename(file_path)})")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("INFO: 다운로드 완료.")
        except Exception as e:
            print(f"ERROR: 모델 파일 다운로드 실패: {e}")
            print(f"       수동으로 다음 URL에서 다운로드 받아 '{file_path}'로 저장해주세요: {url}")
            return False
    return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """문자열 형태의 비율(예: '16:9', '1.0', 'None')을 float으로 변환합니다."""
    if ratio_str is None or ratio_str.lower() == 'none':
        return None
    try:
        if ':' in ratio_str:
            w_str, h_str = ratio_str.split(':')
            w, h = float(w_str), float(h_str)
            if h <= 0 or w <= 0: return None # 너비/높이가 0 이하면 유효하지 않음
            return w / h
        else:
            ratio = float(ratio_str)
            return ratio if ratio > 0 else None # 0보다 큰 비율만 유효
    except ValueError:
        print(f"WARN: 잘못된 비율 문자열 형식입니다: '{ratio_str}'. 원본 비율을 사용합니다.")
        return None

# --- 핵심 로직 함수 ---

def detect_faces_dnn(image: np.ndarray, model_path: str, conf_threshold: float, nms_threshold: float) -> List[Dict[str, Any]]:
    """DNN 모델(YuNet)을 사용하여 얼굴 목록(bbox, bbox_center, eye_center, confidence)을 감지합니다."""
    # (이전 버전과 동일 - 내부 로직 변경 없음)
    if not os.path.exists(model_path):
        print(f"ERROR: DNN 모델 파일을 찾을 수 없습니다: {model_path}")
        return []
    img_h, img_w = image.shape[:2]
    detected_subjects = []
    try:
        detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0))
        detector.setInputSize((img_w, img_h))
        detector.setScoreThreshold(conf_threshold)
        detector.setNMSThreshold(nms_threshold)
        faces = detector.detect(image)
        if faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                x, y, w, h = map(int, face_info[:4])
                r_eye_x, r_eye_y = face_info[4:6]
                l_eye_x, l_eye_y = face_info[6:8]
                confidence = face_info[14]
                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h)
                if w > 0 and h > 0:
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center
                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0:
                        ecx = int(round((r_eye_x + l_eye_x) / 2))
                        ecy = int(round((r_eye_y + l_eye_y) / 2))
                        ecx = max(0, min(img_w - 1, ecx)); ecy = max(0, min(img_h - 1, ecy))
                        eye_center = (ecx, ecy)
                    else: print(f"WARN: 얼굴 ID {idx}의 눈 랜드마크가 유효하지 않아 BBox 중심을 사용합니다.")
                    detected_subjects.append({'bbox': (x, y, w, h), 'bbox_center': bbox_center, 'eye_center': eye_center, 'confidence': confidence})
    except Exception as e: print(f"ERROR: DNN 얼굴 감지 중 문제 발생: {e}")
    return detected_subjects


def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'eye') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """감지된 피사체 목록에서 주 피사체를 선택하고 기준점을 반환합니다."""
    # (이전 버전과 동일 - 내부 로직 변경 없음)
    if not subjects: return None
    best_subject = None
    if len(subjects) == 1: best_subject = subjects[0]
    elif method == 'largest': best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
    elif method == 'center':
        img_h, img_w = img_shape; img_center = (img_w / 2, img_h / 2)
        min_dist = float('inf')
        for s in subjects:
            dist = math.dist(s['bbox_center'], img_center)
            if dist < min_dist: min_dist = dist; best_subject = s
    else:
        if method != 'largest': print(f"WARN: 알 수 없는 주 피사체 선택 방법 '{method}'. 'largest' 사용.")
        best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
    ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
    return best_subject['bbox'], ref_center


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """구도 법칙(3분할 또는 황금비율)에 따른 교차점 목록을 반환합니다."""
    # (이전 버전과 동일 - 내부 로직 변경 없음)
    points = []
    if rule_type == 'thirds': points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
    elif rule_type == 'golden':
        phi_inv = (math.sqrt(5) - 1) / 2
        lines_w = (width * (1 - phi_inv), width * phi_inv); lines_h = (height * (1 - phi_inv), height * phi_inv)
        points = [(w, h) for w in lines_w for h in lines_h]
    else: points = [(width / 2, height / 2)]
    return [(int(round(px)), int(round(py))) for px, py in points]


def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Tuple[int, int, int, int]:
    """주어진 기준점, 구도점, 비율에 맞춰 최적의 크롭 영역(x1, y1, x2, y2)을 계산합니다."""
    height, width = img_shape
    # **개선:** 높이가 0 이하인 경우 처리
    if height <= 0 or width <= 0:
        print("WARN: 이미지 높이 또는 너비가 0 이하이므로 크롭할 수 없습니다.")
        return 0, 0, width, height

    cx, cy = subject_center
    aspect_ratio = target_aspect_ratio if target_aspect_ratio else (width / height)

    closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
    target_x, target_y = closest_point

    max_w = 2 * min(target_x, width - target_x)
    max_h = 2 * min(target_y, height - target_y)

    if max_w <= 0 or max_h <= 0:
         print("WARN: 타겟 포인트가 이미지 경계에 있어 유효한 크롭 불가.")
         return 0, 0, width, height

    # **개선:** aspect_ratio가 0 이하인 경우 (이론상 parse_aspect_ratio에서 걸러지지만 방어 코드)
    if aspect_ratio <= 0:
        print(f"WARN: 유효하지 않은 비율({aspect_ratio})로 계산 불가. 원본 크기 반환.")
        return 0, 0, width, height

    crop_h_from_w = max_w / aspect_ratio
    crop_w_from_h = max_h * aspect_ratio

    if crop_h_from_w <= max_h + 1e-6: # 부동 소수점 오차 감안
        final_w, final_h = max_w, crop_h_from_w
    else:
        final_w, final_h = crop_w_from_h, max_h

    x1 = target_x - final_w / 2; y1 = target_y - final_h / 2
    x2 = x1 + final_w; y2 = y1 + final_h

    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

    if x1 >= x2 or y1 >= y2:
        print("WARN: 유효한 크롭 영역 계산 불가 (크기 0).")
        return 0, 0, width, height

    return x1, y1, x2, y2

# --- 메인 처리 함수 ---

def process_image(image_path: str, output_dir: str, args: argparse.Namespace):
    """단일 이미지를 처리하여 크롭된 결과를 저장합니다."""
    # print(f"INFO: 처리 시작 - {os.path.basename(image_path)}") # tqdm 사용 시 중복 출력 방지

    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)[:, :, ::-1].copy()
    except FileNotFoundError:
        print(f"ERROR: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    except Exception as e:
        print(f"ERROR: 이미지 로드 실패 ({image_path}): {e}")
        return

    img_h, img_w = img.shape[:2]
    if img_h <= 0 or img_w <= 0:
        print(f"WARN: 유효하지 않은 이미지 크기 ({img_w}x{img_h}) - 건너<0xEB><0x84><0x8E>니다: {os.path.basename(image_path)}")
        return

    detected_faces = detect_faces_dnn(img, YUNET_MODEL_PATH, args.confidence, args.nms)
    if not detected_faces:
        # tqdm 사용 시 줄바꿈을 위해 print 대신 file 인자 사용 고려 가능
        # print(f"INFO: 얼굴 감지 실패 - {os.path.basename(image_path)}")
        return # 실패 시 조용히 넘어감 (로그 레벨 조정으로 변경 가능)

    selection_result = select_main_subject(detected_faces, (img_h, img_w), args.method, args.reference)
    if not selection_result:
         # print(f"ERROR: 주 피사체 선택 실패 - {os.path.basename(image_path)}")
         return
    subj_bbox, ref_center = selection_result
    # print(f"INFO: 주 피사체 선택 완료 - {os.path.basename(image_path)}") # 상세 로그 필요 시 주석 해제

    base, ext = os.path.splitext(os.path.basename(image_path))
    # ratio_str 생성 시 args.ratio가 None일 경우 처리
    target_ratio_str = args.ratio if args.ratio else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"
    ref_str = f"_ref{args.reference}"

    crop_configs = [('thirds', OUTPUT_SUFFIX_THIRDS), ('golden', OUTPUT_SUFFIX_GOLDEN)]
    saved_count = 0
    for rule_name, suffix in crop_configs:
        rule_points = get_rule_points(img_w, img_h, rule_name)
        target_ratio_float = parse_aspect_ratio(args.ratio) # 명령줄 인자 사용
        x1, y1, x2, y2 = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, target_ratio_float)

        if x1 < x2 and y1 < y2:
            cropped_img = img[y1:y2, x1:x2]
            out_filename = f"{base}{suffix}{ratio_str}{ref_str}{ext}"
            out_path = os.path.join(output_dir, out_filename)
            try:
                cv2.imwrite(out_path, cropped_img)
                saved_count += 1
                # print(f"INFO: 저장 완료 - {out_filename}") # 성공 시 로그 생략 가능
            except Exception as e:
                print(f"ERROR: 크롭 이미지 저장 실패 ({out_path}): {e}")
        # else: print(f"WARN: '{rule_name}' 규칙 유효 크롭 영역 생성 실패 - {os.path.basename(image_path)}")

    # return saved_count # 배치 처리 시 성공 카운트 반환 등에 사용 가능

# --- 명령줄 인터페이스 및 실행 ---

def main():
    parser = argparse.ArgumentParser(
        description=f"{__doc__}\n버전: {__version__}", # Docstring과 버전 정보 포함
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="처리할 이미지 파일 또는 디렉토리 경로.")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"결과 저장 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-m", "--method", choices=['largest', 'center'], default=DEFAULT_SELECTION_METHOD, help=f"주 피사체 선택 방법 (기본값: {DEFAULT_SELECTION_METHOD})\n  largest: 가장 큰 얼굴\n  center: 이미지 중앙에 가장 가까운 얼굴")
    parser.add_argument("-ref", "--reference", choices=['eye', 'box'], default=DEFAULT_REFERENCE_POINT, help=f"구도 기준점 타입 (기본값: {DEFAULT_REFERENCE_POINT})\n  eye: 양 눈의 중심\n  box: 얼굴 경계 상자의 중심")
    parser.add_argument("-r", "--ratio", type=str, default=DEFAULT_ASPECT_RATIO, help="목표 크롭 비율 (예: '16:9', '1.0', '4:3', 'None').\n'None' 또는 미지정 시 원본 비율 유지 (기본값).")
    parser.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f"얼굴 감지 최소 신뢰도 (기본값: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("-n", "--nms", type=float, default=DEFAULT_NMS_THRESHOLD, help=f"얼굴 감지 NMS 임계값 (기본값: {DEFAULT_NMS_THRESHOLD})")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}') # 버전 표시 인수 추가

    args = parser.parse_args()

    if not download_model(YUNET_MODEL_URL, YUNET_MODEL_PATH):
        print("ERROR: DNN 모델 파일이 준비되지 않아 처리를 중단합니다.")
        return

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"INFO: 출력 디렉토리 생성: {args.output_dir}")
        except OSError as e:
            print(f"ERROR: 출력 디렉토리 생성 실패: {e}")
            return

    if os.path.isfile(args.input_path):
        process_image(args.input_path, args.output_dir, args)
        print(f"INFO: 단일 파일 처리 완료 - {os.path.basename(args.input_path)}") # 완료 메시지 추가
    elif os.path.isdir(args.input_path):
        print(f"INFO: 디렉토리 처리 시작 - {args.input_path}")
        image_files = [f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("INFO: 처리할 이미지 파일이 디렉토리에 없습니다.")
            return

        # tqdm 적용하여 파일 처리 진행 표시
        for filename in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(args.input_path, filename)
            process_image(file_path, args.output_dir, args)
            
        print(f"INFO: 디렉토리 처리 완료 ({len(image_files)}개 파일 처리 시도)")
    else:
        print(f"ERROR: 입력 경로를 찾을 수 없습니다: {args.input_path}")

if __name__ == "__main__":
    main()