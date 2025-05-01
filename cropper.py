# -*- coding: utf-8 -*-
"""
이미지 내 얼굴을 감지하고, 얼굴 랜드마크(눈 중심) 또는 바운딩 박스 중심을 기준으로
3분할 법칙 또는 황금 비율에 맞춰 자동으로 크롭하는 스크립트 (명령줄 인수 사용).
- 원본 EXIF 데이터 보존
- DNN 모델 사전 로딩으로 효율성 향상
- logging 모듈 사용
- 덮어쓰기, 출력 형식, JPEG 품질 제어 옵션 추가
- 최소 얼굴 크기, 크롭 패딩, 구도 규칙 선택 옵션 추가
- 병렬 처리 기능 추가 (디렉토리 처리 시)
- 오류 요약 보고 기능 추가

DNN 모델(YuNet)을 사용하여 얼굴 및 랜드마크를 감지합니다.

필요한 라이브러리:
pip install opencv-python numpy Pillow tqdm

버전: 1.5.0 (병렬 처리 및 오류 요약 추가)
"""
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging
import time
import concurrent.futures # 병렬 처리를 위해 추가
from PIL import Image, UnidentifiedImageError, Exif
from typing import Tuple, List, Optional, Dict, Any, Union

# tqdm import 시도 (없으면 기능 비활성화)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        logging.info("tqdm 라이브러리가 설치되지 않아 진행 표시줄을 생략합니다. (pip install tqdm)")
        return iterable

__version__ = "1.5.0" # 버전 업데이트

# --- 기본 설정값 ---
# (이전 버전과 동일)
DEFAULT_OUTPUT_DIR = "output_final"
DEFAULT_SELECTION_METHOD = 'largest'
DEFAULT_REFERENCE_POINT = 'eye'
DEFAULT_ASPECT_RATIO = None
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_OVERWRITE = True
DEFAULT_OUTPUT_FORMAT = None
DEFAULT_JPEG_QUALITY = 95
DEFAULT_MIN_FACE_WIDTH = 30
DEFAULT_MIN_FACE_HEIGHT = 30
DEFAULT_PADDING_PERCENT = 5
DEFAULT_RULE = 'both'
DEFAULT_WORKERS = os.cpu_count() # 병렬 처리 기본 작업자 수 (CPU 코어 수)

# DNN 모델 정보
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# 결과 파일명 접미사
OUTPUT_SUFFIX_THIRDS = '_thirds'
OUTPUT_SUFFIX_GOLDEN = '_golden'
# --- 설정값 끝 ---

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 유틸리티 함수 ---
# (download_model, parse_aspect_ratio 함수는 이전 버전과 동일)
def download_model(url: str, file_path: str) -> bool:
    """지정된 URL에서 모델 파일을 다운로드합니다."""
    if not os.path.exists(file_path):
        logging.info(f"모델 파일 다운로드 중... ({os.path.basename(file_path)})")
        try:
            urllib.request.urlretrieve(url, file_path)
            logging.info("다운로드 완료.")
            return True
        except Exception as e:
            logging.error(f"모델 파일 다운로드 실패: {e}")
            logging.error(f"       수동으로 다음 URL에서 다운로드 받아 '{file_path}'로 저장해주세요: {url}")
            return False
    else:
        logging.info(f"모델 파일 '{os.path.basename(file_path)}'이(가) 이미 존재합니다.")
        return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """문자열 형태의 비율(예: '16:9', '1.0', 'None')을 float으로 변환합니다."""
    if ratio_str is None or ratio_str.lower() == 'none':
        return None
    try:
        if ':' in ratio_str:
            w_str, h_str = ratio_str.split(':')
            w, h = float(w_str), float(h_str)
            if h <= 0 or w <= 0:
                logging.warning(f"비율의 너비 또는 높이가 0 이하일 수 없습니다: '{ratio_str}'. 원본 비율 사용.")
                return None
            return w / h
        else:
            ratio = float(ratio_str)
            if ratio <= 0:
                logging.warning(f"비율은 0보다 커야 합니다: '{ratio_str}'. 원본 비율 사용.")
                return None
            return ratio
    except ValueError:
        logging.warning(f"잘못된 비율 문자열 형식입니다: '{ratio_str}'. 원본 비율을 사용합니다.")
        return None
    except Exception as e:
        logging.error(f"비율 파싱 중 예상치 못한 오류 발생 ('{ratio_str}'): {e}")
        return None

# --- 핵심 로직 함수 ---
# (detect_faces_dnn, select_main_subject, get_rule_points, calculate_optimal_crop, apply_padding 함수는 이전 버전과 동일)
def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray, min_w: int, min_h: int) -> List[Dict[str, Any]]:
    """
    사전 로드된 DNN 모델(YuNet)을 사용하여 얼굴 목록을 감지합니다.
    지정된 최소 크기(min_w, min_h)보다 작은 얼굴은 제외합니다.
    """
    detected_subjects = []
    if image is None or image.size == 0:
        logging.warning("얼굴 감지를 위한 입력 이미지가 비어 있습니다.")
        return []

    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        logging.warning(f"유효하지 않은 이미지 크기 ({img_w}x{img_h})로 얼굴 감지를 건너<0xEB><0x84><0x8E>니다.")
        return []

    try:
        # 입력 크기 설정 (이미지 크기와 동일하게)
        detector.setInputSize((img_w, img_h))
        # 얼굴 감지 수행
        faces = detector.detect(image)

        # 감지된 얼굴 정보 처리
        if faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                # 얼굴 경계 상자 좌표 (정수형 변환)
                x, y, w, h = map(int, face_info[:4])

                # --- 최소 얼굴 크기 필터링 ---
                if w < min_w or h < min_h:
                    logging.debug(f"얼굴 ID {idx} ({w}x{h})가 최소 크기({min_w}x{min_h})보다 작아 무시합니다.")
                    continue
                # --- 필터링 끝 ---

                # 눈 랜드마크 좌표
                r_eye_x, r_eye_y = face_info[4:6]
                l_eye_x, l_eye_y = face_info[6:8]
                # 신뢰도 점수
                confidence = face_info[14]

                # 경계 상자 좌표 보정 (이미지 경계 내)
                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h) # 보정된 w, h 사용

                # 유효한 경계 상자인 경우 처리 (보정 후에도 크기가 유효한지 확인)
                if w > 0 and h > 0:
                    # 경계 상자 중심 계산
                    bbox_center = (x + w // 2, y + h // 2)
                    eye_center = bbox_center # 기본값은 bbox 중심

                    # 눈 랜드마크가 유효한 경우 눈 중심 계산
                    if r_eye_x > 0 and r_eye_y > 0 and l_eye_x > 0 and l_eye_y > 0:
                        ecx = int(round((r_eye_x + l_eye_x) / 2))
                        ecy = int(round((r_eye_y + l_eye_y) / 2))
                        # 눈 중심 좌표 보정 (이미지 경계 내)
                        ecx = max(0, min(img_w - 1, ecx)); ecy = max(0, min(img_h - 1, ecy))
                        eye_center = (ecx, ecy)
                    else:
                        logging.debug(f"얼굴 ID {idx}의 눈 랜드마크가 유효하지 않아 BBox 중심을 사용합니다.")

                    # 감지된 피사체 정보 추가
                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        # 얼굴 감지 중 OpenCV 관련 오류 발생 시 로깅
        logging.error(f"OpenCV 오류 발생 (얼굴 감지 중): {e}")
    except Exception as e:
        # 그 외 예상치 못한 오류 발생 시 로깅
        logging.error(f"DNN 얼굴 감지 중 예상치 못한 문제 발생: {e}")
    # 감지된 얼굴 목록 반환
    return detected_subjects


def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'eye') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """감지된 피사체 목록에서 주 피사체를 선택하고 기준점을 반환합니다."""
    # 피사체 목록이 비어있으면 None 반환
    if not subjects:
        logging.debug("주 피사체 선택: 감지된 피사체가 없습니다.")
        return None

    img_h, img_w = img_shape # 이미지 크기
    best_subject = None # 최적 피사체 초기화

    try:
        # 피사체가 하나면 바로 선택
        if len(subjects) == 1:
            best_subject = subjects[0]
        # 'largest' 방법: 가장 큰 바운딩 박스 면적 기준
        elif method == 'largest':
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        # 'center' 방법: 이미지 중심에서 가장 가까운 바운딩 박스 중심 기준
        elif method == 'center':
            img_center = (img_w / 2, img_h / 2) # 이미지 중심 좌표
            # 각 피사체의 bbox 중심과 이미지 중심 간의 유클리드 거리 계산하여 최소 거리 피사체 선택
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else:
            # 알 수 없는 방법이면 기본값 'largest' 사용 (경고는 main 함수에서 처리)
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        # 선택된 주 피사체의 기준점(눈 중심 또는 바운딩 박스 중심) 결정
        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"주 피사체 선택됨 (방법: {method}, 기준점: {reference_point_type}). BBox: {best_subject['bbox']}")
        # 주 피사체의 바운딩 박스와 기준점 좌표 튜플 반환
        return best_subject['bbox'], ref_center

    except Exception as e:
        # 주 피사체 선택 중 오류 발생 시 로깅하고 None 반환
        logging.error(f"주 피사체 선택 중 오류 발생: {e}")
        return None


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """구도 법칙(3분할 또는 황금비율)에 따른 교차점 목록을 반환합니다."""
    points = [] # 교차점 좌표 리스트 초기화
    # 이미지 너비 또는 높이가 0 이하이면 빈 리스트 반환
    if width <= 0 or height <= 0:
        logging.warning(f"유효하지 않은 크기({width}x{height})로 구도점 계산 불가.")
        return []

    try:
        # 'thirds' (3분할) 규칙
        if rule_type == 'thirds':
            # 가로, 세로 1/3, 2/3 지점들의 교차점 4개 계산
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        # 'golden' (황금비율) 규칙
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2 # 황금비율의 역수 (약 0.618)
            # 황금 분할 선 좌표 계산
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            # 황금 분할 선들의 교차점 4개 계산
            points = [(w, h) for w in lines_w for h in lines_h]
        # 알 수 없는 규칙이면 이미지 중심점 사용
        else:
            logging.warning(f"알 수 없는 구도 규칙 '{rule_type}'. 이미지 중심 사용.")
            points = [(width / 2, height / 2)]

        # 계산된 교차점 좌표를 정수로 반올림하여 반환
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        # 구도점 계산 중 오류 발생 시 로깅하고 빈 리스트 반환
        logging.error(f"구도점 계산 중 오류 발생 (규칙: {rule_type}): {e}")
        return []


def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    """주어진 기준점, 구도점, 비율에 맞춰 최적의 크롭 영역(x1, y1, x2, y2)을 계산합니다."""
    height, width = img_shape # 이미지 높이, 너비
    # 높이 또는 너비가 0 이하이면 None 반환
    if height <= 0 or width <= 0:
        logging.warning(f"이미지 높이({height}) 또는 너비({width})가 0 이하이므로 크롭할 수 없습니다.")
        return None

    # 구도점이 없으면 None 반환
    if not rule_points:
        logging.warning("구도점이 제공되지 않아 크롭 계산을 건너<0xEB><0x84><0x8E>니다.")
        return None

    cx, cy = subject_center # 주 피사체 기준점 좌표

    try:
        # 목표 종횡비 계산 (미지정 시 원본 비율 사용, 높이 0 방지)
        if height > 0:
            aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        else:
             logging.warning("이미지 높이가 0이어서 원본 비율을 계산할 수 없습니다. 크롭 불가.")
             return None

        # 종횡비 유효성 검사
        if aspect_ratio <= 0:
            logging.warning(f"유효하지 않은 비율({aspect_ratio})로 계산 불가.")
            return None

        # 기준점에서 가장 가까운 구도 교차점 찾기
        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point # 이 점을 크롭 영역의 중심으로 사용

        # 타겟 포인트를 중심으로 가질 수 있는 최대 크롭 너비/높이 계산
        # (타겟 포인트가 이미지 경계에 너무 가까우면 0이 될 수 있음)
        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)

        # 최대 너비 또는 높이가 0 이하이면 유효한 크롭 불가
        if max_w <= 0 or max_h <= 0:
             logging.debug("타겟 포인트가 이미지 경계에 있어 유효한 크롭 불가.")
             return None

        # 목표 종횡비에 맞춰 크롭 크기 계산
        # 1. 최대 너비(max_w) 기준 높이 계산
        crop_h_from_w = max_w / aspect_ratio
        # 2. 최대 높이(max_h) 기준 너비 계산
        crop_w_from_h = max_h * aspect_ratio

        # 두 계산 결과 중 이미지 경계(max_w, max_h) 내에 맞는 크기 선택
        if crop_h_from_w <= max_h + 1e-6: # 부동 소수점 오차 감안
            final_w, final_h = max_w, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h

        # 크롭 영역 좌상단(x1, y1), 우하단(x2, y2) 좌표 계산 (중심 기준)
        x1 = target_x - final_w / 2
        y1 = target_y - final_h / 2
        x2 = x1 + final_w
        y2 = y1 + final_h

        # 좌표를 정수로 반올림하고 이미지 경계 내로 제한
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        # 최종 크롭 영역 크기 유효성 검사 (너비 또는 높이가 0인지 확인)
        if x1 >= x2 or y1 >= y2:
            logging.warning("계산된 크롭 영역의 크기가 0입니다.")
            return None

        logging.debug(f"계산된 크롭 영역: ({x1}, {y1}) - ({x2}, {y2})")
        # 최종 크롭 영역 좌표 튜플 반환
        return x1, y1, x2, y2

    except Exception as e:
        # 크롭 계산 중 오류 발생 시 로깅하고 None 반환
        logging.error(f"최적 크롭 계산 중 오류 발생: {e}")
        return None


def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    """계산된 크롭 영역에 패딩을 적용하고 이미지 경계 내로 조정합니다."""
    x1, y1, x2, y2 = crop_coords # 원본 크롭 좌표
    img_h, img_w = img_shape # 이미지 크기
    crop_w = x2 - x1 # 크롭 너비
    crop_h = y2 - y1 # 크롭 높이

    # 패딩 비율이 0 이하면 원본 좌표 반환
    if padding_percent <= 0:
        return crop_coords

    # 패딩량 계산 (크롭 영역 너비/높이 기준, 좌우/상하 절반씩)
    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    # 패딩 적용하여 새 좌표 계산
    new_x1 = x1 - pad_x
    new_y1 = y1 - pad_y
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y

    # 새 좌표를 이미지 경계 내로 제한
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    # 패딩 적용 후 크기가 유효한지(너비/높이가 0보다 큰지) 확인
    if new_x1 >= new_x2 or new_y1 >= new_y2:
        logging.warning("패딩 적용 후 크롭 영역 크기가 0이 되어 패딩을 적용하지 않습니다.")
        return crop_coords # 유효하지 않으면 원본 크롭 영역 반환

    logging.debug(f"패딩 적용된 크롭 영역 ({padding_percent}%): ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
    # 패딩 적용 및 조정된 좌표 튜플 반환
    return new_x1, new_y1, new_x2, new_y2


# --- 메인 처리 함수 ---

def process_image(image_path: str, output_dir: str, detector: cv2.FaceDetectorYN, args: argparse.Namespace) -> Dict[str, Any]:
    """
    단일 이미지를 처리하여 크롭된 결과를 저장합니다.
    성공/실패 상태와 메시지를 포함하는 딕셔너리를 반환합니다.
    """
    filename = os.path.basename(image_path)
    logging.debug(f"처리 시작: {filename}")
    start_time = time.time()
    status = {'filename': filename, 'success': False, 'saved_files': 0, 'message': ''}
    exif_data = None
    original_ext = os.path.splitext(image_path)[1].lower()

    try:
        # PIL로 이미지 열기
        with Image.open(image_path) as pil_img:
            # EXIF 데이터 추출 시도
            try:
                exif_data = pil_img.info.get('exif')
                if exif_data:
                    _ = Exif.load(exif_data) # 유효성 검사
                    logging.debug(f"{filename}: EXIF 데이터 로드 성공.")
            except Exception as exif_err:
                logging.warning(f"{filename}: EXIF 데이터 처리 중 오류: {exif_err}. EXIF 없이 저장됩니다.")
                exif_data = None

            # OpenCV 형식으로 변환
            pil_img_rgb = pil_img.convert('RGB')
            img = np.array(pil_img_rgb)[:, :, ::-1].copy()

    except FileNotFoundError:
        status['message'] = "이미지 파일을 찾을 수 없습니다."
        logging.error(f"{filename}: {status['message']}")
        return status
    except UnidentifiedImageError:
        status['message'] = "이미지 파일을 열 수 없거나 지원하지 않는 형식입니다."
        logging.error(f"{filename}: {status['message']}")
        return status
    except Exception as e:
        status['message'] = f"이미지 로드 중 오류 발생: {e}"
        logging.error(f"{filename}: {status['message']}")
        return status

    img_h, img_w = img.shape[:2]
    if img_h <= 0 or img_w <= 0:
        status['message'] = f"유효하지 않은 이미지 크기 ({img_w}x{img_h})."
        logging.warning(f"{filename}: {status['message']}")
        return status

    # 얼굴 감지
    detected_faces = detect_faces_dnn(detector, img, args.min_face_width, args.min_face_height)
    if not detected_faces:
        status['message'] = "유효한 얼굴 감지 실패 (최소 크기 조건 포함)."
        logging.info(f"{filename}: {status['message']}")
        # 얼굴 미감지는 오류는 아니므로 success=True로 간주할 수도 있음 (선택사항)
        # status['success'] = True # 얼굴 없는 것도 성공으로 처리하려면 주석 해제
        return status

    # 주 피사체 선택
    selection_result = select_main_subject(detected_faces, (img_h, img_w), args.method, args.reference)
    if not selection_result:
         status['message'] = "주 피사체 선택 실패."
         logging.warning(f"{filename}: {status['message']}")
         return status
    subj_bbox, ref_center = selection_result

    # 출력 파일명 및 확장자 설정
    base = os.path.splitext(filename)[0]
    target_ratio_str = args.ratio if args.ratio else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"
    ref_str = f"_ref{args.reference}"
    output_format = args.output_format.lower() if args.output_format else None
    if output_format:
        output_ext = f".{output_format.lstrip('.')}"
        if output_ext[1:] not in Image.registered_extensions():
             logging.warning(f"{filename}: 지원되지 않는 출력 형식 '{args.output_format}'. 원본 형식 '{original_ext}' 사용.")
             output_ext = original_ext
    else:
        output_ext = original_ext

    # 적용할 구도 규칙 결정
    rules_to_apply = []
    if args.rule == 'thirds' or args.rule == 'both':
        rules_to_apply.append(('thirds', OUTPUT_SUFFIX_THIRDS))
    if args.rule == 'golden' or args.rule == 'both':
        rules_to_apply.append(('golden', OUTPUT_SUFFIX_GOLDEN))

    if not rules_to_apply:
        status['message'] = f"적용할 구도 규칙이 선택되지 않았습니다 ('{args.rule}')."
        logging.warning(f"{filename}: {status['message']}")
        return status

    saved_count = 0
    crop_errors = []
    # 각 구도 규칙에 대해 크롭 및 저장
    for rule_name, suffix in rules_to_apply:
        # 크롭 영역 계산
        rule_points = get_rule_points(img_w, img_h, rule_name)
        target_ratio_float = parse_aspect_ratio(args.ratio)
        crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, target_ratio_float)

        if crop_coords:
            # 패딩 적용
            padded_coords = apply_padding(crop_coords, (img_h, img_w), args.padding_percent)
            x1, y1, x2, y2 = padded_coords

            # 크롭 실행
            cropped_img_bgr = img[y1:y2, x1:x2]

            if cropped_img_bgr.size == 0:
                msg = f"'{rule_name}' 규칙: 최종 크롭 영역 크기가 0."
                logging.warning(f"{filename}: {msg}")
                crop_errors.append(msg)
                continue

            # 출력 파일 경로
            out_filename = f"{base}{suffix}{ratio_str}{ref_str}{output_ext}"
            out_path = os.path.join(output_dir, out_filename)

            # 덮어쓰기 확인
            if not args.overwrite and os.path.exists(out_path):
                logging.info(f"{filename}: 파일이 이미 존재하고 덮어쓰기 비활성화됨 - 건너<0xEB><0x84><0x8E>기: {out_filename}")
                continue

            # 파일 저장
            try:
                cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                pil_cropped_img = Image.fromarray(cropped_img_rgb)
                save_options = {}
                if exif_data and isinstance(exif_data, bytes):
                    save_options['exif'] = exif_data
                if output_ext.lower() in ['.jpg', '.jpeg']:
                    save_options['quality'] = args.jpeg_quality
                    save_options['optimize'] = True
                    save_options['progressive'] = True
                pil_cropped_img.save(out_path, **save_options)
                logging.debug(f"{filename}: 저장 완료 - {out_filename}")
                saved_count += 1
            except IOError as e:
                 msg = f"파일 쓰기 오류 ({out_filename}): {e}"
                 logging.error(f"{filename}: {msg}")
                 crop_errors.append(msg)
            except Exception as e:
                msg = f"크롭 이미지 저장 중 오류 ({out_filename}): {e}"
                logging.error(f"{filename}: {msg}")
                crop_errors.append(msg)
        else:
             msg = f"'{rule_name}' 규칙: 유효 크롭 영역 생성 실패."
             logging.warning(f"{filename}: {msg}")
             crop_errors.append(msg)

    end_time = time.time()
    processing_time = end_time - start_time

    # 최종 상태 업데이트
    if saved_count > 0:
        status['success'] = True
        status['saved_files'] = saved_count
        status['message'] = f"처리 완료 ({saved_count}개 파일 저장, {processing_time:.2f}초 소요)."
        if crop_errors:
            status['message'] += f" 일부 오류 발생: {'; '.join(crop_errors)}"
        logging.info(f"{filename}: {status['message']}")
    elif detected_faces: # 얼굴 감지 O, 저장 X
        status['message'] = f"얼굴은 감지되었으나 유효한 크롭/저장 실패 ({processing_time:.2f}초 소요). 오류: {'; '.join(crop_errors) if crop_errors else '크롭 영역 계산 실패'}"
        logging.info(f"{filename}: {status['message']}")
    # 얼굴 미감지 경우는 이미 위에서 처리됨

    return status


# --- 병렬 처리 래퍼 함수 ---
def process_image_wrapper(args_tuple):
    """ concurrent.futures.Executor.map을 위한 래퍼 함수 """
    image_path, output_dir, detector, args_namespace = args_tuple
    # 각 프로세스/스레드에서 로깅 설정 (필요한 경우)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    try:
        return process_image(image_path, output_dir, detector, args_namespace)
    except Exception as e:
        # process_image 내부에서 처리되지 않은 예외 처리
        filename = os.path.basename(image_path)
        logging.error(f"{filename}: 처리 중 심각한 오류 발생: {e}", exc_info=True)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'message': f"처리 중 심각한 오류: {e}"}


# --- 명령줄 인터페이스 및 실행 ---

def main():
    # 명령줄 인수 파서 설정
    parser = argparse.ArgumentParser(
        description=f"{__doc__}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- 입력/출력 인수 ---
    parser.add_argument("input_path", help="처리할 이미지 파일 또는 디렉토리 경로.")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"결과 저장 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=DEFAULT_OVERWRITE, help="출력 파일 덮어쓰기 허용 여부 (기본값: --overwrite / 허용)")

    # --- 얼굴 감지 및 선택 인수 ---
    parser.add_argument("-m", "--method", choices=['largest', 'center'], default=DEFAULT_SELECTION_METHOD, help=f"주 피사체 선택 방법 (기본값: {DEFAULT_SELECTION_METHOD})")
    parser.add_argument("-ref", "--reference", choices=['eye', 'box'], default=DEFAULT_REFERENCE_POINT, help=f"구도 기준점 타입 (기본값: {DEFAULT_REFERENCE_POINT})")
    parser.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f"얼굴 감지 최소 신뢰도 (기본값: {DEFAULT_CONFIDENCE_THRESHOLD})")
    parser.add_argument("-n", "--nms", type=float, default=DEFAULT_NMS_THRESHOLD, help=f"얼굴 감지 NMS 임계값 (기본값: {DEFAULT_NMS_THRESHOLD})")
    parser.add_argument("--min-face-width", type=int, default=DEFAULT_MIN_FACE_WIDTH, help=f"처리할 최소 얼굴 너비 (픽셀, 기본값: {DEFAULT_MIN_FACE_WIDTH})")
    parser.add_argument("--min-face-height", type=int, default=DEFAULT_MIN_FACE_HEIGHT, help=f"처리할 최소 얼굴 높이 (픽셀, 기본값: {DEFAULT_MIN_FACE_HEIGHT})")

    # --- 크롭 및 구도 인수 ---
    parser.add_argument("-r", "--ratio", type=str, default=DEFAULT_ASPECT_RATIO, help="목표 크롭 비율 (예: '16:9', '1.0', 'None'). 'None' 또는 미지정 시 원본 비율 유지.")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], default=DEFAULT_RULE, help=f"적용할 구도 규칙 (기본값: {DEFAULT_RULE})")
    parser.add_argument("--padding-percent", type=float, default=DEFAULT_PADDING_PERCENT, help=f"크롭 영역 주변 패딩 비율 (%, 기본값: {DEFAULT_PADDING_PERCENT})")

    # --- 출력 형식 인수 ---
    parser.add_argument("--output-format", type=str, default=DEFAULT_OUTPUT_FORMAT, help="출력 이미지 형식 (예: 'jpg', 'png'). 미지정 시 원본 형식 유지.")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, choices=range(1, 101), metavar="[1-100]", help=f"JPEG 저장 품질 (1-100, 기본값: {DEFAULT_JPEG_QUALITY}). --output-format이 jpg/jpeg일 때 유효.")

    # --- 병렬 처리 인수 ---
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"병렬 처리 작업자 수 (기본값: CPU 코어 수 = {DEFAULT_WORKERS}). 0 또는 1이면 순차 처리.")

    # --- 기타 인수 ---
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로깅(DEBUG 레벨) 활성화")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    # 명령줄 인수 파싱
    args = parser.parse_args()

    # 상세 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("상세 로깅 활성화됨.")

    # --- 인수 유효성 검사 ---
    if args.min_face_width < 0:
        logging.warning(f"최소 얼굴 너비는 0 이상이어야 합니다 ({args.min_face_width}). {DEFAULT_MIN_FACE_WIDTH} 사용.")
        args.min_face_width = DEFAULT_MIN_FACE_WIDTH
    if args.min_face_height < 0:
        logging.warning(f"최소 얼굴 높이는 0 이상이어야 합니다 ({args.min_face_height}). {DEFAULT_MIN_FACE_HEIGHT} 사용.")
        args.min_face_height = DEFAULT_MIN_FACE_HEIGHT
    if args.padding_percent < 0:
        logging.warning(f"패딩 비율은 0 이상이어야 합니다 ({args.padding_percent}). 0 사용.")
        args.padding_percent = 0
    if args.workers < 0:
        logging.warning(f"작업자 수는 0 이상이어야 합니다 ({args.workers}). 기본값 {DEFAULT_WORKERS} 사용.")
        args.workers = DEFAULT_WORKERS

    # --- 준비 단계 ---
    # DNN 모델 파일 다운로드
    if not download_model(YUNET_MODEL_URL, YUNET_MODEL_PATH):
        logging.critical("DNN 모델 파일이 준비되지 않아 처리를 중단합니다.")
        return

    # 얼굴 감지기 로드
    try:
        logging.debug("얼굴 감지 모델 로딩 중...")
        # 모델 로드는 한 번만 수행
        global_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
        global_detector.setScoreThreshold(args.confidence)
        global_detector.setNMSThreshold(args.nms)
        logging.debug("얼굴 감지 모델 로딩 완료.")
    except cv2.error as e:
        logging.critical(f"얼굴 감지 모델 로드 실패: {e}")
        return
    except Exception as e:
        logging.critical(f"얼굴 감지 모델 로드 중 예상치 못한 오류: {e}")
        return

    # 출력 디렉토리 생성
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            logging.info(f"출력 디렉토리 생성: {args.output_dir}")
        except OSError as e:
            logging.critical(f"출력 디렉토리 생성 실패: {e}")
            return
    elif not os.path.isdir(args.output_dir):
         logging.critical(f"지정된 출력 경로 '{args.output_dir}'는 디렉토리가 아닙니다.")
         return

    # --- 이미지 처리 ---
    if os.path.isfile(args.input_path):
        # 단일 파일 처리
        logging.info(f"단일 파일 처리 시작: {args.input_path}")
        result = process_image(args.input_path, args.output_dir, global_detector, args)
        logging.info(f"단일 파일 처리 완료. 결과: {result['message']}")
    elif os.path.isdir(args.input_path):
        # 디렉토리 처리
        logging.info(f"디렉토리 처리 시작: {args.input_path}")
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        try:
            all_files = os.listdir(args.input_path)
            image_files = [os.path.join(args.input_path, f) for f in all_files if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(args.input_path, f))]
        except OSError as e:
            logging.critical(f"입력 디렉토리 접근 불가: {e}")
            return

        if not image_files:
            logging.info("처리할 이미지 파일이 디렉토리에 없습니다.")
            return

        logging.info(f"총 {len(image_files)}개의 이미지 파일 처리 시작 (작업자 수: {args.workers if args.workers > 0 else 1})...")
        total_start_time = time.time()
        results = []
        processed_count = 0
        success_count = 0
        total_saved_files = 0
        failed_files = []

        # 병렬 처리 또는 순차 처리 결정
        if args.workers > 1:
             # ThreadPoolExecutor 사용 (I/O 위주 작업에 적합)
             # ProcessPoolExecutor 사용 시 detector 객체 직렬화 문제 발생 가능성 있음
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # 각 작업에 필요한 인자들을 튜플 리스트로 생성
                tasks = [(img_path, args.output_dir, global_detector, args) for img_path in image_files]
                # map 함수를 사용하여 작업 제출 및 결과 수집 (tqdm 적용)
                try:
                    results_iterator = executor.map(process_image_wrapper, tasks)
                    if TQDM_AVAILABLE:
                         results = list(tqdm(results_iterator, total=len(tasks), desc="이미지 처리 중"))
                    else:
                         results = list(results_iterator)
                except Exception as e:
                     logging.critical(f"병렬 처리 중 심각한 오류 발생: {e}", exc_info=True)
                     # 부분 결과라도 처리하기 위해 계속 진행할 수 있음
                     # results = [] # 또는 오류 발생 시 즉시 중단
        else:
            # 순차 처리 (workers가 0 또는 1일 때)
            logging.info("순차 처리 모드로 실행합니다.")
            iterator = tqdm(image_files, desc="이미지 처리 중") if TQDM_AVAILABLE else image_files
            for img_path in iterator:
                 results.append(process_image_wrapper((img_path, args.output_dir, global_detector, args)))


        # 결과 집계 및 요약
        for result in results:
            processed_count += 1
            if result['success']:
                success_count += 1
                total_saved_files += result['saved_files']
            else:
                failed_files.append(f"{result['filename']} ({result['message']})")

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        logging.info("-" * 30)
        logging.info("          처리 결과 요약          ")
        logging.info("-" * 30)
        logging.info(f"총 처리 시도 파일 수: {processed_count} / {len(image_files)}")
        logging.info(f"성공적으로 처리된 파일 수: {success_count}")
        logging.info(f"총 저장된 크롭 이미지 수: {total_saved_files}")
        logging.info(f"실패 또는 부분 실패 파일 수: {len(failed_files)}")
        if failed_files:
            logging.info("실패 상세 정보:")
            for fail_info in failed_files:
                logging.info(f"  - {fail_info}")
        logging.info(f"총 소요 시간: {total_processing_time:.2f} 초")
        logging.info("-" * 30)

    else:
        logging.critical(f"입력 경로를 찾을 수 없거나 유효하지 않습니다: {args.input_path}")

if __name__ == "__main__":
    main()
