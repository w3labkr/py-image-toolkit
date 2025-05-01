# -*- coding: utf-8 -*-
"""
이미지 내 얼굴을 감지하고, 얼굴 랜드마크(눈 중심) 또는 바운딩 박스 중심을 기준으로
3분할 법칙 또는 황금 비율에 맞춰 자동으로 크롭하는 스크립트 (명령줄 인수 사용).
- 원본 EXIF 데이터 보존
- DNN 모델 사전 로딩으로 효율성 향상
- logging 모듈 사용
- 덮어쓰기, 출력 형식, JPEG 품질 제어 옵션 추가
- 최소 얼굴 크기, 크롭 패딩, 구도 규칙 선택 옵션 추가
- 병렬 처리 기능 추가 (ProcessPoolExecutor 사용)
- 오류 요약 보고 기능 추가
- 설정 파일 지원 (--config) 추가
- Dry Run 모드 (--dry-run) 추가
- 주요 옵션 단축 인수 추가 (예: -o, -m, -r)

DNN 모델(YuNet)을 사용하여 얼굴 및 랜드마크를 감지합니다.

필요한 라이브러리:
pip install opencv-python numpy Pillow tqdm

버전: 1.6.3 (ProcessPoolExecutor 적용 및 안정성 향상)
"""
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging
import time
import json # 설정 파일 처리를 위해 추가
import concurrent.futures # ProcessPoolExecutor 사용
# 'Exif' 임포트 제거
from PIL import Image, UnidentifiedImageError
from typing import Tuple, List, Optional, Dict, Any, Union

# tqdm import 시도
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        logging.info("tqdm 라이브러리가 설치되지 않아 진행 표시줄을 생략합니다. (pip install tqdm)")
        return iterable

__version__ = "1.6.3" # 버전 업데이트

# --- 기본 설정값 ---
DEFAULT_CONFIG = {
    "output_dir": "output_final",
    "method": "largest",
    "reference": "eye",
    "ratio": None,
    "confidence": 0.6,
    "nms": 0.3,
    "overwrite": True,
    "output_format": None,
    "jpeg_quality": 95,
    "min_face_width": 30,
    "min_face_height": 30,
    "padding_percent": 5.0,
    "rule": "both",
    "workers": os.cpu_count(),
    "verbose": False,
    "dry_run": False
}

# DNN 모델 정보
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx" # 모델 파일 경로 전역 변수

# 결과 파일명 접미사
OUTPUT_SUFFIX_THIRDS = '_thirds'
OUTPUT_SUFFIX_GOLDEN = '_golden'
# --- 설정값 끝 ---

# --- 로깅 설정 ---
# 기본 로깅 핸들러 설정 (메인 프로세스용)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
# 기존 핸들러 제거 (중복 로깅 방지)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO) # 기본 레벨 INFO

# --- 유틸리티 함수 ---
# (download_model, parse_aspect_ratio, load_config 함수는 이전 버전과 동일)
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
        # 메인 프로세스에서만 이 로그가 유용함
        if os.getpid() == main_process_pid:
             logging.info(f"모델 파일 '{os.path.basename(file_path)}'이(가) 이미 존재합니다.")
        return True

def parse_aspect_ratio(ratio_str: Optional[str]) -> Optional[float]:
    """문자열 형태의 비율(예: '16:9', '1.0', 'None')을 float으로 변환합니다."""
    if ratio_str is None or str(ratio_str).lower() == 'none':
        return None
    try:
        ratio_str = str(ratio_str)
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

def load_config(config_path: str) -> Dict[str, Any]:
    """JSON 설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logging.info(f"설정 파일 로드 완료: {config_path}")
            return config
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"설정 파일 파싱 오류 ({config_path}): {e}")
        return {}
    except Exception as e:
        logging.error(f"설정 파일 로드 중 오류 발생 ({config_path}): {e}")
        return {}

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
        detector.setInputSize((img_w, img_h))
        # 얼굴 감지 수행 - 여기가 오류 발생 지점일 수 있음
        faces = detector.detect(image)

        if faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                x, y, w, h = map(int, face_info[:4])
                if w < min_w or h < min_h:
                    logging.debug(f"얼굴 ID {idx} ({w}x{h})가 최소 크기({min_w}x{min_h})보다 작아 무시합니다.")
                    continue

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
                    else:
                        logging.debug(f"얼굴 ID {idx}의 눈 랜드마크가 유효하지 않아 BBox 중심을 사용합니다.")

                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        # OpenCV 관련 오류 로깅 강화
        logging.error(f"OpenCV 오류 발생 (얼굴 감지 중 - 이미지 크기: {img_w}x{img_h}): {e}")
        # 오류 발생 시 빈 리스트 반환
        return []
    except Exception as e:
        logging.error(f"DNN 얼굴 감지 중 예상치 못한 문제 발생: {e}")
        return []
    return detected_subjects


def select_main_subject(subjects: List[Dict[str, Any]], img_shape: Tuple[int, int],
                        method: str = 'largest', reference_point_type: str = 'eye') -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """감지된 피사체 목록에서 주 피사체를 선택하고 기준점을 반환합니다."""
    if not subjects:
        logging.debug("주 피사체 선택: 감지된 피사체가 없습니다.")
        return None

    img_h, img_w = img_shape
    best_subject = None

    try:
        if len(subjects) == 1:
            best_subject = subjects[0]
        elif method == 'largest':
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])
        elif method == 'center':
            img_center = (img_w / 2, img_h / 2)
            best_subject = min(subjects, key=lambda s: math.dist(s['bbox_center'], img_center))
        else:
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"주 피사체 선택됨 (방법: {method}, 기준점: {reference_point_type}). BBox: {best_subject['bbox']}")
        return best_subject['bbox'], ref_center

    except Exception as e:
        logging.error(f"주 피사체 선택 중 오류 발생: {e}")
        return None


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """구도 법칙(3분할 또는 황금비율)에 따른 교차점 목록을 반환합니다."""
    points = []
    if width <= 0 or height <= 0:
        logging.warning(f"유효하지 않은 크기({width}x{height})로 구도점 계산 불가.")
        return []

    try:
        if rule_type == 'thirds':
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else:
            logging.warning(f"알 수 없는 구도 규칙 '{rule_type}'. 이미지 중심 사용.")
            points = [(width / 2, height / 2)]
        return [(int(round(px)), int(round(py))) for px, py in points]
    except Exception as e:
        logging.error(f"구도점 계산 중 오류 발생 (규칙: {rule_type}): {e}")
        return []


def calculate_optimal_crop(img_shape: Tuple[int, int], subject_center: Tuple[int, int],
                           rule_points: List[Tuple[int, int]], target_aspect_ratio: Optional[float]) -> Optional[Tuple[int, int, int, int]]:
    """주어진 기준점, 구도점, 비율에 맞춰 최적의 크롭 영역(x1, y1, x2, y2)을 계산합니다."""
    height, width = img_shape
    if height <= 0 or width <= 0:
        logging.warning(f"이미지 높이({height}) 또는 너비({width})가 0 이하이므로 크롭할 수 없습니다.")
        return None
    if not rule_points:
        logging.warning("구도점이 제공되지 않아 크롭 계산을 건너<0xEB><0x84><0x8E>니다.")
        return None

    cx, cy = subject_center

    try:
        if height > 0:
            aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        else:
             logging.warning("이미지 높이가 0이어서 원본 비율을 계산할 수 없습니다. 크롭 불가.")
             return None

        if aspect_ratio <= 0:
            logging.warning(f"유효하지 않은 비율({aspect_ratio})로 계산 불가.")
            return None

        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point

        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)

        if max_w <= 0 or max_h <= 0:
             logging.debug("타겟 포인트가 이미지 경계에 있어 유효한 크롭 불가.")
             return None

        crop_h_from_w = max_w / aspect_ratio
        crop_w_from_h = max_h * aspect_ratio

        if crop_h_from_w <= max_h + 1e-6:
            final_w, final_h = max_w, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h

        x1 = target_x - final_w / 2
        y1 = target_y - final_h / 2
        x2 = x1 + final_w
        y2 = y1 + final_h

        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        if x1 >= x2 or y1 >= y2:
            logging.warning("계산된 크롭 영역의 크기가 0입니다.")
            return None

        logging.debug(f"계산된 크롭 영역: ({x1}, {y1}) - ({x2}, {y2})")
        return x1, y1, x2, y2

    except Exception as e:
        logging.error(f"최적 크롭 계산 중 오류 발생: {e}")
        return None


def apply_padding(crop_coords: Tuple[int, int, int, int], img_shape: Tuple[int, int], padding_percent: float) -> Tuple[int, int, int, int]:
    """계산된 크롭 영역에 패딩을 적용하고 이미지 경계 내로 조정합니다."""
    x1, y1, x2, y2 = crop_coords
    img_h, img_w = img_shape
    crop_w = x2 - x1
    crop_h = y2 - y1

    if padding_percent <= 0:
        return crop_coords

    pad_x = int(round(crop_w * padding_percent / 100 / 2))
    pad_y = int(round(crop_h * padding_percent / 100 / 2))

    new_x1 = x1 - pad_x
    new_y1 = y1 - pad_y
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    if new_x1 >= new_x2 or new_y1 >= new_y2:
        logging.warning("패딩 적용 후 크롭 영역 크기가 0이 되어 패딩을 적용하지 않습니다.")
        return crop_coords

    logging.debug(f"패딩 적용된 크롭 영역 ({padding_percent}%): ({new_x1}, {new_y1}) - ({new_x2}, {new_y2})")
    return new_x1, new_y1, new_x2, new_y2


# --- 메인 처리 함수 ---

def process_image(image_path: str, output_dir: str, detector: cv2.FaceDetectorYN, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 이미지를 처리하여 크롭된 결과를 저장하거나 Dry Run 모드에서 로그를 남깁니다.
    성공/실패 상태와 메시지를 포함하는 딕셔너리를 반환합니다.
    (Namespace 대신 Dict를 받도록 수정)
    """
    filename = os.path.basename(image_path)
    logging.debug(f"처리 시작: {filename}")
    start_time = time.time()
    is_dry_run = args_dict.get('dry_run', False)
    status = {'filename': filename, 'success': False, 'saved_files': 0, 'message': '', 'dry_run': is_dry_run}
    exif_data = None
    original_ext = os.path.splitext(image_path)[1].lower()

    try:
        with Image.open(image_path) as pil_img:
            try:
                exif_data = pil_img.info.get('exif')
                if exif_data:
                    logging.debug(f"{filename}: EXIF 데이터 발견됨.")
            except Exception as exif_err:
                logging.warning(f"{filename}: EXIF 데이터 처리 중 오류: {exif_err}. EXIF 없이 저장됩니다.")
                exif_data = None

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
    detected_faces = detect_faces_dnn(detector, img, args_dict['min_face_width'], args_dict['min_face_height'])
    if not detected_faces:
        status['message'] = "유효한 얼굴 감지 실패 (최소 크기 조건 포함)."
        logging.info(f"{filename}: {status['message']}")
        return status

    # 주 피사체 선택
    selection_result = select_main_subject(detected_faces, (img_h, img_w), args_dict['method'], args_dict['reference'])
    if not selection_result:
         status['message'] = "주 피사체 선택 실패."
         logging.warning(f"{filename}: {status['message']}")
         return status
    subj_bbox, ref_center = selection_result

    # 출력 파일명 및 확장자 설정
    base = os.path.splitext(filename)[0]
    target_ratio_str = str(args_dict['ratio']) if args_dict['ratio'] is not None else "Orig"
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"
    ref_str = f"_ref{args_dict['reference']}"
    output_format = args_dict['output_format'].lower() if args_dict['output_format'] else None
    if output_format:
        output_ext = f".{output_format.lstrip('.')}"
        if output_ext[1:] not in Image.registered_extensions():
             logging.warning(f"{filename}: 지원되지 않는 출력 형식 '{args_dict['output_format']}'. 원본 형식 '{original_ext}' 사용.")
             output_ext = original_ext
    else:
        output_ext = original_ext

    # 적용할 구도 규칙 결정
    rules_to_apply = []
    if args_dict['rule'] == 'thirds' or args_dict['rule'] == 'both':
        rules_to_apply.append(('thirds', OUTPUT_SUFFIX_THIRDS))
    if args_dict['rule'] == 'golden' or args_dict['rule'] == 'both':
        rules_to_apply.append(('golden', OUTPUT_SUFFIX_GOLDEN))

    if not rules_to_apply:
        status['message'] = f"적용할 구도 규칙이 선택되지 않았습니다 ('{args_dict['rule']}')."
        logging.warning(f"{filename}: {status['message']}")
        return status

    saved_count = 0
    crop_errors = []
    # 각 구도 규칙에 대해 크롭 및 저장/Dry Run
    for rule_name, suffix in rules_to_apply:
        rule_points = get_rule_points(img_w, img_h, rule_name)
        target_ratio_float = parse_aspect_ratio(args_dict['ratio'])
        crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, target_ratio_float)

        if crop_coords:
            padded_coords = apply_padding(crop_coords, (img_h, img_w), args_dict['padding_percent'])
            x1, y1, x2, y2 = padded_coords

            if x1 >= x2 or y1 >= y2:
                msg = f"'{rule_name}' 규칙: 최종 크롭 영역 크기가 0."
                logging.warning(f"{filename}: {msg}")
                crop_errors.append(msg)
                continue

            out_filename = f"{base}{suffix}{ratio_str}{ref_str}{output_ext}"
            out_path = os.path.join(output_dir, out_filename)

            if not args_dict['overwrite'] and os.path.exists(out_path) and not is_dry_run:
                logging.info(f"{filename}: 파일이 이미 존재하고 덮어쓰기 비활성화됨 - 건너<0xEB><0x84><0x8E>기: {out_filename}")
                continue

            if is_dry_run:
                logging.info(f"[DRY RUN] {filename}: '{out_filename}' 저장 예정 (규칙: {rule_name}, 영역: {x1},{y1}-{x2},{y2})")
                saved_count += 1
            else:
                try:
                    cropped_img_bgr = img[y1:y2, x1:x2]
                    if cropped_img_bgr.size == 0:
                         raise ValueError("크롭된 이미지 크기가 0입니다.")

                    cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                    pil_cropped_img = Image.fromarray(cropped_img_rgb)
                    save_options = {}
                    if exif_data and isinstance(exif_data, bytes):
                        save_options['exif'] = exif_data
                    if output_ext.lower() in ['.jpg', '.jpeg']:
                        save_options['quality'] = args_dict['jpeg_quality']
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
                    # 상세 로깅 시 스택 트레이스 포함 (args_dict 사용)
                    logging.error(f"{filename}: {msg}", exc_info=args_dict.get('verbose', False))
                    crop_errors.append(msg)
        else:
             msg = f"'{rule_name}' 규칙: 유효 크롭 영역 생성 실패."
             logging.warning(f"{filename}: {msg}")
             crop_errors.append(msg)

    end_time = time.time()
    processing_time = end_time - start_time

    if saved_count > 0:
        status['success'] = True
        status['saved_files'] = saved_count
        action_verb = "시뮬레이션 완료" if is_dry_run else "처리 완료"
        status['message'] = f"{action_verb} ({saved_count}개 파일 {'저장 예정' if is_dry_run else '저장'}, {processing_time:.2f}초 소요)."
        if crop_errors:
            status['message'] += f" 일부 오류 발생: {'; '.join(crop_errors)}"
        logging.info(f"{filename}: {status['message']}")
    elif detected_faces:
        status['message'] = f"얼굴은 감지되었으나 유효한 크롭/저장{' 시뮬레이션' if is_dry_run else ''} 실패 ({processing_time:.2f}초 소요). 오류: {'; '.join(crop_errors) if crop_errors else '크롭 영역 계산 실패'}"
        logging.info(f"{filename}: {status['message']}")

    return status


# --- 병렬 처리 래퍼 함수 ---
def process_image_wrapper(args_tuple):
    """ concurrent.futures.ProcessPoolExecutor를 위한 래퍼 함수 """
    # 각 프로세스에서 필요한 것들만 받도록 수정
    image_path, output_dir, model_path, args_dict = args_tuple
    # 각 프로세스에서 로거 재설정 (선택적이지만 권장)
    process_logger = logging.getLogger()
    if not process_logger.hasHandlers(): # 핸들러 중복 추가 방지
        process_handler = logging.StreamHandler()
        process_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(processName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        process_handler.setFormatter(process_formatter)
        process_logger.addHandler(process_handler)
        process_logger.setLevel(logging.DEBUG if args_dict.get('verbose') else logging.INFO)

    detector = None # 감지기 초기화
    try:
        # --- 각 프로세스에서 감지기 로드 ---
        logging.debug(f"프로세스 시작: 모델 로딩 중 ({os.path.basename(model_path)})...")
        detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0))
        detector.setScoreThreshold(args_dict['confidence'])
        detector.setNMSThreshold(args_dict['nms'])
        logging.debug("프로세스 시작: 모델 로딩 완료.")
        # --- 감지기 로드 끝 ---

        # 실제 이미지 처리 함수 호출 (detector 전달)
        return process_image(image_path, output_dir, detector, args_dict)

    except Exception as e:
        # process_image 내부에서 처리되지 않은 예외 또는 모델 로딩 실패 처리
        filename = os.path.basename(image_path)
        logging.error(f"{filename}: 처리 중 심각한 오류 발생 (래퍼): {e}", exc_info=True)
        return {'filename': filename, 'success': False, 'saved_files': 0, 'message': f"처리 중 심각한 오류 (래퍼): {e}", 'dry_run': args_dict.get('dry_run', False)}


# --- 명령줄 인터페이스 및 실행 ---
# 메인 프로세스 PID 저장 (로그 중복 방지용)
main_process_pid = os.getpid()

def main():
    # 명령줄 인수 파서 설정
    parser = argparse.ArgumentParser(
        description=f"{__doc__}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- 기본 인수 ---
    parser.add_argument("input_path", help="처리할 이미지 파일 또는 디렉토리 경로.")
    # --- 단축 인수 추가 ---
    parser.add_argument("-o", "--output_dir", help="결과 저장 디렉토리.")
    parser.add_argument("--config", help="옵션을 불러올 JSON 설정 파일 경로.")

    # --- 동작 제어 인수 ---
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, help="출력 파일 덮어쓰기 허용 여부.")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일 저장 없이 처리 과정 시뮬레이션.")
    parser.add_argument("-w", "--workers", type=int, help="병렬 처리 작업자 수 (0 또는 1이면 순차 처리).")

    # --- 얼굴 감지 및 선택 인수 ---
    parser.add_argument("-m", "--method", choices=['largest', 'center'], help="주 피사체 선택 방법.")
    parser.add_argument("--ref", "--reference", dest="reference", choices=['eye', 'box'], help="구도 기준점 타입.")
    parser.add_argument("-c", "--confidence", type=float, help="얼굴 감지 최소 신뢰도.")
    parser.add_argument("-n", "--nms", type=float, help="얼굴 감지 NMS 임계값.")
    parser.add_argument("--min-face-width", type=int, help="처리할 최소 얼굴 너비 (픽셀).")
    parser.add_argument("--min-face-height", type=int, help="처리할 최소 얼굴 높이 (픽셀).")

    # --- 크롭 및 구도 인수 ---
    parser.add_argument("-r", "--ratio", type=str, help="목표 크롭 비율 (예: '16:9', '1.0', 'None').")
    parser.add_argument("--rule", choices=['thirds', 'golden', 'both'], help="적용할 구도 규칙.")
    parser.add_argument("-p", "--padding-percent", type=float, help="크롭 영역 주변 패딩 비율 (%).")

    # --- 출력 형식 인수 ---
    parser.add_argument("--output-format", type=str, help="출력 이미지 형식 (예: 'jpg', 'png').")
    parser.add_argument("-q", "--jpeg-quality", type=int, choices=range(1, 101), metavar="[1-100]", help="JPEG 저장 품질 (1-100).")

    # --- 기타 인수 ---
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로깅(DEBUG 레벨) 활성화.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    # 1차 파싱
    args = parser.parse_args()

    # --- 설정 로드 및 병합 ---
    config = DEFAULT_CONFIG.copy()
    config_path = args.config
    if config_path:
        loaded_config = load_config(config_path)
        for key, value in loaded_config.items():
            if key in config:
                if key == 'ratio' and value is not None:
                    config[key] = str(value)
                elif key == 'padding_percent' and value is not None:
                    config[key] = float(value)
                else:
                    config[key] = value
            else:
                logging.warning(f"설정 파일의 알 수 없는 키 '{key}'는 무시됩니다.")

    # 명령줄 인수로 설정 덮어쓰기
    cmd_args = vars(args)
    for key, value in cmd_args.items():
        if value is not None:
            is_bool_action = isinstance(parser.get_default(key), bool) or \
                             any(action.dest == key and isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction))
                                 for action in parser._actions)
            if is_bool_action:
                 if isinstance(value, bool):
                      config[key] = value
                 elif isinstance(parser._get_action_from_dest(key), (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                      config[key] = value
            elif value != parser.get_default(key) or (config_path and value != loaded_config.get(key)):
                if key == 'ratio' and value is not None:
                     config[key] = str(value)
                elif key == 'padding_percent' and value is not None:
                     config[key] = float(value)
                elif key != 'config':
                     config[key] = value

    # 최종 설정을 Dictionary로 저장 (ProcessPoolExecutor에 전달하기 위함)
    final_args_dict = config

    # 로깅 레벨 설정 (메인 프로세스)
    if final_args_dict['verbose']:
        logger.setLevel(logging.DEBUG)
        logging.debug("상세 로깅 활성화됨.")
        logging.debug(f"최종 적용 설정: {final_args_dict}")
    else:
         logger.setLevel(logging.INFO)

    if final_args_dict['dry_run']:
        logging.info("***** Dry Run 모드로 실행 중입니다. 실제 파일은 저장되지 않습니다. *****")

    # --- 최종 인수 유효성 검사 ---
    if final_args_dict['min_face_width'] < 0:
        logging.warning(f"최소 얼굴 너비는 0 이상이어야 합니다 ({final_args_dict['min_face_width']}). 기본값 {DEFAULT_CONFIG['min_face_width']} 사용.")
        final_args_dict['min_face_width'] = DEFAULT_CONFIG['min_face_width']
    if final_args_dict['min_face_height'] < 0:
        logging.warning(f"최소 얼굴 높이는 0 이상이어야 합니다 ({final_args_dict['min_face_height']}). 기본값 {DEFAULT_CONFIG['min_face_height']} 사용.")
        final_args_dict['min_face_height'] = DEFAULT_CONFIG['min_face_height']
    if final_args_dict['padding_percent'] < 0:
        logging.warning(f"패딩 비율은 0 이상이어야 합니다 ({final_args_dict['padding_percent']}). 0 사용.")
        final_args_dict['padding_percent'] = 0.0
    if final_args_dict['workers'] < 0:
        logging.warning(f"작업자 수는 0 이상이어야 합니다 ({final_args_dict['workers']}). 기본값 {DEFAULT_CONFIG['workers']} 사용.")
        final_args_dict['workers'] = DEFAULT_CONFIG['workers']

    # --- 준비 단계 ---
    # DNN 모델 파일 다운로드 (메인 프로세스에서만)
    if not download_model(YUNET_MODEL_URL, YUNET_MODEL_PATH):
        logging.critical("DNN 모델 파일이 준비되지 않아 처리를 중단합니다.")
        return

    # 출력 디렉토리 생성 (Dry Run 모드가 아닐 때만)
    if not final_args_dict['dry_run']:
        output_dir = final_args_dict['output_dir']
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logging.info(f"출력 디렉토리 생성: {output_dir}")
            except OSError as e:
                logging.critical(f"출력 디렉토리 생성 실패: {e}")
                return
        elif not os.path.isdir(output_dir):
             logging.critical(f"지정된 출력 경로 '{output_dir}'는 디렉토리가 아닙니다.")
             return
    else:
         logging.info(f"[DRY RUN] 출력 디렉토리 확인/생성 건너<0xEB><0x84><0x8E>기: {final_args_dict['output_dir']}")

    # --- 이미지 처리 ---
    input_path = args.input_path
    if os.path.isfile(input_path):
        # 단일 파일 처리 (병렬 처리 불필요, 메인 프로세스에서 감지기 로드)
        logging.info(f"단일 파일 처리 시작: {input_path}")
        try:
            detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
            detector.setScoreThreshold(final_args_dict['confidence'])
            detector.setNMSThreshold(final_args_dict['nms'])
            # Namespace 객체로 변환하여 전달 (기존 함수 호환성)
            final_args_ns = argparse.Namespace(**final_args_dict)
            result = process_image(input_path, final_args_dict['output_dir'], detector, final_args_ns)
            logging.info(f"단일 파일 처리 완료. 결과: {result['message']}")
        except cv2.error as e:
             logging.critical(f"얼굴 감지 모델 로드 실패 (단일 파일 처리): {e}")
        except Exception as e:
             logging.critical(f"단일 파일 처리 중 오류 발생: {e}", exc_info=True)

    elif os.path.isdir(input_path):
        # 디렉토리 처리
        logging.info(f"디렉토리 처리 시작: {input_path}")
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        try:
            all_files = os.listdir(input_path)
            image_files = [os.path.join(input_path, f) for f in all_files if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(input_path, f))]
        except OSError as e:
            logging.critical(f"입력 디렉토리 접근 불가: {e}")
            return

        if not image_files:
            logging.info("처리할 이미지 파일이 디렉토리에 없습니다.")
            return

        num_workers = final_args_dict['workers'] if final_args_dict['workers'] > 0 else 1
        # ProcessPoolExecutor 사용 시 실제 프로세스 수 제한 가능성 고려
        actual_workers = min(num_workers, os.cpu_count()) if num_workers > 0 else 1
        logging.info(f"총 {len(image_files)}개의 이미지 파일 처리 시작 (요청 작업자: {num_workers}, 실제 최대 작업자: {actual_workers})...")
        total_start_time = time.time()
        results = []
        processed_count = 0
        success_count = 0
        total_saved_files = 0
        failed_files = []

        # 병렬 처리 또는 순차 처리
        if num_workers > 1:
             # ProcessPoolExecutor 사용
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                # 각 작업에 필요한 인자들을 튜플 리스트로 생성
                # detector 객체 대신 model_path와 설정(dict) 전달
                tasks = [(img_path, final_args_dict['output_dir'], YUNET_MODEL_PATH, final_args_dict) for img_path in image_files]
                try:
                    results_iterator = executor.map(process_image_wrapper, tasks)
                    if TQDM_AVAILABLE:
                         results = list(tqdm(results_iterator, total=len(tasks), desc="이미지 처리 중"))
                    else:
                         # tqdm 없을 시 진행 상황 로깅 (간단하게)
                         temp_results = []
                         for i, result in enumerate(results_iterator):
                              temp_results.append(result)
                              if (i + 1) % 50 == 0: # 50개마다 로그 출력
                                   logging.info(f"진행 상황: {i + 1}/{len(tasks)} 파일 처리 완료...")
                         results = temp_results
                except Exception as e:
                     logging.critical(f"병렬 처리 중 심각한 오류 발생: {e}", exc_info=True)
        else:
            # 순차 처리 (메인 프로세스에서 감지기 로드)
            logging.info("순차 처리 모드로 실행합니다.")
            try:
                detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
                detector.setScoreThreshold(final_args_dict['confidence'])
                detector.setNMSThreshold(final_args_dict['nms'])
                iterator = tqdm(image_files, desc="이미지 처리 중") if TQDM_AVAILABLE else image_files
                # Namespace 객체로 변환하여 전달
                final_args_ns = argparse.Namespace(**final_args_dict)
                for img_path in iterator:
                     results.append(process_image(img_path, final_args_dict['output_dir'], detector, final_args_ns))
            except cv2.error as e:
                 logging.critical(f"얼굴 감지 모델 로드 실패 (순차 처리): {e}")
                 return # 순차 처리 시 모델 로드 실패하면 중단
            except Exception as e:
                 logging.critical(f"순차 처리 중 오류 발생: {e}", exc_info=True)
                 return


        # 결과 집계 및 요약
        for result in results:
            processed_count += 1
            if result['success']:
                success_count += 1
                total_saved_files += result['saved_files']
            if not result['success']:
                 failed_files.append(f"{result['filename']} ({result['message']})")

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        action_verb = "시뮬레이션" if final_args_dict['dry_run'] else "처리"

        logging.info("-" * 30)
        logging.info(f"          디렉토리 {action_verb} 결과 요약          ")
        logging.info("-" * 30)
        logging.info(f"총 {action_verb} 시도 파일 수: {processed_count} / {len(image_files)}")
        logging.info(f"성공적으로 {action_verb}된 파일 수: {success_count}")
        logging.info(f"총 {'저장 예정' if final_args_dict['dry_run'] else '저장된'} 크롭 이미지 수: {total_saved_files}")
        logging.info(f"실패 또는 부분 실패 파일 수: {len(failed_files)}")
        if failed_files:
            logging.info("실패 상세 정보 (오류 로그 확인):")
            for i, fail_info in enumerate(failed_files):
                 if i < 10:
                    logging.info(f"  - {fail_info.split(' (')[0]}")
                 elif i == 10:
                    logging.info("  - ... (더 많은 실패 항목은 로그 확인)")
                    break
        logging.info(f"총 소요 시간: {total_processing_time:.2f} 초")
        logging.info("-" * 30)

    else:
        logging.critical(f"입력 경로를 찾을 수 없거나 유효하지 않습니다: {input_path}")

if __name__ == "__main__":
    # 메인 프로세스 PID 저장 (자식 프로세스 로깅 구분용)
    main_process_pid = os.getpid()
    main()
