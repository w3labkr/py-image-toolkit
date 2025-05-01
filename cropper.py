# -*- coding: utf-8 -*-
"""
이미지 내 얼굴을 감지하고, 얼굴 랜드마크(눈 중심) 또는 바운딩 박스 중심을 기준으로
3분할 법칙 또는 황금 비율에 맞춰 자동으로 크롭하는 스크립트 (명령줄 인수 사용).
- 원본 EXIF 데이터 보존
- DNN 모델 사전 로딩으로 효율성 향상
- logging 모듈 사용
- 덮어쓰기, 출력 형식, JPEG 품질 제어 옵션 추가

DNN 모델(YuNet)을 사용하여 얼굴 및 랜드마크를 감지합니다.

필요한 라이브러리:
pip install opencv-python numpy Pillow tqdm

버전: 1.3.0 (로깅, 효율성, 옵션 개선)
"""
import cv2
import numpy as np
import os
import math
import urllib.request
import argparse
import logging # logging 모듈 임포트
from PIL import Image, UnidentifiedImageError # PIL 라이브러리 및 관련 예외 임포트
from typing import Tuple, List, Optional, Dict, Any, Union

# tqdm import 시도 (없으면 기능 비활성화)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs): # tqdm이 없을 경우 대체 함수
        logging.info("tqdm 라이브러리가 설치되지 않아 진행 표시줄을 생략합니다. (pip install tqdm)")
        return iterable

__version__ = "1.3.0" # 버전 업데이트

# --- 기본 설정값 (명령줄 인수로 덮어쓸 수 있음) ---
DEFAULT_OUTPUT_DIR = "output_final"
DEFAULT_SELECTION_METHOD = 'largest' # 'largest' or 'center'
DEFAULT_REFERENCE_POINT = 'eye'     # 'eye' or 'box'
DEFAULT_ASPECT_RATIO = None        # 예: "16:9", "1.0", "4:3", None (원본 유지)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_OVERWRITE = True
DEFAULT_OUTPUT_FORMAT = None # None이면 원본 확장자 유지
DEFAULT_JPEG_QUALITY = 95

# DNN 얼굴 검출 모델 파일 경로 및 URL
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# 결과 파일명 접미사
OUTPUT_SUFFIX_THIRDS = '_thirds'
OUTPUT_SUFFIX_GOLDEN = '_golden'
# --- 설정값 끝 ---

# --- 로깅 설정 ---
# 기본 로깅 레벨 설정 (INFO 이상 출력)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 유틸리티 함수 ---

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

def detect_faces_dnn(detector: cv2.FaceDetectorYN, image: np.ndarray) -> List[Dict[str, Any]]:
    """사전 로드된 DNN 모델(YuNet)을 사용하여 얼굴 목록을 감지합니다."""
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
                # 눈 랜드마크 좌표
                r_eye_x, r_eye_y = face_info[4:6]
                l_eye_x, l_eye_y = face_info[6:8]
                # 신뢰도 점수
                confidence = face_info[14]

                # 경계 상자 좌표 보정 (이미지 경계 내)
                x = max(0, x); y = max(0, y)
                w = min(img_w - x, w); h = min(img_h - y, h)

                # 유효한 경계 상자인 경우 처리
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
                        # 눈 랜드마크가 유효하지 않으면 경고 출력 (디버그 레벨로 변경 가능)
                        logging.debug(f"얼굴 ID {idx}의 눈 랜드마크가 유효하지 않아 BBox 중심을 사용합니다.")

                    # 감지된 피사체 정보 추가
                    detected_subjects.append({
                        'bbox': (x, y, w, h),
                        'bbox_center': bbox_center,
                        'eye_center': eye_center,
                        'confidence': confidence
                    })
    except cv2.error as e:
        logging.error(f"OpenCV 오류 발생 (얼굴 감지 중): {e}")
    except Exception as e:
        # DNN 얼굴 감지 중 오류 발생 시 에러 메시지 출력
        logging.error(f"DNN 얼굴 감지 중 예상치 못한 문제 발생: {e}")
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
            # 기본값인 'largest' 사용 (경고는 main 함수에서 처리)
            best_subject = max(subjects, key=lambda s: s['bbox'][2] * s['bbox'][3])

        # 선택된 주 피사체의 기준점(눈 중심 또는 경계 상자 중심) 결정
        ref_center = best_subject['eye_center'] if reference_point_type == 'eye' else best_subject['bbox_center']
        logging.debug(f"주 피사체 선택됨 (방법: {method}, 기준점: {reference_point_type}).")
        # 주 피사체의 경계 상자와 기준점 반환
        return best_subject['bbox'], ref_center

    except Exception as e:
        logging.error(f"주 피사체 선택 중 오류 발생: {e}")
        return None


def get_rule_points(width: int, height: int, rule_type: str = 'thirds') -> List[Tuple[int, int]]:
    """구도 법칙(3분할 또는 황금비율)에 따른 교차점 목록을 반환합니다."""
    points = []
    if width <= 0 or height <= 0:
        logging.warning(f"유효하지 않은 크기({width}x{height})로 구도점 계산 불가.")
        return [] # 빈 리스트 반환

    try:
        if rule_type == 'thirds':
            points = [(w, h) for w in (width / 3, 2 * width / 3) for h in (height / 3, 2 * height / 3)]
        elif rule_type == 'golden':
            phi_inv = (math.sqrt(5) - 1) / 2 # 1 / phi
            lines_w = (width * (1 - phi_inv), width * phi_inv)
            lines_h = (height * (1 - phi_inv), height * phi_inv)
            points = [(w, h) for w in lines_w for h in lines_h]
        else:
            logging.warning(f"알 수 없는 구도 규칙 '{rule_type}'. 이미지 중심 사용.")
            points = [(width / 2, height / 2)]

        # 계산된 교차점 좌표를 정수로 반올림하여 반환
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

    cx, cy = subject_center # 주 피사체 기준점

    try:
        # 목표 비율이 없으면 원본 이미지 비율 사용
        # 원본 비율 계산 시 높이가 0인 경우 방지
        if height > 0:
            aspect_ratio = target_aspect_ratio if target_aspect_ratio is not None else (width / height)
        else: # 높이가 0이면 비율 계산 불가
             logging.warning("이미지 높이가 0이어서 원본 비율을 계산할 수 없습니다. 크롭 불가.")
             return None

        # 목표 비율 유효성 재확인 (parse_aspect_ratio에서 처리하지만 방어 코드)
        if aspect_ratio <= 0:
            logging.warning(f"유효하지 않은 비율({aspect_ratio})로 계산 불가.")
            return None

        # 기준점에서 가장 가까운 구도 교차점 찾기
        closest_point = min(rule_points, key=lambda p: math.dist((cx, cy), p))
        target_x, target_y = closest_point # 이 점이 크롭 영역의 중심이 되도록 함

        # 타겟 포인트를 중심으로 가질 수 있는 최대 크롭 너비/높이 계산
        # target_x, target_y가 이미지 경계에 있는 경우 max_w 또는 max_h가 0이 될 수 있음
        max_w = 2 * min(target_x, width - target_x)
        max_h = 2 * min(target_y, height - target_y)

        # 최대 너비/높이가 0 이하이면 유효한 크롭 불가
        if max_w <= 0 or max_h <= 0:
             logging.warning("타겟 포인트가 이미지 경계에 있어 유효한 크롭 불가.")
             return None

        # 목표 비율에 맞춰 크롭 크기 계산
        crop_h_from_w = max_w / aspect_ratio
        crop_w_from_h = max_h * aspect_ratio

        # 두 계산 결과 중 이미지 경계 내에 맞는 크기 선택
        if crop_h_from_w <= max_h + 1e-6: # 부동 소수점 오차 감안
            final_w, final_h = max_w, crop_h_from_w
        else:
            final_w, final_h = crop_w_from_h, max_h

        # 크롭 영역 좌상단(x1, y1), 우하단(x2, y2) 좌표 계산
        x1 = target_x - final_w / 2
        y1 = target_y - final_h / 2
        x2 = x1 + final_w
        y2 = y1 + final_h

        # 좌표를 정수로 변환하고 이미지 경계 내로 제한
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        # 최종 크롭 영역 크기가 유효하지 않은 경우 (너비 또는 높이가 0)
        if x1 >= x2 or y1 >= y2:
            logging.warning("계산된 크롭 영역의 크기가 0입니다.")
            return None

        # 최종 크롭 영역 좌표 반환
        return x1, y1, x2, y2

    except Exception as e:
        logging.error(f"최적 크롭 계산 중 오류 발생: {e}")
        return None

# --- 메인 처리 함수 ---

def process_image(image_path: str, output_dir: str, detector: cv2.FaceDetectorYN, args: argparse.Namespace):
    """단일 이미지를 처리하여 크롭된 결과를 저장합니다."""
    logging.debug(f"처리 시작: {os.path.basename(image_path)}")
    exif_data = None
    original_ext = os.path.splitext(image_path)[1].lower()

    try:
        # PIL로 이미지 열기 (EXIF 추출 및 로딩)
        with Image.open(image_path) as pil_img:
            # 원본 EXIF 데이터 가져오기
            exif_data = pil_img.info.get('exif')
            # RGB 모드로 변환 (OpenCV 호환)
            pil_img_rgb = pil_img.convert('RGB')
            # PIL 이미지를 OpenCV 형식(NumPy 배열, BGR)으로 변환
            img = np.array(pil_img_rgb)[:, :, ::-1].copy()

    except FileNotFoundError:
        logging.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    except UnidentifiedImageError:
        logging.error(f"이미지 파일을 열 수 없거나 지원하지 않는 형식입니다: {image_path}")
        return
    except Exception as e:
        logging.error(f"이미지 로드 중 오류 발생 ({image_path}): {e}")
        return

    img_h, img_w = img.shape[:2]
    if img_h <= 0 or img_w <= 0:
        logging.warning(f"유효하지 않은 이미지 크기 ({img_w}x{img_h}) - 건너<0xEB><0x84><0x8E>니다: {os.path.basename(image_path)}")
        return

    # DNN 얼굴 감지 (사전 로드된 detector 사용)
    detected_faces = detect_faces_dnn(detector, img)
    if not detected_faces:
        logging.info(f"얼굴 감지 실패: {os.path.basename(image_path)}")
        return

    # 주 피사체 선택
    selection_result = select_main_subject(detected_faces, (img_h, img_w), args.method, args.reference)
    if not selection_result:
         logging.warning(f"주 피사체 선택 실패: {os.path.basename(image_path)}")
         return
    subj_bbox, ref_center = selection_result

    # 출력 파일명 생성 준비
    base = os.path.splitext(os.path.basename(image_path))[0]
    target_ratio_str = args.ratio if args.ratio else "Orig" # 비율 문자열
    ratio_str = f"_r{target_ratio_str.replace(':', '-')}"
    ref_str = f"_ref{args.reference}" # 기준점 문자열

    # 출력 확장자 결정
    output_format = args.output_format.lower() if args.output_format else None
    if output_format:
        # 지정된 형식이 있으면 '.' 제거하고 사용 (예: 'jpg', 'png')
        output_ext = f".{output_format.lstrip('.')}"
        # Pillow가 지원하는 형식인지 간단히 확인 (더 엄격한 검사 필요시 추가)
        if output_ext[1:] not in Image.registered_extensions():
             logging.warning(f"지원되지 않거나 알 수 없는 출력 형식 '{args.output_format}'. 원본 형식 '{original_ext}' 사용.")
             output_ext = original_ext
    else:
        # 지정되지 않으면 원본 확장자 사용
        output_ext = original_ext

    # 처리할 구도 규칙 목록
    crop_configs = [('thirds', OUTPUT_SUFFIX_THIRDS), ('golden', OUTPUT_SUFFIX_GOLDEN)]
    saved_count = 0

    # 각 구도 규칙에 대해 크롭 및 저장
    for rule_name, suffix in crop_configs:
        # 구도 교차점 계산
        rule_points = get_rule_points(img_w, img_h, rule_name)
        # 목표 비율 파싱 (float 또는 None)
        target_ratio_float = parse_aspect_ratio(args.ratio)
        # 최적 크롭 영역 계산
        crop_coords = calculate_optimal_crop((img_h, img_w), ref_center, rule_points, target_ratio_float)

        # 유효한 크롭 영역이 계산된 경우
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            # OpenCV 이미지(BGR)에서 크롭 영역 추출
            cropped_img_bgr = img[y1:y2, x1:x2]

            # 출력 파일 경로 생성
            out_filename = f"{base}{suffix}{ratio_str}{ref_str}{output_ext}"
            out_path = os.path.join(output_dir, out_filename)

            # 덮어쓰기 옵션 확인
            if not args.overwrite and os.path.exists(out_path):
                logging.info(f"파일이 이미 존재하고 덮어쓰기 비활성화됨 - 건너<0xEB><0x84><0x8E>기: {out_filename}")
                continue

            try:
                # --- PIL을 사용하여 EXIF 보존하며 저장 ---
                # 크롭된 OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
                cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                pil_cropped_img = Image.fromarray(cropped_img_rgb)

                # 저장 옵션 설정
                save_options = {}
                if exif_data:
                    save_options['exif'] = exif_data

                # JPEG 품질 설정 (출력 형식이 JPEG 계열일 경우)
                if output_ext.lower() in ['.jpg', '.jpeg']:
                    save_options['quality'] = args.jpeg_quality
                    save_options['optimize'] = True # 약간의 최적화 시도
                    save_options['progressive'] = True # 프로그레시브 JPEG

                # 이미지 저장
                pil_cropped_img.save(out_path, **save_options)
                logging.debug(f"저장 완료: {out_filename}")
                # --- 저장 로직 끝 ---
                saved_count += 1

            except IOError as e:
                 logging.error(f"파일 쓰기 오류 발생 ({out_path}): {e}")
            except Exception as e:
                # 저장 중 오류 발생 시 에러 메시지 출력
                logging.error(f"크롭 이미지 저장 중 예상치 못한 오류 발생 ({out_path}): {e}")
        else:
             logging.warning(f"'{rule_name}' 규칙에 대한 유효 크롭 영역 생성 실패: {os.path.basename(image_path)}")

    if saved_count > 0:
         logging.info(f"처리 완료 ({saved_count}개 파일 저장): {os.path.basename(image_path)}")
    elif detected_faces: # 얼굴은 감지되었으나 저장은 못한 경우
         logging.info(f"얼굴은 감지되었으나 유효한 크롭/저장 실패: {os.path.basename(image_path)}")


# --- 명령줄 인터페이스 및 실행 ---

def main():
    # 명령줄 인수 파서 설정
    parser = argparse.ArgumentParser(
        description=f"{__doc__}", # Docstring 사용
        formatter_class=argparse.RawTextHelpFormatter # 도움말 형식 유지
    )
    # 입력 경로 인수 (필수)
    parser.add_argument("input_path", help="처리할 이미지 파일 또는 디렉토리 경로.")
    # 출력 디렉토리 인수
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"결과 저장 디렉토리 (기본값: {DEFAULT_OUTPUT_DIR})")
    # 주 피사체 선택 방법 인수
    parser.add_argument("-m", "--method", choices=['largest', 'center'], default=DEFAULT_SELECTION_METHOD, help=f"주 피사체 선택 방법 (기본값: {DEFAULT_SELECTION_METHOD})\n  largest: 가장 큰 얼굴\n  center: 이미지 중앙에 가장 가까운 얼굴")
    # 구도 기준점 타입 인수
    parser.add_argument("-ref", "--reference", choices=['eye', 'box'], default=DEFAULT_REFERENCE_POINT, help=f"구도 기준점 타입 (기본값: {DEFAULT_REFERENCE_POINT})\n  eye: 양 눈의 중심\n  box: 얼굴 경계 상자의 중심")
    # 목표 크롭 비율 인수
    parser.add_argument("-r", "--ratio", type=str, default=DEFAULT_ASPECT_RATIO, help="목표 크롭 비율 (예: '16:9', '1.0', '4:3', 'None').\n'None' 또는 미지정 시 원본 비율 유지 (기본값).")
    # 얼굴 감지 신뢰도 임계값 인수
    parser.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f"얼굴 감지 최소 신뢰도 (기본값: {DEFAULT_CONFIDENCE_THRESHOLD})")
    # 얼굴 감지 NMS 임계값 인수
    parser.add_argument("-n", "--nms", type=float, default=DEFAULT_NMS_THRESHOLD, help=f"얼굴 감지 NMS 임계값 (기본값: {DEFAULT_NMS_THRESHOLD})")
    # 덮어쓰기 제어 플래그
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=DEFAULT_OVERWRITE, help=f"출력 파일 덮어쓰기 허용 여부 (기본값: --overwrite / 허용)") # Python 3.9+
    # 출력 형식 지정 옵션
    parser.add_argument("--output-format", type=str, default=DEFAULT_OUTPUT_FORMAT, help="출력 이미지 형식 (예: 'jpg', 'png'). 미지정 시 원본 형식 유지.")
    # JPEG 품질 지정 옵션
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, choices=range(1, 101), metavar="[1-100]", help=f"JPEG 저장 품질 (1-100, 기본값: {DEFAULT_JPEG_QUALITY}). --output-format이 jpg/jpeg일 때 유효.")
    # 상세 로깅 활성화 플래그
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로깅(DEBUG 레벨) 활성화")
    # 버전 정보 표시 인수
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    # 명령줄 인수 파싱
    args = parser.parse_args()

    # 상세 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("상세 로깅 활성화됨.")

    # --- 유효성 검사 ---
    # 주 피사체 선택 방법 확인
    if args.method not in ['largest', 'center']:
        logging.warning(f"알 수 없는 주 피사체 선택 방법 '{args.method}'. 기본값 '{DEFAULT_SELECTION_METHOD}' 사용.")
        args.method = DEFAULT_SELECTION_METHOD
    # 구도 기준점 타입 확인
    if args.reference not in ['eye', 'box']:
        logging.warning(f"알 수 없는 구도 기준점 타입 '{args.reference}'. 기본값 '{DEFAULT_REFERENCE_POINT}' 사용.")
        args.reference = DEFAULT_REFERENCE_POINT
    # JPEG 품질 범위 확인 (argparse choices에서 처리하지만 추가 확인)
    if not (1 <= args.jpeg_quality <= 100):
         logging.warning(f"JPEG 품질은 1에서 100 사이여야 합니다 ({args.jpeg_quality}). 기본값 {DEFAULT_JPEG_QUALITY} 사용.")
         args.jpeg_quality = DEFAULT_JPEG_QUALITY

    # --- 준비 단계 ---
    # DNN 모델 파일 다운로드 시도
    if not download_model(YUNET_MODEL_URL, YUNET_MODEL_PATH):
        logging.critical("DNN 모델 파일이 준비되지 않아 처리를 중단합니다.")
        return

    # 얼굴 감지기 로드
    try:
        logging.debug("얼굴 감지 모델 로딩 중...")
        detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (0, 0))
        # 감지기 기본 설정 (명령줄 인수로 덮어쓸 수 있음)
        detector.setScoreThreshold(args.confidence)
        detector.setNMSThreshold(args.nms)
        logging.debug("얼굴 감지 모델 로딩 완료.")
    except cv2.error as e:
        logging.critical(f"얼굴 감지 모델 로드 실패: {e}")
        return
    except Exception as e:
        logging.critical(f"얼굴 감지 모델 로드 중 예상치 못한 오류: {e}")
        return

    # 출력 디렉토리 생성 시도
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
    # 입력 경로가 파일인지 디렉토리인지 확인하고 처리
    if os.path.isfile(args.input_path):
        # 단일 파일 처리
        logging.info(f"단일 파일 처리 시작: {args.input_path}")
        process_image(args.input_path, args.output_dir, detector, args)
        logging.info(f"단일 파일 처리 완료.")
    elif os.path.isdir(args.input_path):
        # 디렉토리 처리
        logging.info(f"디렉토리 처리 시작: {args.input_path}")
        # 지원하는 이미지 확장자 목록 (소문자)
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        # 디렉토리 내 이미지 파일 목록 생성
        try:
            all_files = os.listdir(args.input_path)
            image_files = [f for f in all_files if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(args.input_path, f))]
        except OSError as e:
            logging.critical(f"입력 디렉토리 접근 불가: {e}")
            return

        if not image_files:
            logging.info("처리할 이미지 파일이 디렉토리에 없습니다.")
            return

        logging.info(f"총 {len(image_files)}개의 이미지 파일 처리 시작...")
        # tqdm 사용하여 진행률 표시하며 파일 처리
        file_iterator = tqdm(image_files, desc="이미지 처리 중") if TQDM_AVAILABLE else image_files
        processed_count = 0
        for filename in file_iterator:
            file_path = os.path.join(args.input_path, filename)
            process_image(file_path, args.output_dir, detector, args)
            processed_count += 1

        logging.info(f"디렉토리 처리 완료 ({processed_count}개 파일 처리 시도)")
    else:
        # 입력 경로가 유효하지 않은 경우 에러 메시지 출력
        logging.critical(f"입력 경로를 찾을 수 없거나 유효하지 않습니다: {args.input_path}")

if __name__ == "__main__":
    # 스크립트 직접 실행 시 main 함수 호출
    main()
