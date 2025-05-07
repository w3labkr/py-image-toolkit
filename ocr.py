# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더/병렬 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트
import platform # 운영체제 감지를 위해 platform 모듈 임포트
from tqdm import tqdm # 진행률 표시를 위해 tqdm 임포트
import logging # 로깅 모듈 임포트
import multiprocessing # 병렬 처리를 위해 multiprocessing 임포트
import warnings # 경고 메시지 제어를 위해 warnings 모듈 임포트
import csv # CSV 파일 처리를 위해 csv 모듈 임포트
import re # 텍스트 라벨링을 위한 정규 표현식 모듈
import math # 좌표 거리 계산을 위해 math 모듈 임포트
from collections import defaultdict # 기본값을 가진 딕셔너리 사용

# 스크립트 시작 시점에 ccache 경고 필터 설정 (PaddleOCR 임포트 전에 적용되도록)
warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")

from paddleocr import PaddleOCR, draw_ocr

__version__ = "1.0.17" # 스크립트 버전 정보 (최종 요약 CSV 'ocr_text.csv' 생성 기능 추가)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """기본 로거를 설정합니다."""
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False
    else:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
            if isinstance(handler, logging.StreamHandler):
                 logger.propagate = False

# 작업자 프로세스 내에서 사용할 전역 변수
_worker_csv_lock = None
_worker_csv_file_path = None
_worker_log_level_env_var = 'OCR_LOG_LEVEL' # 기본값

def worker_initializer_func(lock, csv_path, log_level_env_name):
    """각 작업자 프로세스 시작 시 호출될 초기화 함수입니다."""
    global _worker_csv_lock, _worker_csv_file_path, _worker_log_level_env_var

    _worker_csv_lock = lock
    _worker_csv_file_path = csv_path
    _worker_log_level_env_var = log_level_env_name

    warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")
    log_level_str = os.environ.get(_worker_log_level_env_var, 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level)
    logger.debug(f"작업자 {os.getpid()}: 초기화 완료 (Lock, CSV 경로 설정, 로깅 레벨: {log_level_str}).")


def preprocess_image_for_ocr(image_path):
    """OCR 정확도 향상을 위해 이미지를 전처리합니다. (버전 0.9.6/0.9.7 기준)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 불러올 수 없습니다: {image_path}")
            return None

        # 1. 수평 보정 작업 추가
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Canny 엣지 감지를 사용하여 이미지의 엣지 검출
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        # 허프 변환을 사용하여 직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 수평에 가까운 선만 고려 (수직선은 제외)
                if abs(x2 - x1) > abs(y2 - y1):  # 수평에 가까운 선
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles.append(angle)

            if angles:
                # 중앙값을 사용하여 이상치의 영향 감소
                median_angle = np.median(angles)

                # 각도가 -45° ~ 45° 범위를 벗어나면 조정 (큰 회전 방지)
                if median_angle > 45:
                    median_angle -= 90
                elif median_angle < -45:
                    median_angle += 90

                # 회전 각도가 너무 작으면 회전하지 않음 (작은 기울기는 무시)
                if abs(median_angle) > 0.5:
                    logger.debug(f"이미지 회전 보정: {os.path.basename(image_path)}, 각도: {median_angle:.2f}°")
                    # 이미지 중심을 기준으로 회전 변환 행렬 계산
                    height, width = img.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    # 회전 변환 적용
                    img = cv2.warpAffine(img, rotation_matrix, (width, height),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    # 회전 후 다시 그레이스케일 변환
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    logger.debug(f"이미지 회전이 필요하지 않음: {os.path.basename(image_path)}, 감지된 각도: {median_angle:.2f}°")
        else:
            logger.debug(f"직선을 감지할 수 없어 회전 보정을 건너뜁니다: {os.path.basename(image_path)}")

        # 2. 기존 전처리 과정
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"전처리 완료 (수평 보정, CLAHE, Median Blur): {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"예외 발생 (전처리, 파일: {os.path.basename(image_path)}): {e}")
        return None


def extract_text_from_image_worker(ocr_engine_params, image_data, filename_for_log=""):
    """주어진 이미지 데이터에서 텍스트를 추출합니다. (작업자 프로세스용)"""
    ocr_engine = None
    try:
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 중... (파일: {filename_for_log}, 파라미터: {ocr_engine_params})")
        ocr_engine = PaddleOCR(**ocr_engine_params)
        logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 초기화 완료. (파일: {filename_for_log})")

        logger.debug(f"작업자 {os.getpid()}: OCR 시작: {filename_for_log}")
        result = ocr_engine.ocr(image_data, cls=True)
        logger.debug(f"작업자 {os.getpid()}: OCR 완료: {filename_for_log}")

        extracted_items = []
        if result and result[0] is not None:
            for line_info in result[0]:
                text, confidence = line_info[1]
                bounding_box = line_info[0]
                try:
                    float_box = [[float(p[0]), float(p[1])] for p in bounding_box]
                    extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": float_box
                    })
                except (ValueError, TypeError):
                     logger.warning(f"잘못된 바운딩 박스 데이터 형식: {bounding_box} (텍스트: {text})")
                     extracted_items.append({
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": None
                    })

        return extracted_items
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: OCR 처리 중 오류 (파일: {filename_for_log}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return None
    finally:
        if ocr_engine:
            del ocr_engine
            logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 삭제됨 (파일: {filename_for_log})")

# --- 바운딩 박스 유틸리티 함수 ---
def get_box_coordinates(box):
    if box and len(box) == 4 and all(len(pt) == 2 for pt in box):
        return box
    return None

def get_box_center(box):
    coords = get_box_coordinates(box)
    if coords:
        x_center = sum(pt[0] for pt in coords) / 4
        y_center = sum(pt[1] for pt in coords) / 4
        return x_center, y_center
    return None, None

def get_box_top(box):
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[1] for pt in coords)
    return float('inf')

def get_box_bottom(box):
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[1] for pt in coords)
    return float('-inf')

def get_box_left(box):
    coords = get_box_coordinates(box)
    if coords:
        return min(pt[0] for pt in coords)
    return float('inf')

def get_box_right(box):
    coords = get_box_coordinates(box)
    if coords:
        return max(pt[0] for pt in coords)
    return float('-inf')

def get_box_height(box):
    coords = get_box_coordinates(box)
    if coords:
        min_y = min(pt[1] for pt in coords)
        max_y = max(pt[1] for pt in coords)
        return max_y - min_y
    return 0

def get_box_width(box):
    coords = get_box_coordinates(box)
    if coords:
        min_x = min(pt[0] for pt in coords)
        max_x = max(pt[0] for pt in coords)
        return max_x - min_x
    return 0
# --- 바운딩 박스 유틸리티 함수 끝 ---

def reorder_korean_address(address_text):
    """한국식 주소 형식을 올바른 순서로 재정렬합니다."""
    # 특별시/광역시/도가 중간에 있는 경우 맨 앞으로 이동
    address_patterns = [
        # 도로명 주소 + 시/도가 뒤에 나온 경우 (법원로11길 서울특별시송파구 -> 서울특별시송파구 법원로11길)
        r'([가-힣0-9]+(?:로|길)[0-9]*)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)',
        # 건물번호가 시/도보다 앞에 오는 경우
        r'([0-9]+[A-Za-z]?(?:동|호|층|번지)?)\s+([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)[가-힣]*(?:시|군|구)?)'
    ]

    for pattern in address_patterns:
        match = re.search(pattern, address_text)
        if match:
            part1, part2 = match.groups()
            # 그룹2(시/도) + 그룹1(도로명/번지) 순서로 재정렬
            reordered = re.sub(pattern, r'\2 \1', address_text)
            logger.debug(f"주소 순서 재정렬: '{address_text}' -> '{reordered}'")
            address_text = reordered

    # 동/호수 사이 띄어쓰기 수정 (예: B 동610호 -> B동610호)
    address_text = re.sub(r'([0-9]+)\s+([A-Za-z])\s+동', r'\1\2동', address_text)
    address_text = re.sub(r'([A-Za-z])\s+동', r'\1동', address_text)

    return address_text

def label_text_item_initial(extracted_item, image_width, image_height):
    """추출된 텍스트 항목에 초기 라벨을 할당합니다."""
    text = extracted_item['text']
    box = extracted_item['bounding_box']
    x_center, y_center = get_box_center(box)

    if x_center is None or y_center is None:
        logger.warning(f"유효하지 않은 바운딩 박스로 초기 라벨링 시도: {box} (텍스트: {text})")
        return "기타"

    # 1. 주민등록번호
    if re.fullmatch(r"\d{6}\s*-\s*\d{7}", text) or \
       re.fullmatch(r"\d{13}", text.replace("-","").replace(" ","")):
        # 13자리 숫자인 경우 '-' 추가
        cleaned_text = text.replace("-","").replace(" ","")
        if len(cleaned_text) == 13 and cleaned_text.isdigit():
            text = f"{cleaned_text[:6]}-{cleaned_text[6:]}"
            extracted_item['text'] = text # 수정된 텍스트를 extracted_item에 반영
        return "주민등록번호"

    # 2. 이름
    cleaned_text_for_name = text.replace(" ","")
    if 2 <= len(cleaned_text_for_name) <= 5 and re.fullmatch(r"^[가-힣]+$", cleaned_text_for_name):
        if image_height > 0 and (image_height * 0.15 < y_center < image_height * 0.45) and len(cleaned_text_for_name) <= 4:
             return "이름"

    # 3. 날짜 후보 (YYYY.MM.DD 형식)
    date_match = re.search(r"(\d{4})\s*[\.,년]?\s*(\d{1,2})\s*[\.,월]?\s*(\d{1,2})\s*[\.일]?", text)
    if date_match:
        year, month, day = map(int, date_match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
             extracted_item['normalized_date'] = f"{year}.{month:02d}.{day:02d}"
             return "날짜_후보"

    # 3.1 날짜 부분 후보 (YYYY, MM, DD) - 숫자만 있을 때 더 유연하게 식별
    if text.isdigit():
        if len(text) == 4 and 1900 <= int(text) <= 2100: # 연도 - 유효한 연도 범위 확인
            extracted_item['date_part_type'] = 'year'
            return "날짜_부분"
        elif 1 <= len(text) <= 2: # MM 또는 DD
             month_day_val = int(text)
             if 1 <= month_day_val <= 31: # 월 또는 일 가능성
                 extracted_item['date_part_type'] = 'month_or_day'
                 return "날짜_부분"

    # 4. 발급기관 직인 후보
    authority_suffix = ["청장", "시장", "군수", "구청장", "경찰서장", "지방경찰청장"]
    if any(text.endswith(suffix) for suffix in authority_suffix):
        if image_height > 0 and y_center > image_height * 0.8:
            return "발급기관_직인"

    # 5. 발급기관 지역 후보
    region_keywords = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주"]
    region_suffixes = ["특별시", "광역시", "특별자치시", "도", "특별자치도"]
    is_region = False
    cleaned_text_for_region = text.replace(" ","")
    for keyword in region_keywords:
        if keyword in cleaned_text_for_region: is_region = True; break
    if not is_region:
         for suffix in region_suffixes:
             if cleaned_text_for_region.endswith(suffix): is_region = True; break

    if is_region:
        if image_height > 0 and image_width > 0 and (image_height * 0.7 < y_center < image_height * 0.95) and x_center < image_width * 0.7:
             return "발급기관_지역"

    # 6. 주소
    address_keywords = ["특별시", "광역시", "도", "시 ", "군 ", "구 ", "읍 ", "면 ", "동 ", "리 ", "로 ", "길 ", "아파트", "빌라", " 번지", " 호"]
    is_address = False
    for keyword in address_keywords:
        if keyword.strip() in text and len(text) > len(keyword.strip()): is_address = True; break
    if not is_address:
         pass

    if is_address:
        if image_height > 0 and (image_height * 0.3 < y_center < image_height * 0.8):
            return "주소"

    # 7. 문서명
    doc_titles = ["주민등록증", "운전면허증", "공무원증"]
    for title in doc_titles:
        if title in text and image_height > 0 and y_center < image_height * 0.25:
            return "문서명"

    return "기타"

def refine_and_finalize_labels(labeled_items):
    """초기 라벨링된 결과를 바탕으로 관계를 파악하여 최종 라벨을 결정하고 조합합니다."""
    final_labeled_items = []

    # 1단계: 후보 분류
    seal_items = [] # 직인 후보 여러 개 가능
    region_items = [] # 지역명 후보 여러 개 가능
    potential_dates = [] # "날짜_후보"
    potential_date_parts = [] # "날짜_부분"

    for item in labeled_items:
        # 아이템 복사하여 원본 리스트 수정 방지
        current_item = item.copy()
        if current_item['label'] == '발급기관_직인':
            seal_items.append(current_item)
        elif current_item['label'] == '발급기관_지역':
            region_items.append(current_item)
        elif current_item['label'] == '날짜_후보':
            potential_dates.append(current_item)
        elif current_item['label'] == '날짜_부분':
             potential_date_parts.append(current_item)

    # 추가: 발급기관_직인 항목이 없을 경우 텍스트 내용으로 후보 확인
    if not seal_items:
        for item in labeled_items:
            text = item['text'].replace(" ", "")
            if any(suffix in text for suffix in ["청장", "시장", "군수", "구청장", "경찰서장"]):
                seal_items.append(item.copy())

    # 추가: 발급기관_지역 항목이 없을 경우 텍스트 내용으로 후보 확인
    if not region_items:
        region_keywords = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충청", "전라", "경상", "제주"]
        region_suffixes = ["특별시", "광역시", "특별자치시", "도", "특별자치도"]

        for item in labeled_items:
            text = item['text'].replace(" ", "")
            is_region = False
            for keyword in region_keywords:
                if keyword in text:
                    is_region = True
                    break
            if not is_region:
                for suffix in region_suffixes:
                    if suffix in text:
                        is_region = True
                        break

            if is_region:
                region_items.append(item.copy())

    processed_indices = set()
    combined_authority_text = ""
    final_authority_item = None
    final_date_item = None # 최종 발급일자 항목

    # 2단계: 발급기관 조합
    if seal_items:
        primary_seal_item = seal_items[0]  # 첫번째 직인 후보 선택
        seal_x, seal_y = get_box_center(primary_seal_item['bounding_box'])

        # "구청장" 등 텍스트가 포함된 항목의 이미지 내 y좌표(높이)
        seal_bottom_y = get_box_bottom(primary_seal_item['bounding_box']) if primary_seal_item['bounding_box'] else 0

        # 지역명과 직인 조합
        best_region_item = None
        min_distance = float('inf')

        for region_candidate in region_items:
            region_x, region_y = get_box_center(region_candidate['bounding_box'])
            if region_candidate['bounding_box'] and primary_seal_item['bounding_box']:
                # 수직 거리와 수평 거리의 가중 합으로 거리 계산
                vertical_dist = abs(seal_y - region_y) if seal_y is not None and region_y is not None else float('inf')
                horizontal_dist = abs(seal_x - region_x) if seal_x is not None and region_x is not None else float('inf')

                # 수직 거리에 더 큰 가중치 부여 (같은 행에 있을 가능성이 높음)
                weighted_dist = vertical_dist * 2 + horizontal_dist

                if weighted_dist < min_distance:
                    min_distance = weighted_dist
                    best_region_item = region_candidate

        # 발급기관 최종 조합
        if best_region_item:
            # 지역명이 직인 앞에 오는지 확인
            region_left = get_box_left(best_region_item['bounding_box'])
            seal_left = get_box_left(primary_seal_item['bounding_box'])

            if region_left < seal_left:
                combined_authority_text = f"{best_region_item['text']} {primary_seal_item['text']}"
            else:
                combined_authority_text = f"{primary_seal_item['text']} {best_region_item['text']}"

            final_authority_item = primary_seal_item.copy()
            final_authority_item['label'] = "발급기관"
            final_authority_item['text'] = combined_authority_text

            # 사용된 항목 인덱스 추가
            try:
                processed_indices.add(labeled_items.index(primary_seal_item))
                processed_indices.add(labeled_items.index(best_region_item))
            except ValueError:
                logger.warning("발급기관 항목을 원본 리스트에서 찾을 수 없습니다.")

            logger.debug(f"발급기관 조합: {combined_authority_text}")
        else:
            # 지역명 없이 직인만 있는 경우
            final_authority_item = primary_seal_item.copy()
            final_authority_item['label'] = "발급기관"

            try:
                processed_indices.add(labeled_items.index(primary_seal_item))
            except ValueError:
                logger.warning(f"직인 항목을 원본 리스트에서 찾을 수 없습니다: {primary_seal_item}")

            logger.debug(f"발급기관 (직인만): {final_authority_item['text']}")

        # 3단계: 발급일자 확정
        if final_authority_item:
            authority_top_y = get_box_top(final_authority_item['bounding_box'])
            authority_left_x = get_box_left(final_authority_item['bounding_box'])
            authority_right_x = get_box_right(final_authority_item['bounding_box'])
            authority_height = get_box_height(final_authority_item['bounding_box'])

            # 3.1 "날짜_후보" 우선 탐색
            best_full_date_item = None
            min_vertical_gap_full = float('inf')
            for date_candidate in potential_dates:
                date_bottom_y = get_box_bottom(date_candidate['bounding_box'])
                date_left_x = get_box_left(date_candidate['bounding_box'])
                date_right_x = get_box_right(date_candidate['bounding_box'])

                # 조건 강화: 수직 간격 더 가깝게, 수평 겹침 더 확실하게
                if date_bottom_y < authority_top_y and \
                   (authority_top_y - date_bottom_y) < authority_height * 2.0 and \
                   max(authority_left_x, date_left_x) < min(authority_right_x, date_right_x):
                    vertical_gap = authority_top_y - date_bottom_y
                    if vertical_gap < min_vertical_gap_full:
                        min_vertical_gap_full = vertical_gap
                        best_full_date_item = date_candidate

            if best_full_date_item:
                best_full_date_item['label'] = "발급일자"
                best_full_date_item['text'] = best_full_date_item.get('normalized_date', best_full_date_item['text'])
                try:
                    processed_indices.add(labeled_items.index(best_full_date_item))
                except ValueError:
                     logger.warning(f"best_full_date_item을 labeled_items에서 찾을 수 없습니다: {best_full_date_item}")
                final_date_item = best_full_date_item
                logger.debug(f"발급일자 확정 (날짜_후보): {final_date_item['text']}")

            # 3.2 "날짜_후보"가 없으면 숫자 항목들을 발급일자로 조합 시도
            if not final_date_item:
                # 모든 숫자 텍스트 항목 수집 (위치 제한 완화)
                year_candidates = []
                month_candidates = []
                day_candidates = []

                # 발급기관 위에 있거나 가까이 있는 모든 숫자 텍스트 항목 수집
                for item in labeled_items:
                    if not item['bounding_box'] or not item['text'].isdigit():
                        continue

                    item_y = get_box_center(item['bounding_box'])[1]
                    authority_y = get_box_center(final_authority_item['bounding_box'])[1]

                    # 발급기관 위에 있는 숫자 항목만 수집
                    if item_y < authority_y:
                        value = int(item['text'])
                        if len(item['text']) == 4 and 1900 <= value <= 2100:  # 연도 후보
                            year_candidates.append({
                                'item': item,
                                'value': value,
                                'y': item_y,
                                'x': get_box_center(item['bounding_box'])[0]
                            })
                        elif 1 <= value <= 12 and len(item['text']) <= 2:  # 월 후보
                            month_candidates.append({
                                'item': item,
                                'value': value,
                                'y': item_y,
                                'x': get_box_center(item['bounding_box'])[0]
                            })
                        elif 1 <= value <= 31 and len(item['text']) <= 2:  # 일 후보
                            day_candidates.append({
                                'item': item,
                                'value': value,
                                'y': item_y,
                                'x': get_box_center(item['bounding_box'])[0]
                            })

                # 발급기관에 가장 가까운 날짜 항목 선택
                if year_candidates and month_candidates and day_candidates:
                    # 발급기관과의 거리가 가장 가까운 날짜 항목 선택
                    min_authority_distance = float('inf')
                    selected_date_items = None

                    for year in year_candidates:
                        for month in month_candidates:
                            for day in day_candidates:
                                # 3개 항목의 Y좌표 평균과 발급기관 Y좌표의 차이
                                avg_y = (year['y'] + month['y'] + day['y']) / 3
                                distance_to_authority = abs(avg_y - authority_y)

                                # 거리가 더 가까운 조합 선택
                                if distance_to_authority < min_authority_distance:
                                    min_authority_distance = distance_to_authority
                                    selected_date_items = (year, month, day)

                    if selected_date_items:
                        year_item, month_item, day_item = selected_date_items
                        combined_date_text = f"{year_item['value']}.{month_item['value']:02d}.{day_item['value']:02d}"

                        # 결과 항목 생성
                        final_date_item = {
                            'text': combined_date_text,
                            'label': '발급일자',
                            'confidence': min(
                                year_item['item']['confidence'],
                                month_item['item']['confidence'],
                                day_item['item']['confidence']
                            ),
                            'bounding_box': None
                        }

                        # 바운딩 박스 생성 - 연도, 월, 일 항목의 포괄적인 바운딩 박스 계산
                        if (year_item['item']['bounding_box'] and
                            month_item['item']['bounding_box'] and
                            day_item['item']['bounding_box']):

                            # 모든 바운딩 박스 좌표 추출
                            all_coords = []
                            all_coords.extend(year_item['item']['bounding_box'])
                            all_coords.extend(month_item['item']['bounding_box'])
                            all_coords.extend(day_item['item']['bounding_box'])

                            # 전체 영역을 감싸는 바운딩 박스 계산
                            min_x = min(coord[0] for coord in all_coords)
                            min_y = min(coord[1] for coord in all_coords)
                            max_x = max(coord[0] for coord in all_coords)
                            max_y = max(coord[1] for coord in all_coords)

                            # 새 바운딩 박스 생성
                            final_date_item['bounding_box'] = [
                                [min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y]
                            ]

                            logger.debug(f"발급일자 조합 바운딩 박스 생성: {final_date_item['bounding_box']}")

                        # 처리된 항목 인덱스 추가
                        try:
                            processed_indices.add(labeled_items.index(year_item['item']))
                            processed_indices.add(labeled_items.index(month_item['item']))
                            processed_indices.add(labeled_items.index(day_item['item']))
                            logger.debug(f"발급일자 확정 (모든 조합 시도): {combined_date_text}")
                        except ValueError:
                            logger.warning("조합된 날짜 부분 항목을 원본 리스트에서 찾을 수 없습니다.")
                            final_date_item = None

            # 3.5단계: 주소 항목 조합
            address_items = []
            for item in labeled_items:
                if item['label'] == '주소' and item not in [item_val for idx, item_val in enumerate(labeled_items) if idx in processed_indices]:
                    # 단일 주소 항목도 순서 재정렬 적용
                    item['text'] = reorder_korean_address(item['text'])
                    address_items.append(item)

            # 주소 항목이 여러 개면 좌상단에서 우하단 순으로 정렬하여 조합
            if len(address_items) > 1:
                # Y축 기준으로 그룹화(같은 줄에 있는지 확인)
                y_groups = {}
                for addr_item in address_items:
                    if addr_item['bounding_box']:
                        y_center = get_box_center(addr_item['bounding_box'])[1]
                        # 같은 높이(Y) 근처에 있는 항목들을 같은 그룹으로 취급
                        group_key = int(y_center / 20) * 20  # 20픽셀 간격으로 그룹화
                        if group_key not in y_groups:
                            y_groups[group_key] = []
                        y_groups[group_key].append(addr_item)

                # 각 그룹 내에서 X좌표로 정렬하고 Y좌표 그룹은 오름차순 정렬
                sorted_groups = sorted(y_groups.keys())
                combined_address_text = ""

                for group_key in sorted_groups:
                    # 같은 행에서는 X좌표 기준 왼쪽에서 오른쪽으로 정렬
                    sorted_row_items = sorted(y_groups[group_key],
                                            key=lambda item: get_box_left(item['bounding_box'])
                                                            if item['bounding_box'] else float('inf'))

                    for item in sorted_row_items:
                        combined_address_text += item['text'] + " "
                        try:
                            processed_indices.add(labeled_items.index(item))
                        except ValueError:
                            logger.warning(f"주소 항목을 원본 리스트에서 찾을 수 없습니다: {item['text']}")

                combined_address_text = combined_address_text.strip()

                # 주소 순서 재정렬 로직 추가
                if combined_address_text:
                    combined_address_text = reorder_korean_address(combined_address_text)

                    # 모든 주소 바운딩 박스를 포함하는 하나의 바운딩 박스 생성
                    all_coords = []
                    for addr_item in address_items:
                        if addr_item['bounding_box']:
                            all_coords.extend(addr_item['bounding_box'])

                    final_address_bounding_box = None
                    if all_coords:
                        min_x = min(coord[0] for coord in all_coords)
                        min_y = min(coord[1] for coord in all_coords)
                        max_x = max(coord[0] for coord in all_coords)
                        max_y = max(coord[1] for coord in all_coords)

                        final_address_bounding_box = [
                            [min_x, min_y],
                            [max_x, min_y],
                            [max_x, max_y],
                            [min_x, max_y]
                        ]

                    # 결합된 주소 항목 생성
                    final_address_item = {
                        'text': combined_address_text,
                        'label': '주소',
                        'confidence': min(item['confidence'] for item in address_items),
                        'bounding_box': final_address_bounding_box
                    }

                    logger.debug(f"주소 항목 조합: {combined_address_text}")
                    final_labeled_items.append(final_address_item)
            elif len(address_items) == 1: # 주소 항목이 하나만 있는 경우
                final_labeled_items.append(address_items[0])
                try:
                    processed_indices.add(labeled_items.index(address_items[0]))
                except ValueError:
                    logger.warning(f"단일 주소 항목을 원본 리스트에서 찾을 수 없습니다: {address_items[0]['text']}")


    # 4단계: 최종 결과 리스트 생성
    for i, item in enumerate(labeled_items):
        if i not in processed_indices:
            # 초기 라벨이 후보군이었으나 최종 선택되지 않은 경우 '기타'로 변경
            if item['label'] in ['발급기관_지역', '발급기관_직인', '날짜_후보', '날짜_부분']:
                item['label'] = '기타'
            final_labeled_items.append(item)

    # 최종 확정된 발급기관 항목 추가
    if final_authority_item:
        final_labeled_items.append(final_authority_item)

    # 조합된 발급일자가 있다면 추가
    if final_date_item:
        final_labeled_items.append(final_date_item)

    return final_labeled_items


def get_system_font_path():
    """운영 체제에 맞는 기본 시스템 폰트 경로를 반환합니다 (존재 여부 확인 포함)."""
    system = platform.system()
    font_path = None
    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif system == "Darwin":
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    elif system == "Linux":
        common_linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", # 나눔고딕 (일반적)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # DejaVu (많은 배포판 기본)
            "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
            "/usr/share/fonts/truetype/unfonts-core/UnBatang.ttf",
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break

    if font_path and os.path.exists(font_path):
        logger.debug(f"OS ({system}) 시스템 폰트 확인: {font_path}")
        return font_path
    logger.debug(f"OS ({system}) 시스템 폰트를 찾지 못했습니다 (경로: {font_path}).")
    return None

def determine_font_for_visualization():
    """시각화에 사용할 폰트 경로를 결정하고 관련 정보를 로깅합니다."""
    font_path_to_use = get_system_font_path()

    if font_path_to_use:
        logger.info(f"텍스트 시각화를 위해 시스템 자동 감지 폰트 사용: {font_path_to_use}")
        return font_path_to_use

    logger.info("시스템 폰트를 찾지 못하여 로컬 폰트를 확인합니다.")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logger.debug(f"__file__ 변수를 찾을 수 없어 현재 작업 디렉토리 사용: {script_dir}")

    local_korean_font = os.path.join(script_dir, 'fonts', 'malgun.ttf') # 예시 로컬 폰트
    if os.path.exists(local_korean_font):
        font_path_to_use = local_korean_font
        logger.info(f"로컬 한국어 폰트 사용: {font_path_to_use}")
        return font_path_to_use

    logger.info(f"로컬 한국어 폰트 '{local_korean_font}'를 찾지 못했습니다.")
    # 추가적인 로컬 폰트 경로 탐색 또는 PaddleOCR 기본 폰트 사용 유도
    logger.warning("사용 가능한 특정 폰트를 찾지 못했습니다. 시각화 시 PaddleOCR 내부 기본 폰트가 사용될 수 있습니다 (한글 깨짐 가능성).")
    return None # PaddleOCR 기본 폰트 사용


def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename,
                       preprocessed_img=None, show_image_flag=False, font_path_to_use=None):
    """OCR 결과를 이미지 위에 표시하고 저장합니다."""
    if not extracted_data:
        logger.info(f"{original_filename}: 시각화할 OCR 결과가 없습니다.")
        return

    try:
        if preprocessed_img is not None:
            # 전처리된 이미지가 그레이스케일이면 RGB로 변환
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB') # 원본 이미지를 RGB로 로드

        # draw_ocr에 필요한 형식으로 데이터 변환
        # boxes: [[좌상x, 좌상y], [우상x, 우상y], [우하x, 우하y], [좌하x, 좌하y]], ...
        # txts: [text1, text2, ...]
        # scores: [score1, score2, ...]
        boxes = [item['bounding_box'] for item in extracted_data if item.get('bounding_box')]
        txts = [item['text'] for item in extracted_data if item.get('bounding_box')] # 텍스트가 있는 항목만
        scores = [item['confidence'] for item in extracted_data if item.get('bounding_box')] # 신뢰도

        if not boxes: # 그릴 박스가 없으면 종료
            logger.info(f"{original_filename}: 시각화할 바운딩 박스가 없습니다.")
            return

        # draw_ocr 호출
        im_show = draw_ocr(np.array(image), boxes, txts, scores, font_path=font_path_to_use)
        im_show_pil = Image.fromarray(im_show) # PIL 이미지로 다시 변환

        # 결과 이미지 저장
        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr_result{ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)

        im_show_pil.save(output_image_path)
        logger.debug(f"OCR 결과 시각화 이미지가 {output_image_path}에 저장되었습니다.")

        if show_image_flag:
            logger.info(f"{original_filename} 결과 이미지를 표시합니다. (창을 닫아야 다음 작업 진행 가능성 있음)")
            im_show_pil.show()

    except FileNotFoundError:
        logger.error(f"원본 이미지 파일 '{original_image_path}'을 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        logger.error(f"OCR 결과 표시/저장 중 예외 발생 (파일: {original_filename}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))


# 병렬 처리를 위한 작업자 함수
def process_single_image_task(task_args_tuple):
    """단일 이미지 처리 작업을 수행하는 함수 (multiprocessing.Pool의 작업자용)."""
    global _worker_csv_lock, _worker_csv_file_path

    (current_image_path, filename, ocr_engine_params, output_dir,
     skip_preprocessing, show_image, font_path) = task_args_tuple

    status_message = f"{filename} 처리 중 오류 발생"
    processed_ok = False

    try:
        logger.debug(f"작업자 {os.getpid()}: 처리 시작: {filename}")

        ocr_input_data = current_image_path
        processed_image_for_display = None
        image_width, image_height = 0, 0

        try:
            with Image.open(current_image_path) as img_pil:
                image_width, image_height = img_pil.size
        except Exception as e:
            logger.error(f"작업자 {os.getpid()}: 이미지 크기 읽기 오류 ({filename}): {e}")
            # 이미지 크기를 못 읽어도 처리는 계속 시도 (label_text_item_initial에서 0으로 처리)

        if not skip_preprocessing:
            logger.debug(f"작업자 {os.getpid()}: 외부 전처리 시작: {filename}")
            processed_img_data = preprocess_image_for_ocr(current_image_path)
            if processed_img_data is not None:
                ocr_input_data = processed_img_data
                processed_image_for_display = processed_img_data # 시각화용으로 저장
            else:
                logger.warning(f"작업자 {os.getpid()}: {filename}: 외부 전처리에 실패하여 원본 이미지로 OCR을 시도합니다.")

        # 1단계: 초기 텍스트 추출
        extracted_items = extract_text_from_image_worker(ocr_engine_params, ocr_input_data, filename_for_log=filename)

        if extracted_items:
            logger.debug(f"작업자 {os.getpid()}: 추출된 텍스트 항목 수 ({filename}): {len(extracted_items)}")

            # 2단계: 초기 라벨링
            initially_labeled_items = []
            for item in extracted_items:
                initial_label = "기타" # 기본값
                if image_width > 0 and image_height > 0: # 이미지 크기를 성공적으로 읽었을 때만 라벨링 시도
                    initial_label = label_text_item_initial(item, image_width, image_height)
                item['label'] = initial_label
                initially_labeled_items.append(item)
                if logger.isEnabledFor(logging.DEBUG):
                     logger.debug(f"  - 초기 라벨: {initial_label}, 텍스트: \"{item['text']}\", 위치: {item.get('bounding_box')}")


            # 3단계: 최종 라벨 결정 및 조합
            final_labeled_items = refine_and_finalize_labels(initially_labeled_items)
            logger.debug(f"작업자 {os.getpid()}: 최종 라벨링된 항목 수 ({filename}): {len(final_labeled_items)}")
            if logger.isEnabledFor(logging.DEBUG):
                for fi in final_labeled_items:
                    logger.debug(f"  - 최종 라벨: {fi['label']}, 텍스트: \"{fi['text']}\"")


            # 4단계: CSV 저장 (항상 수행)
            csv_rows_to_write = []
            for item in final_labeled_items:
                # CSV에는 최종 라벨과 관련 정보만 기록
                csv_rows_to_write.append([
                    filename,
                    item.get('label', '기타'),
                    item.get('text', '').replace('"', '""'), # 큰따옴표 이스케이프
                    round(item.get('confidence', 0.0), 4),
                    str(item.get('bounding_box', 'N/A')) # 바운딩 박스는 문자열로
                ])

            if csv_rows_to_write and _worker_csv_lock and _worker_csv_file_path:
                with _worker_csv_lock:
                    try:
                        with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile: # utf-8-sig for Excel
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerows(csv_rows_to_write)
                        logger.info(f"작업자 {os.getpid()}: {filename}의 텍스트를 {os.path.basename(_worker_csv_file_path)}에 추가했습니다.")
                    except IOError as e:
                        logger.error(f"작업자 {os.getpid()}: CSV 파일 쓰기 중 오류 ({_worker_csv_file_path}, 파일: {filename}): {e}")
            elif not csv_rows_to_write:
                 logger.info(f"작업자 {os.getpid()}: {filename}에서 CSV에 쓸 최종 라벨링된 항목이 없습니다.")


            # 5단계: 시각화 (최종 라벨링된 결과가 아닌, 초기 추출된 모든 텍스트를 시각화)
            # 시각화는 원본 OCR 결과를 보여주는 것이 더 유용할 수 있음 (모든 추출된 텍스트 확인)
            # 또는 final_labeled_items를 사용하여 최종 라벨과 함께 표시할 수도 있음
            display_ocr_result(current_image_path, extracted_items, output_dir, filename, # extracted_items 사용
                               preprocessed_img=processed_image_for_display,
                               show_image_flag=show_image,
                               font_path_to_use=font_path)
            status_message = f"{filename} 처리 성공"
            processed_ok = True
        else:
            logger.info(f"작업자 {os.getpid()}: {filename}에서 텍스트를 추출하지 못했습니다.")
            status_message = f"{filename} 텍스트 없음"
            if _worker_csv_lock and _worker_csv_file_path: # 텍스트가 없어도 파일명은 기록
                 with _worker_csv_lock:
                     try:
                         with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                             csv_writer = csv.writer(csvfile)
                             csv_writer.writerow([filename, "N/A", "No text extracted", 0.0, "N/A"])
                     except IOError as e:
                         logger.error(f"작업자 {os.getpid()}: CSV 파일 쓰기 중 오류 ({_worker_csv_file_path}, 파일: {filename}): {e}")


        return status_message # 처리 상태 메시지 반환
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: process_single_image_task 내에서 예상치 못한 오류 발생 (파일: {filename}): {e}", exc_info=True)
        # 오류 발생 시에도 CSV에 기록 시도
        if _worker_csv_lock and _worker_csv_file_path:
             with _worker_csv_lock:
                 try:
                     with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                         csv_writer = csv.writer(csvfile)
                         csv_writer.writerow([filename, "오류", str(e), 0.0, "N/A"])
                 except IOError as io_err:
                     logger.error(f"작업자 {os.getpid()}: CSV 파일 쓰기 중 오류 ({_worker_csv_file_path}, 파일: {filename}): {io_err}")
        return status_message # 오류 메시지 반환


def parse_arguments():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default='input',
                        help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다.\n(기본값: 'input')")
    parser.add_argument("--output_dir", default='output',
                        help="OCR 결과 이미지 및 텍스트 파일을 저장할 폴더 경로입니다.\n(기본값: 'output')")
    parser.add_argument("--lang", default='korean',
                        help="OCR에 사용할 언어입니다. 예: 'korean', 'en', 'japan', 'ch_sim'.\n(기본값: 'korean')")
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', help="디버그 레벨 로깅을 활성화하여 더 상세한 로그를 출력합니다.")
    parser.add_argument('--use_gpu', action='store_true', help="사용 가능한 경우 GPU를 사용하여 OCR 처리를 시도합니다.\n(NVIDIA GPU 및 CUDA 환경 필요)")
    parser.add_argument("--num_workers", type=int, default=None, help="병렬 처리 시 사용할 작업자 수 (기본값: 시스템 CPU 코어 수)")

    return parser.parse_args()

def prepare_directories(input_dir, output_dir):
    """입력 및 출력 디렉토리를 확인하고, 출력 디렉토리가 없으면 생성합니다."""
    if not os.path.isdir(input_dir):
        logger.error(f"입력 디렉토리 '{input_dir}'가 없거나 디렉토리가 아닙니다. 종료합니다.")
        exit(1)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"출력 디렉토리 생성: {output_dir}")
        except OSError as e:
            logger.error(f"출력 디렉토리 '{output_dir}' 생성 실패: {e}")
            exit(1)

def create_final_summary_csv(labeled_csv_path, output_csv_path):
    """
    ocr_labeled_text.csv 파일을 읽어 최종 요약 CSV 파일 (ocr_text.csv)을 생성합니다.
    열 순서: Image Filename, 문서명, 이름, 주민등록번호, 주소, 발급일자, 발급기관, 기타
    """
    logger.info(f"'{labeled_csv_path}' 파일을 읽어 최종 요약 CSV ('{os.path.basename(output_csv_path)}') 생성을 시작합니다...")

    file_data_map = defaultdict(lambda: {
        "문서명": "", "이름": "", "주민등록번호": "",
        "주소": "", "발급일자": "", "발급기관": "", "기타": []
    })

    try:
        with open(labeled_csv_path, 'r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            header = next(reader) # 헤더 읽기
            # CSV 헤더: ["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"]
            try:
                filename_idx = header.index('Image Filename')
                label_idx = header.index('Label')
                text_idx = header.index('Extracted Text')
            except ValueError:
                logger.error(f"'{labeled_csv_path}' 파일의 헤더가 올바르지 않습니다. 'Image Filename', 'Label', 'Extracted Text' 열이 필요합니다.")
                return

            for row in reader:
                if len(row) <= max(filename_idx, label_idx, text_idx):
                    logger.warning(f"잘못된 형식의 행 발견 (건너뜀): {row}")
                    continue

                image_filename = row[filename_idx]
                label = row[label_idx]
                text = row[text_idx]

                if label == "N/A" or label == "오류": # 텍스트 없음 또는 오류 행은 건너뜀
                    if not file_data_map[image_filename]["문서명"]: # 해당 파일에 대한 데이터가 아직 없으면 기본값 유지
                         pass # 이미 defaultdict로 초기화됨
                    continue


                if label in file_data_map[image_filename]:
                    if label == "기타":
                        if text: file_data_map[image_filename][label].append(text)
                    else:
                        # 첫 번째로 발견된 유효한 텍스트를 사용
                        if not file_data_map[image_filename][label] and text:
                            file_data_map[image_filename][label] = text
                        elif file_data_map[image_filename][label] and text:
                            # 이미 값이 있는데 다른 값이 들어오면, 로깅하고 기존 값 유지 또는 특정 규칙 적용 가능
                            logger.debug(f"파일 '{image_filename}'의 라벨 '{label}'에 이미 값 '{file_data_map[image_filename][label]}'이 있지만, 새 값 '{text}' 발견. 기존 값 유지.")
                else:
                    # 정의되지 않은 라벨이 refine_and_finalize_labels에서 나올 수 있으므로, 기타로 처리
                    logger.warning(f"알 수 없는 라벨 '{label}' 발견 (파일: {image_filename}, 텍스트: {text}). '기타'로 처리합니다.")
                    if text: file_data_map[image_filename]["기타"].append(text)


    except FileNotFoundError:
        logger.error(f"입력 CSV 파일을 찾을 수 없습니다: {labeled_csv_path}")
        return
    except Exception as e:
        logger.error(f"'{labeled_csv_path}' 파일 읽기 중 오류 발생: {e}", exc_info=True)
        return

    # 출력 CSV 파일 작성
    output_header = ["Image Filename", "문서명", "이름", "주민등록번호", "주소", "발급일자", "발급기관", "기타"]
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(output_header)

            for image_filename, data in sorted(file_data_map.items()): # 파일명 순으로 정렬하여 출력
                # 기타 항목은 세미콜론으로 병합 (중복 제거 및 빈 문자열 필터링)
                unique_other_texts = sorted(list(set(filter(None, data["기타"]))))
                other_text_combined = "; ".join(unique_other_texts)

                row_to_write = [
                    image_filename,
                    data["문서명"],
                    data["이름"],
                    data["주민등록번호"],
                    data["주소"],
                    data["발급일자"],
                    data["발급기관"],
                    other_text_combined
                ]
                writer.writerow(row_to_write)
        logger.info(f"최종 요약 CSV 파일 생성 완료: {output_csv_path}")
    except IOError as e:
        logger.error(f"'{output_csv_path}' 파일 쓰기 중 오류 발생: {e}")
    except Exception as e:
        logger.error(f"최종 요약 CSV 파일 생성 중 예기치 않은 오류 발생: {e}", exc_info=True)


def main():
    """스크립트의 메인 실행 로직입니다."""
    try:
        # macOS 등에서 'fork' 방식 사용 시 문제 발생 가능성 있어 'spawn'으로 명시적 설정 시도
        if platform.system() != "Windows" and multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
            logger.debug("멀티프로세싱 시작 방식을 'spawn'으로 설정했습니다.")
    except RuntimeError:
        logger.debug("멀티프로세싱 시작 방식이 이미 설정되어 변경할 수 없습니다.")
    except AttributeError: # 일부 Python 버전/환경에서는 get_start_method(allow_none=True) 없을 수 있음
        logger.debug("multiprocessing.get_start_method(allow_none=True)를 사용할 수 없습니다.")


    if platform.system() == "Windows": # Windows에서는 freeze_support()가 필요할 수 있음
        multiprocessing.freeze_support()

    args = parse_arguments()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_level_env_name = 'OCR_WORKER_LOG_LEVEL' # 작업자용 로깅 레벨 환경 변수 이름
    os.environ[log_level_env_name] = logging.getLevelName(log_level) # 환경 변수에 로깅 레벨 설정
    setup_logging(log_level) # 메인 프로세스 로거 설정

    logger.info(f"OCR 스크립트 버전: {__version__}")
    logger.info(f"명령줄 인자: {args}")

    prepare_directories(args.input_dir, args.output_dir)

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_filenames = [
        f for f in os.listdir(args.input_dir) if f.lower().endswith(supported_extensions)
    ]

    if not image_filenames:
        logger.warning(f"입력 디렉토리 '{args.input_dir}'에서 지원되는 이미지 파일을 찾을 수 없습니다.")
        exit(0)

    num_workers_to_use = args.num_workers if args.num_workers is not None and args.num_workers > 0 else os.cpu_count()
    if num_workers_to_use is None: # os.cpu_count()가 None을 반환하는 경우 대비
        num_workers_to_use = 1
        logger.warning("CPU 코어 수를 감지할 수 없어 작업자 수를 1로 설정합니다.")
    logger.info(f"사용할 작업자 프로세스 수: {num_workers_to_use}")
    logger.info(f"총 {len(image_filenames)}개의 이미지 파일을 처리합니다.")


    ocr_engine_params = {
        'use_angle_cls': True,
        'lang': args.lang,
        'use_gpu': args.use_gpu,
        'show_log': False # PaddleOCR 자체 로그는 제어 (스크립트 로거 사용)
    }
    if '+' in args.lang: # 예: 'korean+en'
        logger.info(f"복합 언어 설정 '{args.lang}' 감지. PaddleOCR이 해당 설정을 지원하는지 확인하세요.")

    font_path_for_tasks = determine_font_for_visualization() # 시각화에 사용할 폰트 결정

    # 중간 결과 CSV 파일 (ocr_labeled_text.csv) 경로
    intermediate_csv_filename = "ocr_labeled_text.csv"
    intermediate_csv_path = os.path.join(args.output_dir, intermediate_csv_filename)


    # Manager는 Pool 생성 전에 시작되어야 할 수 있음 (특히 Lock 객체)
    manager = multiprocessing.Manager()
    csv_file_lock = manager.Lock() # CSV 파일 접근 동기화를 위한 Lock

    # CSV 헤더 정의 (라벨링된 각 텍스트 조각 기준)
    # 이 헤더는 process_single_image_task 내부에서 사용되는 CSV 파일의 헤더입니다.
    csv_fieldnames_intermediate = ["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"]

    try:
        with open(intermediate_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile: # utf-8-sig for Excel
            writer = csv.writer(csvfile)
            writer.writerow(csv_fieldnames_intermediate) # 헤더 작성
        logger.info(f"중간 CSV 파일 '{intermediate_csv_path}'가 초기화되었습니다 (헤더 작성 완료).")
    except IOError as e:
        logger.error(f"중간 CSV 파일 초기화 중 오류 ({intermediate_csv_path}): {e}")
        logger.error("CSV 파일 생성에 실패하여 스크립트를 종료합니다.")
        exit(1)


    task_arguments_list = []
    for filename in image_filenames:
        current_image_path = os.path.join(args.input_dir, filename)
        task_arguments_list.append((
            current_image_path,
            filename,
            ocr_engine_params,
            args.output_dir,
            args.no_preprocess,
            args.show_image,
            font_path_for_tasks
            # Lock과 CSV 경로는 worker_initializer_func를 통해 전달됨
        ))

    pool = None
    try:
        # Pool 생성 시 초기화 함수와 인자 전달
        pool = multiprocessing.Pool(
            processes=num_workers_to_use,
            initializer=worker_initializer_func,
            initargs=(csv_file_lock, intermediate_csv_path, log_level_env_name) # Lock과 CSV 경로 전달
        )

        logger.info("병렬 이미지 처리 시작...")

        results_from_pool = []
        # imap_unordered는 작업 완료 순서대로 결과를 반환 (tqdm과 함께 사용 용이)
        for result_status in tqdm(pool.imap_unordered(process_single_image_task, task_arguments_list),
                                  total=len(task_arguments_list), desc="전체 이미지 처리 중"):
            if result_status: # 작업자 함수가 반환한 상태 메시지
                results_from_pool.append(result_status)
                logger.debug(f"작업 결과 수신: {result_status}")

        logger.info("모든 작업이 풀에 제출되었고 결과 반복이 완료되었습니다.")

    except Exception as e:
        logger.error(f"병렬 처리 중 주 프로세스에서 예상치 못한 오류 발생: {e}", exc_info=True)
    finally:
        if pool:
            logger.info("풀 종료를 시작합니다...")
            pool.close() # 더 이상 새 작업 받지 않음
            logger.info("Pool.close() 호출 완료. 작업자 프로세스 종료 대기 중...")
            pool.join() # 모든 작업자 프로세스가 종료될 때까지 대기
            logger.info("Pool.join() 호출 완료. 모든 작업자 프로세스가 종료되었습니다.")

    logger.info(f"총 {len(image_filenames)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info(f"개별 추출된 텍스트 정보는 '{intermediate_csv_path}' 에 저장되었습니다.")

    # 최종 요약 CSV 파일 (ocr_text.csv) 생성
    final_summary_csv_filename = "ocr_text.csv"
    final_summary_csv_path = os.path.join(args.output_dir, final_summary_csv_filename)
    create_final_summary_csv(intermediate_csv_path, final_summary_csv_path)


    logger.info(f"스크립트 실행 완료. 모든 결과는 '{args.output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
