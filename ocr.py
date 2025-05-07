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

# 스크립트 시작 시점에 ccache 경고 필터 설정 (PaddleOCR 임포트 전에 적용되도록)
warnings.filterwarnings("ignore", category=UserWarning, message="No ccache found.*")

from paddleocr import PaddleOCR, draw_ocr 

__version__ = "1.0.3" # 스크립트 버전 정보 (CSV 항상 출력으로 변경)

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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3) 
        logger.debug(f"전처리 완료 (CLAHE, Median Blur): {os.path.basename(image_path)}")
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
                extracted_items.append({
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": bounding_box
                })
        return extracted_items 
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: OCR 처리 중 오류 (파일: {filename_for_log}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))
        return None 
    finally:
        if ocr_engine:
            del ocr_engine 
            logger.debug(f"작업자 {os.getpid()}: PaddleOCR 엔진 삭제됨 (파일: {filename_for_log})")

def label_text_item(extracted_item, image_width, image_height):
    """추출된 텍스트 항목에 라벨을 할당합니다."""
    text = extracted_item['text']
    box = extracted_item['bounding_box'] 
    
    y_center = 0
    if box and len(box) == 4 and all(len(pt) == 2 for pt in box):
        y_center = sum(pt[1] for pt in box) / 4
    else:
        logger.warning(f"유효하지 않은 바운딩 박스 데이터로 라벨링 시도: {box} (텍스트: {text})")

    if re.fullmatch(r"\d{6}\s*-\s*\d{7}", text) or \
       re.fullmatch(r"\d{13}", text.replace("-","").replace(" ","")):
        return "주민등록번호"
        
    cleaned_text = text.replace(" ","")
    if 2 <= len(cleaned_text) <= 5 and re.fullmatch(r"^[가-힣]+$", cleaned_text):
        if y_center > 0 and y_center < image_height * 0.45 and len(cleaned_text) <= 4 :
             return "이름"

    if re.search(r"\d{4}\s*[\.,년]\s*\d{1,2}\s*[\.,월]\s*\d{1,2}\s*[\.일]?", text): 
        return "발급일"
        
    if text.endswith("청장") or text.endswith("시장") or text.endswith("군수") or text.endswith("구청장") or \
       text.endswith("경찰서장") or text.endswith("지방경찰청장"):
        return "발급기관"
        
    address_keywords = ["특별시", "광역시", "도", "시 ", "군 ", "구 ", "읍 ", "면 ", "동 ", "리 ", "로 ", "길 ", "아파트", "빌라", " 번지", " 호"]
    for keyword in address_keywords:
        if keyword.strip() in text: 
            return "주소" 

    doc_titles = ["주민등록증", "운전면허증"]
    for title in doc_titles:
        if title in text and y_center > 0 and y_center < image_height * 0.25: 
            return "문서명"

    return "기타" 


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
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/korean/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 
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

    local_korean_font = os.path.join(script_dir, 'fonts', 'malgun.ttf')
    if os.path.exists(local_korean_font):
        font_path_to_use = local_korean_font
        logger.info(f"로컬 한국어 폰트 사용: {font_path_to_use}")
        return font_path_to_use

    logger.info(f"로컬 한국어 폰트 '{local_korean_font}'를 찾지 못했습니다.")
    local_english_font = os.path.join(script_dir, 'fonts', 'arial.ttf')
    if os.path.exists(local_english_font):
        font_path_to_use = local_english_font
        logger.warning(f"로컬 영문 대체 폰트 사용: {font_path_to_use} (한글이 깨질 수 있습니다.)")
        return font_path_to_use
    
    logger.warning(f"로컬 영문 대체 폰트 '{local_english_font}'도 찾지 못했습니다.")
    logger.warning("사용 가능한 특정 폰트를 찾지 못했습니다. 시각화 시 PaddleOCR 내부 기본 폰트가 사용될 수 있습니다.")
    return None


def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image_flag=False, font_path_to_use=None):
    """OCR 결과를 이미지 위에 표시하고 저장합니다."""
    if not extracted_data: 
        logger.info(f"{original_filename}: 시각화할 OCR 결과가 없습니다.")
        return

    try:
        if preprocessed_img is not None:
            image_to_draw_on = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB) if len(preprocessed_img.shape) == 2 else preprocessed_img
            image = Image.fromarray(image_to_draw_on)
        else:
            image = Image.open(original_image_path).convert('RGB')

        boxes = [item['bounding_box'] for item in extracted_data]
        txts = [item['text'] for item in extracted_data]
        scores = [item['confidence'] for item in extracted_data]
        
        try:
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path_to_use)
        except Exception as e:
            logger.error(f"draw_ocr 중 오류 (폰트: {font_path_to_use}, 파일: {original_filename}): {e}. 폰트 없이 재시도합니다.")
            im_show = draw_ocr(image, boxes, font_path=None) 

        im_show_pil = Image.fromarray(im_show)
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
    global _worker_csv_lock, _worker_csv_file_path # 작업자 내 전역 변수 사용
    
    (current_image_path, filename, ocr_engine_params, output_dir, 
     skip_preprocessing, show_image, font_path) = task_args_tuple # save_text_flag 제거
    
    status_message = f"{filename} 처리 중 오류 발생" 

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

        if not skip_preprocessing:
            logger.debug(f"작업자 {os.getpid()}: 외부 전처리 시작: {filename}")
            processed_img_data = preprocess_image_for_ocr(current_image_path)
            if processed_img_data is not None:
                ocr_input_data = processed_img_data 
                processed_image_for_display = processed_img_data 
            else:
                logger.warning(f"작업자 {os.getpid()}: {filename}: 외부 전처리에 실패하여 원본 이미지로 OCR을 시도합니다.")
        
        extracted_items = extract_text_from_image_worker(ocr_engine_params, ocr_input_data, filename_for_log=filename)

        if extracted_items:
            logger.debug(f"작업자 {os.getpid()}: 추출된 텍스트 항목 수 ({filename}): {len(extracted_items)}")
            
            labeled_results_for_csv = []
            for item in extracted_items:
                assigned_label = "기타" 
                if image_width > 0 and image_height > 0: 
                    assigned_label = label_text_item(item, image_width, image_height)
                
                if logger.isEnabledFor(logging.DEBUG):
                     logger.debug(f"  - 라벨: {assigned_label}, 텍스트: \"{item['text']}\" (신뢰도: {item['confidence']:.3f})")
                
                # CSV 저장은 항상 수행 (save_text_flag 제거)
                labeled_results_for_csv.append([
                    filename,
                    assigned_label,
                    item['text'].replace('"', '""'), 
                    round(item['confidence'], 4),
                    str(item['bounding_box']) 
                ])
            
            if labeled_results_for_csv and _worker_csv_lock and _worker_csv_file_path:
                with _worker_csv_lock: 
                    try:
                        with open(_worker_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerows(labeled_results_for_csv)
                        logger.info(f"작업자 {os.getpid()}: {filename}의 텍스트를 {os.path.basename(_worker_csv_file_path)}에 추가했습니다.")
                    except IOError as e:
                        logger.error(f"작업자 {os.getpid()}: CSV 파일 쓰기 중 오류 ({_worker_csv_file_path}, 파일: {filename}): {e}")


            display_ocr_result(current_image_path, extracted_items, output_dir, filename,
                               preprocessed_img=processed_image_for_display, 
                               show_image_flag=show_image,
                               font_path_to_use=font_path)
            status_message = f"{filename} 처리 성공"
        else:
            logger.info(f"작업자 {os.getpid()}: {filename}에서 텍스트를 추출하지 못했습니다.")
            status_message = f"{filename} 텍스트 없음"
        
        return status_message
    except Exception as e:
        logger.error(f"작업자 {os.getpid()}: process_single_image_task 내에서 예상치 못한 오류 발생 (파일: {filename}): {e}", exc_info=True)
        return status_message 


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
    # --save_text 옵션 제거됨
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

def main():
    """스크립트의 메인 실행 로직입니다."""
    try:
        if multiprocessing.get_start_method(allow_none=True) is None: 
            multiprocessing.set_start_method('spawn', force=True)
            logger.debug("멀티프로세싱 시작 방식을 'spawn'으로 설정했습니다.")
    except RuntimeError:
        logger.debug("멀티프로세싱 시작 방식이 이미 설정되어 변경할 수 없습니다.")
    
    multiprocessing.freeze_support() 

    args = parse_arguments()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_level_env_name = 'OCR_WORKER_LOG_LEVEL' 
    os.environ[log_level_env_name] = logging.getLevelName(log_level) 
    setup_logging(log_level) 

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

    num_workers = os.cpu_count()
    logger.info(f"사용할 작업자 프로세스 수 (시스템 CPU 코어 수): {num_workers}")
    logger.info(f"총 {len(image_filenames)}개의 이미지 파일을 처리합니다.")

    ocr_engine_params = {
        'use_angle_cls': True, 
        'lang': args.lang, 
        'use_gpu': args.use_gpu, 
        'show_log': False 
    }
    if '+' in args.lang: 
        logger.info(f"복합 언어 설정 '{args.lang}' 감지. PaddleOCR이 해당 설정을 지원하는지 확인하세요.")

    font_path_for_tasks = determine_font_for_visualization()
    
    # CSV 파일은 항상 생성됨
    csv_output_file_path = os.path.join(args.output_dir, "ocr_labeled_text.csv")
    manager = multiprocessing.Manager()
    csv_file_lock = manager.Lock() # Lock 객체는 항상 생성

    try:
        with open(csv_output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Image Filename", "Label", "Extracted Text", "Confidence", "Bounding Box (str)"])
        logger.info(f"CSV 파일 '{csv_output_file_path}'가 초기화되었습니다 (헤더 작성 완료).")
    except IOError as e:
        logger.error(f"CSV 파일 초기화 중 오류 ({csv_output_file_path}): {e}")
        logger.error("CSV 파일 생성에 실패하여 스크립트를 종료합니다.")
        exit(1) # CSV 파일 생성 실패 시 종료


    task_arguments_list = []
    for filename in image_filenames:
        current_image_path = os.path.join(args.input_dir, filename)
        task_arguments_list.append((
            current_image_path, 
            filename, 
            ocr_engine_params, 
            args.output_dir,
            args.no_preprocess, 
            # args.save_text, # 제거됨. CSV 저장은 항상 True로 간주
            args.show_image,
            font_path_for_tasks
            # csv_lock, csv_output_file_path 는 worker_initializer_func 를 통해 전달됨
        ))
    
    pool = None 
    try:
        pool = multiprocessing.Pool(
            processes=num_workers, 
            initializer=worker_initializer_func, 
            initargs=(csv_file_lock, csv_output_file_path, log_level_env_name) 
        )

        logger.info("병렬 이미지 처리 시작...")
        
        results_from_pool = []
        for result_status in tqdm(pool.imap_unordered(process_single_image_task, task_arguments_list), 
                                  total=len(task_arguments_list), desc="전체 이미지 처리 중"):
            if result_status: 
                results_from_pool.append(result_status) 
                logger.debug(f"작업 결과 수신: {result_status}")
            
        logger.info("모든 작업이 풀에 제출되었고 결과 반복이 완료되었습니다.")
            
    except Exception as e:
        logger.error(f"병렬 처리 중 주 프로세스에서 예상치 못한 오류 발생: {e}", exc_info=True)
    finally:
        if pool:
            logger.info("풀 종료를 시작합니다...")
            pool.close() 
            logger.info("Pool.close() 호출 완료. 작업자 프로세스 종료 대기 중...")
            pool.join() 
            logger.info("Pool.join() 호출 완료. 모든 작업자 프로세스가 종료되었습니다.")
        
    logger.info(f"총 {len(image_filenames)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info(f"추출된 텍스트는 {csv_output_file_path} 에 저장되었습니다.")
    logger.info("스크립트 실행 종료.")

if __name__ == "__main__":
    main()
