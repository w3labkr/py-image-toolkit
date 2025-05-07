# PaddleOCR을 사용하여 이미지에서 텍스트를 추출하는 스크립트 (OpenCV 전처리 및 폴더 처리 추가)
import cv2 # OpenCV 라이브러리 임포트
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np # NumPy 임포트 (OpenCV 이미지 처리에 사용)
import os # 파일 및 디렉토리 관리를 위해 os 모듈 임포트
import argparse # 명령줄 인자 처리를 위해 argparse 모듈 임포트
import platform # 운영체제 감지를 위해 platform 모듈 임포트
from tqdm import tqdm # 진행률 표시를 위해 tqdm 임포트
import logging # 로깅 모듈 임포트

__version__ = "0.6.0" # 스크립트 버전 정보 (텍스트 파일 저장, GPU 옵션 추가)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """기본 로거를 설정합니다."""
    if not logger.handlers: # 핸들러가 중복 추가되는 것을 방지
        logger.setLevel(level)
        ch = logging.StreamHandler() # 콘솔 핸들러
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def preprocess_image_for_ocr(image_path):
    """
    OCR 정확도 향상을 위해 이미지를 전처리합니다.
    (그레이스케일, 대비 향상, 노이즈 제거)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
            return None

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_img = clahe.apply(gray_img)
        denoised_img = cv2.medianBlur(enhanced_contrast_img, 3)
        logger.debug(f"이미지 전처리 완료: {os.path.basename(image_path)}")
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 발생 (이미지 전처리 중 {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"이미지 전처리 중 예외 발생 ({os.path.basename(image_path)}): {e}")
        return None

def extract_text_from_image(image_path_or_data, lang='korean', use_gpu=False, preprocess_flag=True, filename_for_log=""):
    """
    주어진 이미지 경로 또는 데이터에서 텍스트를 추출합니다.
    """
    try:
        # PaddleOCR 초기화 시 GPU 사용 여부 전달
        ocr_params = {'use_angle_cls': True, 'use_gpu': use_gpu, 'show_log': False}
        if lang == 'korean':
            ocr_params['lang'] = 'korean'
        elif lang == 'english':
            ocr_params['lang'] = 'en'
        elif lang == 'korean+en':
            ocr_params['lang'] = 'korean' 
            # 이 메시지는 lang='korean+en'일 때마다 출력될 수 있으므로 INFO 레벨 유지
            logger.info("참고: 'korean+en' 설정 시, 기본 한국어 모델을 사용하며 영어도 일부 인식 가능합니다. 정확한 다국어 처리를 위해서는 특정 모델 또는 설정이 필요할 수 있습니다.")
        else:
            logger.warning(f"지원하지 않는 언어 설정입니다: {lang}. 기본값으로 한국어를 사용합니다.")
            ocr_params['lang'] = 'korean'
        
        logger.debug(f"PaddleOCR 초기화 파라미터: {ocr_params} (파일: {filename_for_log})")
        ocr = PaddleOCR(**ocr_params)

        input_data = image_path_or_data
        if isinstance(image_path_or_data, str) and preprocess_flag:
            logger.debug(f"내부 전처리 시작: {filename_for_log}")
            processed_img = preprocess_image_for_ocr(image_path_or_data)
            if processed_img is None:
                logger.warning(f"{filename_for_log}: 이미지 전처리에 실패하여 OCR을 진행할 수 없습니다.")
                return None
            input_data = processed_img
        elif not isinstance(image_path_or_data, (str, np.ndarray)):
            logger.error(f"{filename_for_log}: 잘못된 입력 데이터 타입입니다. 이미지 경로(str) 또는 NumPy 배열이어야 합니다.")
            return None

        logger.debug(f"OCR 시작: {filename_for_log}")
        result = ocr.ocr(input_data, cls=True)
        logger.debug(f"OCR 완료: {filename_for_log}")


        extracted_texts = []
        if result and result[0] is not None:
            for line_info in result[0]:
                text = line_info[1][0]
                confidence = line_info[1][1]
                bounding_box = line_info[0]
                extracted_texts.append({
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": bounding_box
                })
        return extracted_texts

    except ImportError:
        logger.error("PaddleOCR 또는 PaddlePaddle 라이브러리가 설치되지 않았습니다.")
        logger.error("pip install paddlepaddle paddleocr opencv-python tqdm 명령어로 설치해주세요.")
        return None
    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생 ({filename_for_log}): {e}")
        return None

def _initialize_font_message_flag():
    if not hasattr(get_os_specific_font_path, 'font_message_printed'):
        get_os_specific_font_path.font_message_printed = False
_initialize_font_message_flag()


def get_os_specific_font_path():
    """
    운영 체제에 맞는 기본 한글 폰트 경로를 반환합니다.
    """
    system = platform.system()
    font_path = None

    if system == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf" 
    elif system == "Darwin": 
        font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc" 
    elif system == "Linux":
        common_linux_fonts = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/korean/NanumGothic.ttf", # 일부 배포판 경로
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", # Noto Sans (통합)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # 기본 영문 폰트 (대체용)
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    
    # 이 함수가 호출될 때마다 메시지를 출력하지 않도록 플래그 관리
    if not get_os_specific_font_path.font_message_printed:
        if font_path and os.path.exists(font_path):
            logger.info(f"OS 감지: {system}, 시스템 폰트 사용 시도: {font_path}")
        elif font_path: # 경로는 있으나 파일이 없는 경우
            logger.warning(f"OS 감지: {system}, 지정된 시스템 폰트 '{font_path}'를 찾을 수 없습니다.")
        else: # OS에 대한 특정 경로가 없는 경우
            logger.warning(f"OS 감지: {system}, 이 OS에 대한 기본 한글 폰트 경로가 설정되지 않았습니다.")
        get_os_specific_font_path.font_message_printed = True # 메시지 출력 후 플래그 설정

    if font_path and os.path.exists(font_path):
        return font_path
    return None # 최종적으로 유효한 경로가 없으면 None 반환

def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image_flag=False):
    """
    OCR 결과를 이미지 위에 표시하고 저장합니다.
    """
    if not extracted_data:
        logger.info(f"표시할 OCR 결과가 없습니다: {original_filename}")
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

        # 폰트 경로 결정 (get_os_specific_font_path가 먼저 호출되어 font_message_printed 플래그를 설정할 수 있음)
        font_path = get_os_specific_font_path() 
        
        # 로컬 폴더 폰트 시도 (OS 폰트 실패 시)
        if not font_path:
            local_font_path_korean = './fonts/malgun.ttf' # 스크립트 위치 기준
            if os.path.exists(local_font_path_korean):
                font_path = local_font_path_korean
                if not get_os_specific_font_path.font_message_printed: # 아직 폰트 관련 메시지 안나갔으면
                    logger.info(f"로컬 한국어 폰트 사용 시도: {font_path}")
                    get_os_specific_font_path.font_message_printed = True 
        
        # 영문 대체 폰트 시도 (모든 한국어 폰트 실패 시)
        if not font_path:
            local_font_path_english = './fonts/arial.ttf'
            if os.path.exists(local_font_path_english):
                font_path = local_font_path_english
                if not get_os_specific_font_path.font_message_printed:
                    logger.info(f"로컬 영문 대체 폰트 사용 시도: {font_path} (한글 표시 안될 수 있음)")
                    get_os_specific_font_path.font_message_printed = True
        
        # 최종 폰트 경로 유효성 검사 및 로그 (한 번만)
        if not hasattr(display_ocr_result, 'final_font_choice_logged'):
            if font_path and os.path.exists(font_path):
                logger.info(f"텍스트 렌더링에 사용할 최종 폰트: {font_path}")
            elif font_path: # 경로는 있으나 파일이 없는 경우 (이 경우는 위에서 처리되어야 함)
                logger.warning(f"최종 선택된 폰트 '{font_path}'를 찾을 수 없습니다. 폰트 없이 텍스트를 표시합니다.")
                font_path = None
            else: # font_path가 None인 경우
                logger.warning("사용 가능한 특정 폰트를 찾지 못했습니다. PaddleOCR 내부 기본 폰트를 사용합니다.")
            display_ocr_result.final_font_choice_logged = True


        try:
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        except Exception as e:
            logger.error(f"draw_ocr 중 오류 발생 (폰트: {font_path}, 파일: {original_filename}): {e}. 폰트 없이 재시도합니다.")
            im_show = draw_ocr(image, boxes, font_path=None) # 폰트 없이 시도

        im_show_pil = Image.fromarray(im_show)
        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr_result{ext}" # 결과 이미지 파일명에 "_ocr_result" 추가
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        im_show_pil.save(output_image_path)
        logger.debug(f"OCR 결과 시각화 이미지가 {output_image_path}에 저장되었습니다.")
        
        if show_image_flag:
            im_show_pil.show()

    except FileNotFoundError:
        logger.error(f"원본 이미지 파일 '{original_image_path}'를 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        logger.error(f"OCR 결과 표시/저장 중 오류 발생 ({original_filename}): {e}")

def save_extracted_text(extracted_data, output_dir, original_filename):
    """추출된 텍스트를 .txt 파일로 저장합니다."""
    if not extracted_data:
        logger.info(f"{original_filename}: 저장할 추출된 텍스트가 없습니다.")
        return

    base, _ = os.path.splitext(original_filename)
    output_text_filename = f"{base}_ocr_text.txt" # 텍스트 파일명에 "_ocr_text" 추가
    output_text_path = os.path.join(output_dir, output_text_filename)

    try:
        with open(output_text_path, 'w', encoding='utf-8') as f:
            for item in extracted_data:
                f.write(item['text'] + '\n')
        logger.info(f"추출된 텍스트가 {output_text_path}에 저장되었습니다.")
    except IOError as e:
        logger.error(f"텍스트 파일 저장 중 오류 발생 ({output_text_path}): {e}")


if __name__ == "__main__":
    setup_logger() 
    # 함수 속성으로 플래그 관리 (스크립트 실행 시마다 초기화)
    get_os_specific_font_path.font_message_printed = False
    display_ocr_result.final_font_choice_logged = False


    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.")
    parser.add_argument("input_dir", nargs='?', default='input', help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다. (기본값: 'input')")
    parser.add_argument("--output_dir", default='output', help="OCR 결과 이미지를 저장할 폴더 경로입니다. (기본값: 'output')")
    parser.add_argument("--lang", help="OCR에 사용할 언어입니다 (예: korean, en, korean+en). (기본값: korean)", default='korean')
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', help="디버그 레벨 로깅을 활성화합니다.")
    parser.add_argument('--use_gpu', action='store_true', help="사용 가능한 경우 GPU를 사용하여 OCR 처리를 시도합니다.")
    parser.add_argument('--save_text', action='store_true', help="추출된 텍스트를 이미지와 동일한 기반 이름의 .txt 파일로 저장합니다.")


    args = parser.parse_args()

    if args.debug:
        setup_logger(logging.DEBUG) 
        logger.debug("디버그 모드가 활성화되었습니다.")

    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    selected_language = args.lang
    display_images_flag = args.show_image
    skip_preprocessing = args.no_preprocess
    use_gpu_flag = args.use_gpu
    save_text_flag = args.save_text


    if not os.path.isdir(input_dir_path):
        logger.error(f"입력 디렉토리 '{input_dir_path}'를 찾을 수 없거나 디렉토리가 아닙니다. 스크립트를 종료합니다.")
        logger.info(f"팁: '{input_dir_path}' 디렉토리를 생성하거나 올바른 경로를 지정해주세요.")
        exit(1)

    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            logger.info(f"출력 디렉토리 생성: {output_dir_path}")
        except OSError as e:
            logger.error(f"출력 디렉토리 '{output_dir_path}'를 생성할 수 없습니다. {e}")
            exit(1)
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files_to_process = [
        filename for filename in os.listdir(input_dir_path)
        if filename.lower().endswith(supported_extensions)
    ]
    
    if not image_files_to_process:
        logger.warning(f"입력 디렉토리 '{input_dir_path}'에서 지원되는 이미지 파일을 찾을 수 없습니다.")
        logger.warning(f"지원 확장자: {supported_extensions}")
        exit(0)

    logger.info(f"OCR 스크립트 버전: {__version__}")
    logger.info(f"입력 디렉토리: {input_dir_path}")
    logger.info(f"출력 디렉토리: {output_dir_path}")
    logger.info(f"선택 언어: {selected_language}")
    logger.info(f"GPU 사용 시도: {use_gpu_flag}")
    logger.info(f"전처리 수행: {not skip_preprocessing}")
    logger.info(f"결과 이미지 표시: {display_images_flag}")
    logger.info(f"추출 텍스트 파일 저장: {save_text_flag}")
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일을 처리합니다.")
    
    # 루프 시작 전 폰트 관련 플래그 초기화 (get_os_specific_font_path 내부 플래그는 이미 초기화됨)
    display_ocr_result.final_font_choice_logged = False 


    for filename in tqdm(image_files_to_process, desc="OCR 처리 중", unit="개"):
        current_image_path = os.path.join(input_dir_path, filename)
        logger.debug(f"처리 시작: {filename}")

        processed_image_data = None
        ocr_input_data = current_image_path 

        if not skip_preprocessing:
            logger.debug(f"외부 전처리 시작: {filename}")
            processed_image_data = preprocess_image_for_ocr(current_image_path)
            if processed_image_data is not None:
                ocr_input_data = processed_image_data 
            else:
                logger.warning(f"{filename} 외부 전처리에 실패. 원본 이미지로 OCR 시도.")
        else:
            logger.info(f"{filename}: 전처리 단계를 건너뜁니다.") 

        extracted_data = extract_text_from_image(ocr_input_data, 
                                                 lang=selected_language, 
                                                 use_gpu=use_gpu_flag,
                                                 preprocess_flag=False, # 외부에서 이미 전처리했거나 건너뛰므로 False
                                                 filename_for_log=filename)

        if extracted_data:
            logger.debug(f"추출된 텍스트 항목 수 ({filename}): {len(extracted_data)}")
            for i, item in enumerate(extracted_data):
                 logger.debug(f"  - \"{item['text']}\" (신뢰도: {item['confidence']:.3f})")
            
            if save_text_flag:
                save_extracted_text(extracted_data, output_dir_path, filename)

            display_ocr_result(current_image_path, extracted_data, output_dir_path, filename,
                               preprocessed_img=processed_image_data if not skip_preprocessing else None, 
                               show_image_flag=display_images_flag)
        else:
            logger.info(f"{filename}에서 텍스트를 추출하지 못했습니다.")
        
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info("스크립트 실행 종료.")

