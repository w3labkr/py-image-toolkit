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

__version__ = "0.5.0" # 스크립트 버전 정보 (로깅 기능 추가)

# --- 로거 설정 ---
# 기본 로거 설정 (스크립트 최상단 또는 main 함수 시작 부분에서 설정 가능)
# 여기서는 main 함수 내에서 로거를 가져오고 설정합니다.
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
        # logger.debug(f"이미지 전처리 완료: {image_path}") # 디버그 레벨에서만 출력
        return denoised_img
    except cv2.error as e:
        logger.error(f"OpenCV 오류 발생 (이미지 전처리 중 {os.path.basename(image_path)}): {e}")
        return None
    except Exception as e:
        logger.error(f"이미지 전처리 중 예외 발생 ({os.path.basename(image_path)}): {e}")
        return None

def extract_text_from_image(image_path_or_data, lang='korean', preprocess_flag=True, filename_for_log=""):
    """
    주어진 이미지 경로 또는 데이터에서 텍스트를 추출합니다.
    preprocess_flag 인자명을 preprocess에서 변경하여 내장 함수와 혼동 방지
    """
    try:
        ocr_params = {'use_angle_cls': True, 'use_gpu': False, 'show_log': False}
        if lang == 'korean':
            ocr_params['lang'] = 'korean'
        elif lang == 'english':
            ocr_params['lang'] = 'en'
        elif lang == 'korean+en':
            ocr_params['lang'] = 'korean' 
            logger.info("'korean+en'의 경우, 기본 한국어 모델을 사용합니다. 영어 인식 성능은 모델에 따라 다를 수 있습니다.")
        else:
            logger.warning(f"지원하지 않는 언어 설정입니다: {lang}. 기본값으로 한국어를 사용합니다.")
            ocr_params['lang'] = 'korean'
        
        ocr = PaddleOCR(**ocr_params)

        input_data = image_path_or_data
        if isinstance(image_path_or_data, str) and preprocess_flag:
            # logger.debug(f"이미지 전처리를 시작합니다: {filename_for_log if filename_for_log else image_path_or_data}")
            processed_img = preprocess_image_for_ocr(image_path_or_data)
            if processed_img is None:
                logger.warning(f"{filename_for_log}: 이미지 전처리에 실패하여 OCR을 진행할 수 없습니다.")
                return None
            input_data = processed_img
        elif not isinstance(image_path_or_data, (str, np.ndarray)):
            logger.error(f"{filename_for_log}: 잘못된 입력 데이터 타입입니다. 이미지 경로(str) 또는 NumPy 배열이어야 합니다.")
            return None

        result = ocr.ocr(input_data, cls=True)

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

# 폰트 메시지 중복 출력을 방지하기 위한 플래그 (함수 속성으로 관리)
# 이 플래그는 get_os_specific_font_path 함수 내에서 사용됩니다.
# 전역 변수 대신 함수 속성을 사용하여 상태를 관리합니다.
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
            "/usr/share/fonts/korean/NanumGothic.ttf",
        ]
        for p in common_linux_fonts:
            if os.path.exists(p):
                font_path = p
                break
    
    if font_path and os.path.exists(font_path):
        if not get_os_specific_font_path.font_message_printed:
            logger.info(f"OS 감지: {system}, 시스템 폰트 사용 시도: {font_path}")
            get_os_specific_font_path.font_message_printed = True
        return font_path
    else:
        if not get_os_specific_font_path.font_message_printed:
            if font_path: 
                 logger.warning(f"OS 감지: {system}, 지정된 시스템 폰트 '{font_path}'를 찾을 수 없습니다.")
            else: 
                 logger.warning(f"OS 감지: {system}, 이 OS에 대한 기본 한글 폰트 경로가 설정되지 않았습니다.")
            get_os_specific_font_path.font_message_printed = True
        return None

def display_ocr_result(original_image_path, extracted_data, output_dir, original_filename, 
                       preprocessed_img=None, show_image_flag=False): # show_image -> show_image_flag
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

        font_path = get_os_specific_font_path() 
        
        if not font_path:
            local_font_path = './fonts/malgun.ttf'
            if os.path.exists(local_font_path):
                font_path = local_font_path
                if not get_os_specific_font_path.font_message_printed:
                    logger.info(f"로컬 폰트 사용 시도: {font_path}")
                    get_os_specific_font_path.font_message_printed = True
        
        if not font_path:
            default_font_path = './fonts/arial.ttf'
            if os.path.exists(default_font_path):
                font_path = default_font_path
                if not get_os_specific_font_path.font_message_printed:
                    logger.info(f"영문 대체 로컬 폰트 사용 시도: {font_path} (한글이 깨질 수 있음)")
                    get_os_specific_font_path.font_message_printed = True
        
        if font_path and not os.path.exists(font_path):
            if not get_os_specific_font_path.font_message_printed:
                logger.warning(f"최종 선택된 폰트 '{font_path}'를 찾을 수 없습니다. 폰트 없이 텍스트를 표시합니다.")
            font_path = None 
            get_os_specific_font_path.font_message_printed = True

        try:
            if font_path and not get_os_specific_font_path.font_message_printed: # 메시지 한 번만 출력
                 logger.info(f"텍스트 렌더링에 사용할 폰트: {font_path}")
                 get_os_specific_font_path.font_message_printed = True
            elif not font_path and not get_os_specific_font_path.font_message_printed:
                 logger.warning("사용 가능한 폰트를 찾지 못했습니다. 폰트 없이 텍스트를 표시합니다 (PaddleOCR 기본값 사용).")
                 get_os_specific_font_path.font_message_printed = True
            
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        except Exception as e:
            logger.error(f"draw_ocr 중 오류 발생 (폰트: {font_path}, 파일: {original_filename}): {e}. 폰트 없이 재시도합니다.")
            im_show = draw_ocr(image, boxes, font_path=None) 

        im_show_pil = Image.fromarray(im_show)
        base, ext = os.path.splitext(original_filename)
        output_image_filename = f"{base}_ocr{ext}"
        output_image_path = os.path.join(output_dir, output_image_filename)
        
        im_show_pil.save(output_image_path)
        logger.debug(f"OCR 결과가 표시된 이미지가 {output_image_path}에 저장되었습니다.") # 디버그 레벨
        
        if show_image_flag:
            im_show_pil.show()

    except FileNotFoundError:
        logger.error(f"원본 이미지 파일 '{original_image_path}'를 찾을 수 없습니다 (결과 표시 중).")
    except Exception as e:
        logger.error(f"OCR 결과 표시/저장 중 오류 발생 ({original_filename}): {e}")

if __name__ == "__main__":
    setup_logger() # 로거 설정 실행

    parser = argparse.ArgumentParser(description="이미지 폴더에서 텍스트를 추출하는 OCR 스크립트입니다.")
    parser.add_argument("input_dir", nargs='?', default=None, help="텍스트를 추출할 이미지가 포함된 폴더의 경로입니다. (생략 시 버전 정보만 표시)")
    parser.add_argument("--output_dir", help="OCR 결과 이미지를 저장할 폴더 경로입니다. (기본값: 입력 폴더 내 'output_ocr_results')", default=None)
    parser.add_argument("--lang", help="OCR에 사용할 언어입니다 (예: korean, en, korean+en). (기본값: korean)", default='korean')
    parser.add_argument("--show_image", action='store_true', help="처리된 각 이미지와 OCR 결과를 화면에 표시합니다.")
    parser.add_argument("--no_preprocess", action='store_true', help="이미지 전처리 단계를 건너뜁니다.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}', help="스크립트 버전을 표시하고 종료합니다.")
    parser.add_argument('--debug', action='store_true', help="디버그 레벨 로깅을 활성화합니다.")

    args = parser.parse_args()

    if args.debug:
        setup_logger(logging.DEBUG) # 디버그 모드 시 로깅 레벨 변경
        logger.debug("디버그 모드가 활성화되었습니다.")

    if args.input_dir is None:
        if not any(arg.startswith('--version') or arg == '-v' for arg in os.sys.argv):
             parser.print_help()
        exit(0)

    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    selected_language = args.lang
    display_images_flag = args.show_image # 변수명 변경
    skip_preprocessing = args.no_preprocess

    if not os.path.isdir(input_dir_path):
        logger.error(f"입력 디렉토리 '{input_dir_path}'를 찾을 수 없거나 디렉토리가 아닙니다.")
        exit(1)

    if output_dir_path is None:
        output_dir_path = os.path.join(input_dir_path, "output_ocr_results")

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
    logger.info(f"전처리 수행: {not skip_preprocessing}")
    logger.info(f"결과 이미지 표시: {display_images_flag}")
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일을 처리합니다.")
    
    # 폰트 메시지 플래그 초기화 (여러 번 실행 시를 위함이지만, 스크립트가 한 번만 실행되므로 여기서는 큰 의미 없음)
    get_os_specific_font_path.font_message_printed = False


    for filename in tqdm(image_files_to_process, desc="OCR 처리 중", unit="개"):
        current_image_path = os.path.join(input_dir_path, filename)
        logger.debug(f"처리 시작: {filename}")

        processed_image_data = None
        ocr_input_data = current_image_path 

        if not skip_preprocessing:
            processed_image_data = preprocess_image_for_ocr(current_image_path)
            if processed_image_data is not None:
                ocr_input_data = processed_image_data 
            else:
                logger.warning(f"{filename} 전처리에 실패. 원본 이미지로 OCR 시도.")
        else:
            logger.info(f"{filename} 전처리 단계를 건너뜁니다.")

        extracted_data = extract_text_from_image(ocr_input_data, 
                                                 lang=selected_language, 
                                                 preprocess_flag=False, # 외부에서 전처리 또는 건너뜀
                                                 filename_for_log=filename)

        if extracted_data:
            logger.debug(f"추출된 텍스트 항목 수 ({filename}): {len(extracted_data)}")
            # for i, item in enumerate(extracted_data): # 너무 많은 로그를 유발할 수 있어 디버그 레벨에서만
            #     logger.debug(f"  {i+1}. 텍스트: \"{item['text']}\", 신뢰도: {item['confidence']:.2f}")
            
            display_ocr_result(current_image_path, extracted_data, output_dir_path, filename,
                               preprocessed_img=processed_image_data if not skip_preprocessing else None, 
                               show_image_flag=display_images_flag)
        else:
            logger.info(f"{filename}에서 텍스트를 추출하지 못했습니다.")
        
    logger.info(f"총 {len(image_files_to_process)}개의 이미지 파일 처리가 완료되었습니다.")
    logger.info("스크립트 실행 종료.")
    # 최종 폰트 관련 안내는 display_ocr_result 내부에서 처리됨
