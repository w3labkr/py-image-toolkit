# -*- coding: utf-8 -*-
import os
import sys
import importlib.util # 라이브러리 확인용
import argparse # 커맨드 라인 인자 처리용

# --- 라이브러리 확인 및 로드 ---
REQUIRED_LIBS = {
    "Pillow": "PIL",
    "piexif": "piexif" # EXIF 처리용
}

missing_libs = []
for package_name, import_name in REQUIRED_LIBS.items():
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        missing_libs.append(package_name)

if missing_libs:
    print("(!) 오류: 스크립트 실행에 필요한 라이브러리가 설치되어 있지 않습니다.")
    print("    아래 명령어를 사용하여 설치해주세요:")
    for lib in missing_libs:
        print(f"    pip install {lib}")
    sys.exit(1)

# 라이브러리 로드
from PIL import Image, UnidentifiedImageError
import piexif # EXIF 처리

# --- 상수 정의 ---
SCRIPT_VERSION = "1.5" # 버전 업데이트

# Pillow 버전 호환성을 위한 리샘플링 필터 정의
try:
    # Pillow 9.1.0 이상
    RESAMPLE_FILTERS = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST
    }
    FILTER_NAMES = {
        "lanczos": "LANCZOS (고품질)",
        "bicubic": "BICUBIC (중간 품질)",
        "bilinear": "BILINEAR (낮은 품질)",
        "nearest": "NEAREST (최저 품질)"
    }
except AttributeError:
    # 이전 Pillow 버전 호환성
    RESAMPLE_FILTERS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST
    }
    FILTER_NAMES = {
        "lanczos": "LANCZOS (고품질)",
        "bicubic": "BICUBIC (중간 품질)",
        "bilinear": "BILINEAR (낮은 품질)",
        "nearest": "NEAREST (최저 품질)"
    }

# 지원하는 출력 포맷 정의 (argparse에서 사용할 이름)
SUPPORTED_OUTPUT_FORMATS = {
    "original": "원본 유지",
    "png": "PNG",
    "jpg": "JPG",
    "webp": "WEBP",
}

# 지원하는 입력 이미지 확장자
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- 이미지 처리 핵심 함수 (기존과 거의 동일) ---

def resize_image_maintain_aspect_ratio(img, max_width, max_height, resample_filter):
    """ 가로세로 비율을 유지하며 이미지 크기를 조절합니다. """
    original_width, original_height = img.size
    if original_width == 0 or original_height == 0: return img
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = max(1, int(original_width * ratio))
    new_height = max(1, int(original_height * ratio))
    if (new_width, new_height) == (original_width, original_height): return img
    # 리사이즈 수행 전 필터 값 확인
    if not isinstance(resample_filter, int) and not hasattr(resample_filter, 'value'):
         # Pillow 10+ 대응: Resampling 열거형 객체 직접 사용
         # 이전 버전에서는 int 값이므로 이 조건 건너<0xEB><0x84>
         pass
    return img.resize((new_width, new_height), resample_filter)


def resize_image_fixed_size(img, target_width, target_height, resample_filter):
    """ 지정된 크기로 이미지 크기를 강제 조절합니다. """
    # 리사이즈 수행 전 필터 값 확인
    if not isinstance(resample_filter, int) and not hasattr(resample_filter, 'value'):
         # Pillow 10+ 대응: Resampling 열거형 객체 직접 사용
         pass
    return img.resize((target_width, target_height), resample_filter)


def get_unique_filepath(filepath):
    """ 파일 경로가 이미 존재하면 고유한 이름(숫자 추가)을 반환합니다. """
    if not os.path.exists(filepath): return filepath
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            print(f"   -> 정보: '{filename}' 파일이 이미 존재하여 새 이름 '{new_filename}'으로 저장합니다.")
            return new_filepath
        counter += 1

def prepare_image_for_save(img, output_format_str):
    """ 지정된 출력 형식(문자열)에 맞게 이미지 모드를 변환합니다 (특히 JPG). """
    save_img = img
    # output_format_str 은 'JPG', 'PNG' 등의 문자열
    if output_format_str == 'JPG':
        if img.mode in ('RGBA', 'LA', 'P', 'L'):
            print(f"   -> 정보: 이미지 모드({img.mode})를 JPG 저장을 위해 RGB로 변환합니다 (투명도/회색조 손실).")
            # 투명도가 있는 P 모드는 RGBA로 먼저 변환
            if img.mode == 'P' and 'transparency' in img.info:
                save_img = img.convert('RGBA')
            # RGBA 또는 LA 모드 처리
            if save_img.mode in ('RGBA', 'LA'):
                # 흰색 배경 생성
                background = Image.new("RGB", save_img.size, (255, 255, 255))
                try:
                    # 알파 채널을 마스크로 사용하여 배경 위에 붙여넣기
                    mask = save_img.split()[-1]
                    background.paste(save_img, mask=mask)
                    save_img = background
                except IndexError: # 알파 채널이 없는 경우 (LA 등)
                    save_img = save_img.convert('RGB') # 그냥 RGB로 변환
            else: # P(투명도 없는), L 모드
                save_img = save_img.convert('RGB')
    return save_img


def process_images(input_folder, output_folder, resize_options, output_format_options, process_recursive, preserve_exif):
    """ 지정된 폴더(및 하위 폴더)의 이미지들을 처리하고 결과를 요약합니다. """
    processed_count = 0
    error_count = 0
    skipped_files = []
    error_files = [] # 오류 발생 파일 목록 (파일명, 오류 메시지)

    # 처리할 파일 목록 생성
    files_to_process = [] # (input_path, relative_path) 튜플 저장
    try:
        if process_recursive:
            print(f"\n하위 폴더 포함하여 '{input_folder}' 탐색 중...")
            for root, _, files in os.walk(input_folder):
                # 출력 폴더 자체는 탐색에서 제외 (무한 루프 방지)
                if os.path.abspath(root).startswith(os.path.abspath(output_folder)):
                    continue
                for filename in files:
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        input_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(input_path, input_folder)
                        files_to_process.append((input_path, relative_path))
                    else:
                         relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                         skipped_files.append(relative_path + " (미지원 형식)")
        else:
            print(f"\n'{input_folder}' 폴더 탐색 중...")
            for filename in os.listdir(input_folder):
                input_path = os.path.join(input_folder, filename)
                if os.path.isfile(input_path):
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        files_to_process.append((input_path, filename))
                    else:
                        skipped_files.append(filename + " (미지원 형식)")
                elif os.path.isdir(input_path):
                    # 출력 폴더가 입력 폴더 바로 아래에 있을 경우 건너뛰기 목록에 추가
                    if os.path.abspath(input_path) == os.path.abspath(output_folder):
                         skipped_files.append(filename + " (출력 폴더)")
                    else:
                         skipped_files.append(filename + " (폴더)")

        total_files = len(files_to_process)
        if total_files == 0:
             print("(!) 처리할 이미지 파일이 지정된 경로에 없습니다.")
             return
    except Exception as e:
        print(f"(!) 치명적 오류: 입력 경로 '{input_folder}' 접근 실패. ({e})")
        return

    print(f"\n--- 총 {total_files}개의 이미지 처리 시작 ---")

    # 파일 처리 루프
    for i, (input_path, relative_path) in enumerate(files_to_process):
        progress = f"({i+1}/{total_files})"
        filename = os.path.basename(input_path)

        # 출력 경로 설정 (하위 폴더 구조 유지)
        output_relative_dir = os.path.dirname(relative_path)
        output_dir_for_file = os.path.join(output_folder, output_relative_dir)
        if not os.path.exists(output_dir_for_file):
            try:
                os.makedirs(output_dir_for_file)
            except OSError as e:
                error_msg = f"출력 하위 폴더 생성 실패: {output_dir_for_file} ({e})"
                print(f" {progress} ✗ 오류: {error_msg}")
                error_files.append((relative_path, error_msg))
                error_count += 1
                continue

        base_name, original_ext = os.path.splitext(filename)
        output_format_str = output_format_options['format_str'] # 'original', 'png', 'jpg', 'webp'
        output_ext = ""

        if output_format_str == 'original': output_ext = original_ext
        elif output_format_str == 'jpg': output_ext = '.jpg'
        elif output_format_str == 'webp': output_ext = '.webp'
        elif output_format_str == 'png': output_ext = '.png'
        else: output_ext = original_ext # 혹시 모를 예외 처리

        output_filename = base_name + output_ext
        output_path_base = os.path.join(output_dir_for_file, output_filename)
        output_path = get_unique_filepath(output_path_base)

        exif_data = None

        try:
            with Image.open(input_path) as img:
                # EXIF 데이터 로드
                if preserve_exif and 'exif' in img.info:
                    try:
                        # piexif는 bytes 형태의 EXIF 데이터를 처리
                        exif_data = piexif.load(img.info['exif'])
                    except Exception as exif_err:
                        print(f"   -> 경고: '{filename}' EXIF 데이터 로드 실패. 건너<0xEB><0x84>. ({exif_err})")
                        exif_data = None

                # 출력 형식에 맞게 이미지 준비 (JPG 변환 등)
                # prepare_image_for_save 함수는 대문자 형식 문자열('JPG', 'PNG' 등)을 기대
                img_prepared = prepare_image_for_save(img, output_format_str.upper() if output_format_str != 'original' else None)

                # 리사이즈
                resample_filter = resize_options['filter_obj'] # 실제 Pillow 필터 객체 사용
                if resize_options['mode'] == 'aspect_ratio':
                    img_resized = resize_image_maintain_aspect_ratio(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )
                else: # 'fixed'
                    img_resized = resize_image_fixed_size(
                        img_prepared, resize_options['width'], resize_options['height'], resample_filter
                    )

                # 저장 옵션 설정
                save_kwargs = {}
                # Pillow save() 함수의 format 인자는 대문자 형식을 선호 (예: 'JPEG', 'PNG')
                save_format_arg = None
                if output_format_str != 'original':
                    save_format_arg = output_format_str.upper()
                    # JPG는 JPEG로 지정해야 할 수 있음
                    if save_format_arg == 'JPG':
                         save_format_arg = 'JPEG'

                if output_format_str == 'jpg':
                    save_kwargs['quality'] = output_format_options.get('quality', 95)
                    save_kwargs['optimize'] = True
                    save_kwargs['progressive'] = True
                    if exif_data:
                         try:
                              exif_bytes = piexif.dump(exif_data)
                              save_kwargs['exif'] = exif_bytes
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패. EXIF 없이 저장. ({dump_err})")
                elif output_format_str == 'png':
                    save_kwargs['optimize'] = True
                    if exif_data:
                         try:
                              # Pillow 9.1.0+ 에서 PNG EXIF 지원 개선됨
                              # piexif.dump는 여전히 필요
                              exif_bytes = piexif.dump(exif_data)
                              # Pillow는 'pnginfo' 매개변수를 통해 EXIF를 저장할 수 있음
                              pnginfo = img.info.get('pnginfo')
                              if pnginfo:
                                   pnginfo.add_exif(exif_bytes)
                                   save_kwargs['pnginfo'] = pnginfo
                              else:
                                   # Pillow < 9.1.0 에서는 직접 지원하지 않을 수 있음
                                   # save_kwargs['exif'] = exif_bytes # 시도해볼 수 있으나 보장 안됨
                                   print(f"   -> 정보: '{filename}' PNG EXIF 저장은 Pillow 9.1.0+ 에서 더 잘 지원됩니다.")
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패 (PNG). EXIF 없이 저장. ({dump_err})")
                elif output_format_str == 'webp':
                    save_kwargs['quality'] = output_format_options.get('quality', 80)
                    save_kwargs['lossless'] = False # 기본적으로 손실 압축 사용
                    if exif_data:
                         try:
                              exif_bytes = piexif.dump(exif_data)
                              save_kwargs['exif'] = exif_bytes # Pillow 7.1.0+ 지원
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패 (WEBP). EXIF 없이 저장. ({dump_err})")

                # 결과 이미지 저장
                # format 인자가 None이면 Pillow가 확장자로부터 추측
                img_resized.save(output_path, format=save_format_arg, **save_kwargs)
                print(f" {progress} ✓ '{relative_path}' 처리 완료 -> '{os.path.relpath(output_path, output_folder)}'")
                processed_count += 1

        # --- 개별 파일 처리 오류 핸들링 ---
        except UnidentifiedImageError:
            error_msg = "유효하지 않거나 손상된 이미지 파일"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except PermissionError:
            error_msg = "파일 또는 출력 경로 접근 권한 부족"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
        except OSError as e:
            error_msg = f"파일 저장/처리 오류 ({e})"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            error_files.append((relative_path, error_msg))
            error_count += 1
            # 실패 시 생성되었을 수 있는 빈 파일 삭제 시도
            if os.path.exists(output_path):
                  try: os.remove(output_path)
                  except OSError: pass
        except Exception as e:
            # 예상치 못한 오류 발생 시 상세 정보 출력
            import traceback
            error_msg = f"예상치 못한 오류 ({type(e).__name__}: {e})"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
            # traceback.print_exc() # 디버깅 시 주석 해제
            error_files.append((relative_path, error_msg))
            error_count += 1


    # --- 최종 결과 요약 ---
    print(f"\n--- 처리 결과 요약 ---")
    print(f"총 시도한 이미지 수: {total_files}")
    print(f"성공적으로 처리된 이미지 수: {processed_count}")
    print(f"오류 발생 건수: {error_count}")
    if error_files:
        print("\n[오류 발생 파일 목록]")
        for filepath, errmsg in error_files:
            print(f"  - {filepath}: {errmsg}")

    if skipped_files:
        print(f"\n건너<0xEB><0x8><0xB7> 파일/폴더 수: {len(skipped_files)}")
        limit = 10
        if len(skipped_files) <= limit:
             print("  건너<0xEB><0x8><0xB7> 목록:", ", ".join(skipped_files))
        else:
             print(f"  건너<0xEB><0x8><0xB7> 목록 (처음 {limit}개):", ", ".join(skipped_files[:limit]), "...")
    else:
        print("\n건너<0xEB><0x8><0xB7> 파일/폴더 없음")


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"이미지 크기 일괄 변경 스크립트 (v{SCRIPT_VERSION})",
        formatter_class=argparse.RawTextHelpFormatter # help 메시지 줄바꿈 유지
    )

    # 필수 인자
    parser.add_argument("input_dir", help="이미지가 있는 원본 폴더 경로")
    parser.add_argument("--resize-mode", required=True, choices=['aspect_ratio', 'fixed'],
                        help="리사이즈 방식:\n"
                             "  aspect_ratio: 가로세로 비율 유지 (최대 너비/높이 내 맞춤)\n"
                             "  fixed: 지정된 크기로 강제 변경 (비율 왜곡 가능)")
    parser.add_argument("--width", required=True, type=int, help="리사이즈 너비 (px)")
    parser.add_argument("--height", required=True, type=int, help="리사이즈 높이 (px)")
    parser.add_argument("--filter", required=True, choices=FILTER_NAMES.keys(),
                        help="리사이즈 필터(품질/속도):\n" +
                             "\n".join([f"  {k}: {v}" for k, v in FILTER_NAMES.items()]))
    parser.add_argument("--output-format", required=True, choices=SUPPORTED_OUTPUT_FORMATS.keys(),
                        help="저장할 파일 형식:\n" +
                             "\n".join([f"  {k}: {v}" for k, v in SUPPORTED_OUTPUT_FORMATS.items()]))

    # 선택 인자
    parser.add_argument("--output-dir", help="결과 저장 폴더 경로 (기본값: 원본 폴더 하위의 'resized_images')")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더의 이미지도 포함하여 처리")
    parser.add_argument("--quality", type=int, default=None,
                        help="JPG 또는 WEBP 저장 품질 (1-100). 기본값: JPG=95, WEBP=80")
    parser.add_argument("--preserve-exif", action="store_true", help="EXIF 메타데이터(촬영 정보 등) 유지 시도")
    parser.add_argument('--version', action='version', version=f'%(prog)s {SCRIPT_VERSION}')


    args = parser.parse_args()

    # --- 인자 값 검증 및 설정 ---

    # 입력 폴더 검증
    if not os.path.isdir(args.input_dir):
        print(f"(!) 오류: 입력 폴더 경로가 유효하지 않습니다: {args.input_dir}")
        sys.exit(1)
    absolute_input_dir = os.path.abspath(args.input_dir)

    # 너비, 높이 검증
    if args.width <= 0 or args.height <= 0:
        print("(!) 오류: 너비와 높이는 0보다 큰 정수여야 합니다.")
        sys.exit(1)

    # 출력 폴더 설정 및 검증
    if args.output_dir:
        absolute_output_dir = os.path.abspath(args.output_dir)
    else:
        absolute_output_dir = os.path.join(absolute_input_dir, "resized_images")
        print(f"   -> 정보: 출력 폴더를 기본값으로 설정합니다: '{absolute_output_dir}'")

    # 입력/출력 폴더 충돌 검증
    if absolute_input_dir == absolute_output_dir:
        print("(!) 오류: 입력 폴더와 출력 폴더는 동일할 수 없습니다.")
        sys.exit(1)
    if args.recursive and absolute_output_dir.startswith(absolute_input_dir + os.sep):
         print("(!) 오류: 하위 폴더 포함(--recursive) 처리 시, 출력 폴더는 입력 폴더 내부에 지정할 수 없습니다.")
         sys.exit(1)

    # 출력 폴더 생성 (미리 생성)
    try:
        if not os.path.exists(absolute_output_dir):
            os.makedirs(absolute_output_dir)
            print(f"   -> 정보: 출력 폴더를 생성했습니다: '{absolute_output_dir}'")
    except OSError as e:
        print(f"(!) 오류: 출력 폴더를 생성할 수 없습니다: {absolute_output_dir} ({e})")
        sys.exit(1)

    # 리사이즈 옵션 설정
    resize_opts = {
        'mode': args.resize_mode,
        'width': args.width,
        'height': args.height,
        'filter_str': args.filter, # 사용자가 입력한 문자열 (로그용)
        'filter_obj': RESAMPLE_FILTERS[args.filter] # 실제 Pillow 필터 객체
    }

    # 출력 포맷 및 품질 설정
    output_format_opts = {
        'format_str': args.output_format # 사용자가 입력한 문자열 ('original', 'png', 'jpg', 'webp')
    }
    if args.output_format in ('jpg', 'webp'):
        default_quality = 95 if args.output_format == 'jpg' else 80
        quality = args.quality if args.quality is not None else default_quality
        if not (1 <= quality <= 100):
            print(f"(!) 오류: 품질 값은 1에서 100 사이여야 합니다 (입력값: {args.quality}).")
            sys.exit(1)
        output_format_opts['quality'] = quality
    elif args.quality is not None:
        print(f"   -> 경고: --quality 옵션은 'jpg' 또는 'webp' 형식에만 적용됩니다. 입력된 값({args.quality})은 무시됩니다.")


    # --- 최종 설정 확인 및 처리 시작 ---
    print("\n" + "="*30 + " 최종 설정 확인 " + "="*30)
    print(f"입력 폴더: {absolute_input_dir}")
    print(f"  하위 폴더 포함: {'예' if args.recursive else '아니오'}")
    print(f"출력 폴더: {absolute_output_dir}")
    print(f"리사이즈 방식: {'비율 유지' if resize_opts['mode'] == 'aspect_ratio' else '고정 크기'}")
    print(f"  크기 설정: {resize_opts['width']}x{resize_opts['height']} px")
    print(f"리사이즈 필터: {FILTER_NAMES[resize_opts['filter_str']]}") # 이름으로 출력
    print(f"출력 형식: {SUPPORTED_OUTPUT_FORMATS[output_format_opts['format_str']]}") # 이름으로 출력
    if 'quality' in output_format_opts: print(f"  품질: {output_format_opts['quality']}")
    print(f"EXIF 메타데이터 유지: {'예' if args.preserve_exif else '아니오'}")
    print("="*75)

    # 사용자 확인 없이 바로 처리 시작
    print("\n이미지 처리를 시작합니다...")
    process_images(
        absolute_input_dir,
        absolute_output_dir,
        resize_opts,
        output_format_opts,
        args.recursive,
        args.preserve_exif
    )

    print("\n--- 모든 작업 완료 ---")
    print(f"결과는 '{absolute_output_dir}' 폴더에서 확인할 수 있습니다.")