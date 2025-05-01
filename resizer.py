# -*- coding: utf-8 -*-
import os
import sys
import importlib.util # 라이브러리 확인용

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
SCRIPT_VERSION = "1.4"

# Pillow 버전 호환성을 위한 리샘플링 필터 정의
try:
    RESAMPLE_FILTERS = {
        "1": Image.Resampling.LANCZOS,   # 고품질, 느림
        "2": Image.Resampling.BICUBIC,   # 중간 품질, 중간 속도
        "3": Image.Resampling.BILINEAR,  # 낮은 품질, 빠름
        "4": Image.Resampling.NEAREST    # 최저 품질, 가장 빠름 (픽셀 아트 등에 적합)
    }
    FILTER_NAMES = {
        "1": "LANCZOS (고품질)",
        "2": "BICUBIC (중간 품질)",
        "3": "BILINEAR (낮은 품질)",
        "4": "NEAREST (최저 품질)"
    }
except AttributeError:
    # 이전 Pillow 버전 호환성
    RESAMPLE_FILTERS = {
        "1": Image.LANCZOS,
        "2": Image.BICUBIC,
        "3": Image.BILINEAR,
        "4": Image.NEAREST
    }
    FILTER_NAMES = {
        "1": "LANCZOS (고품질)",
        "2": "BICUBIC (중간 품질)",
        "3": "BILINEAR (낮은 품질)",
        "4": "NEAREST (최저 품질)"
    }

# 지원하는 출력 포맷 정의
SUPPORTED_OUTPUT_FORMATS = {
    "1": "원본 유지",
    "2": "PNG",
    "3": "JPG",
    "4": "WEBP",
}

# 지원하는 입력 이미지 확장자
SUPPORTED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- 입력 처리 헬퍼 함수 ---

def get_validated_input(prompt, validation_func, error_message="오류: 유효하지 않은 입력입니다."):
    """ 사용자 입력을 받고 유효성을 검사하며, 중단 신호(Ctrl+C/D)를 처리합니다. """
    while True:
        try:
            user_input = input(prompt).strip()
            if validation_func(user_input):
                return user_input
            else:
                print(error_message)
        except (EOFError, KeyboardInterrupt):
             print("\n사용자 요청으로 스크립트를 중단합니다.")
             sys.exit(1) # 오류 코드로 종료

def get_positive_integer_input(prompt):
    """ 양의 정수 입력을 안전하게 받습니다. """
    return int(get_validated_input(
        prompt,
        lambda s: s.isdigit() and int(s) > 0,
        "   오류: 0보다 큰 정수를 숫자로 입력해야 합니다."
    ))

def get_integer_in_range_input(prompt, min_val, max_val, default_val=None):
    """ 지정된 범위 내의 정수 또는 기본값 입력을 안전하게 받습니다. """
    full_prompt = prompt
    if default_val is not None:
        full_prompt += f"(기본값: {default_val}): "
    else:
        full_prompt += ": "

    input_value = get_validated_input(
        full_prompt,
        lambda s: (s == "" and default_val is not None) or \
                  (s.isdigit() and min_val <= int(s) <= max_val),
        f"   오류: {min_val}에서 {max_val} 사이의 정수를 입력하거나" + \
        (f" 비워두세요(기본값 {default_val})." if default_val is not None else ".")
    )
    return int(input_value) if input_value else default_val # 입력이 비었으면 기본값 반환

def get_yes_no_input(prompt, default_yes=False):
    """ 'y' 또는 'n' 입력을 받습니다. """
    default_char = 'y' if default_yes else 'n'
    prompt_with_default = f"{prompt} (y/n, 기본값: {default_char}): "
    response = get_validated_input(
        prompt_with_default,
        lambda s: s.lower() in ['y', 'n', ''],
        "   오류: 'y' 또는 'n'을 입력하거나 비워두세요."
    )
    return (response.lower() == 'y') if response else default_yes

# --- 이미지 처리 핵심 함수 ---

def resize_image_maintain_aspect_ratio(img, max_width, max_height, resample_filter):
    """ 가로세로 비율을 유지하며 이미지 크기를 조절합니다. """
    original_width, original_height = img.size
    if original_width == 0 or original_height == 0: return img
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = max(1, int(original_width * ratio))
    new_height = max(1, int(original_height * ratio))
    if (new_width, new_height) == (original_width, original_height): return img
    return img.resize((new_width, new_height), resample_filter)

def resize_image_fixed_size(img, target_width, target_height, resample_filter):
    """ 지정된 크기로 이미지 크기를 강제 조절합니다. """
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

def prepare_image_for_save(img, output_format):
    """ 지정된 출력 형식에 맞게 이미지 모드를 변환합니다 (특히 JPG). """
    save_img = img
    if output_format == 'JPG':
        if img.mode in ('RGBA', 'LA', 'P', 'L'):
            print(f"   -> 정보: 이미지 모드({img.mode})를 JPG 저장을 위해 RGB로 변환합니다 (투명도/회색조 손실).")
            if img.mode == 'P' and 'transparency' in img.info:
                save_img = img.convert('RGBA')
            if save_img.mode in ('RGBA', 'LA'):
                background = Image.new("RGB", save_img.size, (255, 255, 255))
                try:
                    mask = save_img.split()[-1]
                    background.paste(save_img, mask=mask)
                    save_img = background
                except IndexError:
                    save_img = save_img.convert('RGB')
            else: # P(투명도 없는), L 모드
                save_img = save_img.convert('RGB')
    return save_img

def process_images(input_folder, output_folder, resize_options, output_format_options, process_recursive, preserve_exif):
    """ 지정된 폴더(및 하위 폴더)의 이미지들을 처리하고 결과를 요약합니다. """
    processed_count = 0
    error_count = 0
    skipped_files = []
    error_files = [] # 오류 발생 파일 목록 (파일명, 오류 메시지)

    # 처리할 파일 목록 생성 (os.walk 사용 여부 결정)
    files_to_process = [] # (input_path, relative_path) 튜플 저장
    try:
        if process_recursive:
            print(f"\n하위 폴더 포함하여 '{input_folder}' 탐색 중...")
            for root, _, files in os.walk(input_folder):
                for filename in files:
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        input_path = os.path.join(root, filename)
                        # 출력 폴더 구조 유지를 위한 상대 경로 계산
                        relative_path = os.path.relpath(input_path, input_folder)
                        files_to_process.append((input_path, relative_path))
                    else:
                         # 지원하지 않는 파일도 건너<0xEB><0x8><0xB7> 목록에 추가 (상대 경로 기준)
                         relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                         skipped_files.append(relative_path + " (미지원 형식)")
        else:
            print(f"\n'{input_folder}' 폴더 탐색 중...")
            for filename in os.listdir(input_folder):
                input_path = os.path.join(input_folder, filename)
                if os.path.isfile(input_path):
                    if filename.lower().endswith(SUPPORTED_INPUT_EXTENSIONS):
                        # 하위 폴더 미포함 시 relative_path는 그냥 filename
                        files_to_process.append((input_path, filename))
                    else:
                        skipped_files.append(filename + " (미지원 형식)")
                elif os.path.isdir(input_path):
                     skipped_files.append(filename + " (폴더)") # 최상위 폴더만 건너<0xEB><0x8><0xB7> 목록에 추가

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
        filename = os.path.basename(input_path) # 현재 파일명

        # 출력 경로 설정 (하위 폴더 구조 유지)
        output_relative_dir = os.path.dirname(relative_path)
        output_dir_for_file = os.path.join(output_folder, output_relative_dir)
        # 출력 하위 폴더 생성 (필요시)
        if not os.path.exists(output_dir_for_file):
            try:
                os.makedirs(output_dir_for_file)
            except OSError as e:
                error_msg = f"출력 하위 폴더 생성 실패: {output_dir_for_file} ({e})"
                print(f" {progress} ✗ 오류: {error_msg}")
                error_files.append((relative_path, error_msg))
                error_count += 1
                continue # 이 파일 처리 건너<0xEB><0x84>

        base_name, original_ext = os.path.splitext(filename)
        output_format = output_format_options['format']
        output_ext = ""

        if output_format == '원본 유지': output_ext = original_ext
        elif output_format == 'JPG': output_ext = '.jpg'
        elif output_format == 'WEBP': output_ext = '.webp'
        else: output_ext = f'.{output_format.lower()}' # PNG 등

        output_filename = base_name + output_ext
        output_path_base = os.path.join(output_dir_for_file, output_filename)
        output_path = get_unique_filepath(output_path_base) # 덮어쓰기 방지

        exif_data = None # EXIF 데이터 저장 변수 초기화

        try:
            # 이미지 열기
            with Image.open(input_path) as img:
                # EXIF 데이터 로드 시도 (옵션 활성화 시)
                if preserve_exif and 'exif' in img.info:
                    try:
                        exif_data = piexif.load(img.info['exif'])
                        # print(f"   -> 정보: '{filename}'에서 EXIF 데이터 로드 성공.")
                    except Exception as exif_err:
                        # piexif.InvalidImageDataError 등 다양한 오류 가능
                        print(f"   -> 경고: '{filename}' EXIF 데이터 로드 실패. 건너<0xEB><0x84>. ({exif_err})")
                        exif_data = None # 오류 시 EXIF 데이터 없음으로 처리

                # 출력 형식에 맞게 이미지 준비 (모드 변환 등)
                img_prepared = prepare_image_for_save(img, output_format)

                # 리사이즈
                resample_filter = resize_options['filter']
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
                save_format_arg = output_format if output_format != '원본 유지' else None
                if output_format == 'JPG':
                    save_kwargs['quality'] = output_format_options.get('quality', 95)
                    save_kwargs['optimize'] = True
                    save_kwargs['progressive'] = True
                    # EXIF 데이터 추가 (존재하고 옵션 활성화 시)
                    if exif_data:
                         try:
                              # Pillow는 바이트 형태의 EXIF 데이터를 요구
                              exif_bytes = piexif.dump(exif_data)
                              save_kwargs['exif'] = exif_bytes
                              # print(f"   -> 정보: '{filename}' 저장 시 EXIF 데이터 포함.")
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패. EXIF 없이 저장. ({dump_err})")
                elif output_format == 'PNG':
                    save_kwargs['optimize'] = True
                    # PNG는 EXIF 표준 지원이 약하지만, Pillow/piexif가 시도함
                    if exif_data:
                         try:
                              exif_bytes = piexif.dump(exif_data)
                              save_kwargs['exif'] = exif_bytes # Pillow 9.1.0+ 에서 PNG EXIF 지원 개선
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패 (PNG). EXIF 없이 저장. ({dump_err})")
                elif output_format == 'WEBP':
                    save_kwargs['quality'] = output_format_options.get('quality', 80)
                    save_kwargs['lossless'] = False
                    # WEBP도 EXIF 지원 (Pillow 7.1.0+)
                    if exif_data:
                         try:
                              exif_bytes = piexif.dump(exif_data)
                              save_kwargs['exif'] = exif_bytes
                         except Exception as dump_err:
                              print(f"   -> 경고: '{filename}' EXIF 데이터 변환 실패 (WEBP). EXIF 없이 저장. ({dump_err})")

                # 결과 이미지 저장
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
            if os.path.exists(output_path):
                  try: os.remove(output_path)
                  except OSError: pass
        except Exception as e:
            error_msg = f"예상치 못한 오류 ({e})"
            print(f" {progress} ✗ 오류: '{relative_path}' ({error_msg})")
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
    print(f"--- 이미지 크기 일괄 변경 스크립트 (v{SCRIPT_VERSION}) ---")

    # 1. 입력 폴더 경로 받기
    input_dir = get_validated_input(
        "1. 이미지가 있는 폴더 경로를 입력하세요: ",
        os.path.isdir,
        "   오류: 유효한 폴더 경로가 아닙니다. 다시 입력해주세요."
    )
    absolute_input_dir = os.path.abspath(input_dir)

    # 2. 하위 폴더 포함 여부
    process_recursive = get_yes_no_input("2. 하위 폴더의 이미지도 포함하여 처리하시겠습니까?", default_yes=False)

    # 3. 출력 폴더 경로 받기
    while True:
        output_dir_prompt = "3. 결과 저장 폴더 경로 (비워두면 원본 하위 'resized_images' 폴더): "
        output_dir_input = input(output_dir_prompt).strip()
        if not output_dir_input:
            output_dir = os.path.join(absolute_input_dir, "resized_images")
            print(f"   -> 출력 폴더를 '{output_dir}'로 자동 설정합니다.")
        else:
            output_dir = output_dir_input
        absolute_output_dir = os.path.abspath(output_dir)

        # 입력 폴더와 출력 폴더가 동일하거나 하위 관계인지 확인 (하위 폴더 처리 시 중요)
        if absolute_input_dir == absolute_output_dir:
            print("   (!) 오류: 입력 폴더와 출력 폴더는 동일할 수 없습니다.")
        # 출력 폴더가 입력 폴더의 하위 폴더인지 확인 (재귀 처리 시 무한 루프 방지)
        elif process_recursive and absolute_output_dir.startswith(absolute_input_dir + os.sep):
             print("   (!) 오류: 하위 폴더 포함 처리 시, 출력 폴더는 입력 폴더 내부에 지정할 수 없습니다.")
        else:
            break # 유효한 경로

    # 4. 리사이즈 모드 선택
    print("\n4. 리사이즈 방식 선택:")
    print("   1: 가로세로 비율 유지 (최대 너비/높이 내 맞춤)")
    print("   2: 지정된 크기로 강제 변경 (비율 왜곡 가능)")
    resize_mode_choice = get_validated_input(
        "   선택 (1 또는 2): ", lambda c: c in ["1", "2"], "   오류: 1 또는 2를 입력해야 합니다."
    )
    resize_opts = {'mode': 'aspect_ratio' if resize_mode_choice == "1" else 'fixed'}

    # 5. 리사이즈 크기 설정
    print(f"\n5. 리사이즈 크기 설정 ({'비율 유지' if resize_opts['mode'] == 'aspect_ratio' else '고정 크기'} 모드):")
    width_prompt = "   - 최대 너비(px): " if resize_opts['mode'] == 'aspect_ratio' else "   - 원하는 너비(px): "
    height_prompt = "   - 최대 높이(px): " if resize_opts['mode'] == 'aspect_ratio' else "   - 원하는 높이(px): "
    resize_opts['width'] = get_positive_integer_input(width_prompt)
    resize_opts['height'] = get_positive_integer_input(height_prompt)

    # 6. 리샘플링 필터 선택
    print("\n6. 리사이즈 필터(품질/속도) 선택:")
    for key, name in FILTER_NAMES.items(): print(f"   {key}: {name}")
    filter_choice = get_validated_input(
        f"   선택 ({', '.join(FILTER_NAMES.keys())}): ", lambda c: c in RESAMPLE_FILTERS,
        f"   오류: {', '.join(FILTER_NAMES.keys())} 중 하나를 입력해야 합니다."
    )
    resize_opts['filter'] = RESAMPLE_FILTERS[filter_choice]
    print(f"   -> 선택된 필터: {FILTER_NAMES[filter_choice]}")

    # 7. 출력 파일 형식 선택
    print("\n7. 저장할 파일 형식 선택:")
    for key, name in SUPPORTED_OUTPUT_FORMATS.items(): print(f"   {key}: {name}")
    format_choice = get_validated_input(
        f"   선택 ({', '.join(SUPPORTED_OUTPUT_FORMATS.keys())}): ", lambda c: c in SUPPORTED_OUTPUT_FORMATS,
        f"   오류: {', '.join(SUPPORTED_OUTPUT_FORMATS.keys())} 중 하나를 입력해야 합니다."
    )
    output_format_opts = {'format': SUPPORTED_OUTPUT_FORMATS[format_choice]}
    print(f"   -> 선택된 형식: {output_format_opts['format']}")

    # JPG 또는 WEBP 선택 시 품질 설정
    if output_format_opts['format'] in ('JPG', 'WEBP'):
        default_quality = 95 if output_format_opts['format'] == 'JPG' else 80
        quality = get_integer_in_range_input(
            f"   - {output_format_opts['format']} 품질 (1-100)", 1, 100, default_val=default_quality
        )
        output_format_opts['quality'] = quality
        print(f"   -> {output_format_opts['format']} 품질을 {quality}로 설정합니다.")

    # 8. EXIF 메타데이터 유지 여부
    preserve_exif = get_yes_no_input("8. EXIF 메타데이터(촬영 정보 등)를 유지하시겠습니까?", default_yes=False)


    # --- 최종 설정 확인 및 처리 시작 ---
    print("\n" + "="*30 + " 최종 설정 확인 " + "="*30)
    print(f"입력 폴더: {absolute_input_dir}")
    print(f"  하위 폴더 포함: {'예' if process_recursive else '아니오'}")
    print(f"출력 폴더: {absolute_output_dir}")
    print(f"리사이즈 방식: {'비율 유지' if resize_opts['mode'] == 'aspect_ratio' else '고정 크기'}")
    print(f"  크기 설정: {resize_opts['width']}x{resize_opts['height']} px")
    print(f"리사이즈 필터: {FILTER_NAMES[filter_choice]}")
    print(f"출력 형식: {output_format_opts['format']}")
    if 'quality' in output_format_opts: print(f"  품질: {output_format_opts['quality']}")
    print(f"EXIF 메타데이터 유지: {'예' if preserve_exif else '아니오'}")
    print("="*75)

    confirm = get_validated_input("\n(!) 설정을 확인했습니다. 이미지 처리를 시작하려면 'y'를 입력하세요 (다른 키는 취소): ", lambda s: True)
    if confirm.lower() == 'y':
        process_images(absolute_input_dir, absolute_output_dir, resize_opts, output_format_opts, process_recursive, preserve_exif)
        print("\n--- 모든 작업 완료 ---")
        print(f"결과는 '{absolute_output_dir}' 폴더에서 확인할 수 있습니다.")
    else:
        print("\n작업이 취소되었습니다.")

