import json
import os
from pathlib import Path

def create_coco_txt_files(coco_root_path, split='val2017', num_samples=None):
    """
    COCO 데이터셋에서 이미지 경로와 캡션 정보를 담은 txt 파일들을 생성합니다.
    
    Args:
        coco_root_path (str): COCO 데이터셋의 루트 경로
        split (str): 'train2017' 또는 'val2017'
        num_samples (int): 생성할 샘플 수 (None이면 전체)
    """
    
    # 경로 설정
    coco_path = Path(coco_root_path)
    images_path = coco_path / 'images' / split
    annotations_path = coco_path / 'annotations' / 'annotations' / f'captions_{split}.json'
    
    # annotation 파일 로드
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 이미지 ID와 파일명 매핑 생성
    image_id_to_filename = {}
    for image_info in coco_data['images']:
        image_id_to_filename[image_info['id']] = image_info['file_name']
    
    # 이미지 ID별로 캡션들을 그룹화
    image_captions = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # 실제 존재하는 이미지 파일들만 필터링
    valid_images = []
    for image_id, filename in image_id_to_filename.items():
        image_file_path = images_path / filename
        if image_file_path.exists() and image_id in image_captions:
            valid_images.append((image_id, filename))
    
    print(f"Found {len(valid_images)} valid images with captions")
    
    # 샘플 수 제한
    if num_samples and num_samples < len(valid_images):
        valid_images = valid_images[:num_samples]
        print(f"Limited to {num_samples} samples")
    
    # 출력 파일 경로 (data 폴더 안에 생성)
    data_path = Path('/root/stable-diffusion/data')
    data_path.mkdir(exist_ok=True)  # data 폴더가 없으면 생성
    images_txt_path = data_path / 'coco_images.txt'
    captions_txt_path = data_path / 'coco_txt.txt'
    
    # 파일 생성
    print(f"Creating {images_txt_path}...")
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        for image_id, filename in valid_images:
            # 상대 경로로 저장 (data/coco_images/ 형식)
            relative_path = f"../coco/images/{split}/{filename}"
            f.write(f"{relative_path}\n")
    
    print(f"Creating {captions_txt_path}...")
    with open(captions_txt_path, 'w', encoding='utf-8') as f:
        for image_id, filename in valid_images:
            # 첫 번째 캡션만 사용 (여러 캡션이 있는 경우)
            caption = image_captions[image_id][0]
            # 파일명에서 확장자 제거
            image_name = filename.replace('.jpg', '')
            f.write(f"caption of {image_name}.jpg\n")
    
    print(f"Successfully created:")
    print(f"  - {images_txt_path} ({len(valid_images)} entries)")
    print(f"  - {captions_txt_path} ({len(valid_images)} entries)")
    
    return len(valid_images)

def create_coco_txt_files_with_actual_captions(coco_root_path, split='val2017', num_samples=None):
    """
    실제 캡션 내용을 포함한 버전을 생성합니다.
    """
    
    # 경로 설정
    coco_path = Path(coco_root_path)
    images_path = coco_path / 'images' / split
    annotations_path = coco_path / 'annotations' / 'annotations' / f'captions_{split}.json'
    
    # annotation 파일 로드
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 이미지 ID와 파일명 매핑 생성
    image_id_to_filename = {}
    for image_info in coco_data['images']:
        image_id_to_filename[image_info['id']] = image_info['file_name']
    
    # 이미지 ID별로 캡션들을 그룹화
    image_captions = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # 실제 존재하는 이미지 파일들만 필터링
    valid_images = []
    for image_id, filename in image_id_to_filename.items():
        image_file_path = images_path / filename
        if image_file_path.exists() and image_id in image_captions:
            valid_images.append((image_id, filename))
    
    print(f"Found {len(valid_images)} valid images with captions")
    
    # 샘플 수 제한
    if num_samples and num_samples < len(valid_images):
        valid_images = valid_images[:num_samples]
        print(f"Limited to {num_samples} samples")
    
    # 출력 파일 경로 (실제 캡션 버전 - data 폴더 안에 생성)
    data_path = Path('/root/stable-diffusion/data')
    data_path.mkdir(exist_ok=True)  # data 폴더가 없으면 생성
    images_txt_path = data_path / 'coco_images_with_captions.txt'
    captions_txt_path = data_path / 'coco_captions_actual.txt'
    
    # 파일 생성
    print(f"Creating {images_txt_path}...")
    with open(images_txt_path, 'w', encoding='utf-8') as f:
        for image_id, filename in valid_images:
            relative_path = f"../coco/images/{split}/{filename}"
            f.write(f"{relative_path}\n")
    
    print(f"Creating {captions_txt_path}...")
    with open(captions_txt_path, 'w', encoding='utf-8') as f:
        for image_id, filename in valid_images:
            # 첫 번째 캡션 사용
            caption = image_captions[image_id][0]
            # 캡션 정리 (줄바꿈 제거, 특수문자 처리 등)
            caption = caption.strip().replace('\n', ' ').replace('\r', ' ')
            f.write(f"{caption}\n")
    
    print(f"Successfully created:")
    print(f"  - {images_txt_path} ({len(valid_images)} entries)")
    print(f"  - {captions_txt_path} ({len(valid_images)} entries)")
    
    return len(valid_images)

# 사용 예시
if __name__ == "__main__":
    # COCO 데이터셋 경로 설정
    coco_root = "/root/coco"  # 실제 경로로 설정
    
    print("=== 방법 1: 샘플 형식 (caption of 파일명.jpg) ===")
    # 샘플 데이터와 같은 형식으로 생성 (35개 샘플)
    #create_coco_txt_files(coco_root, split='val2017', num_samples=35)
    
    print("\n=== 방법 2: 실제 캡션 내용 포함 ===")
    # 실제 캡션 내용을 포함한 버전
    #create_coco_txt_files_with_actual_captions(coco_root, split='val2017', num_samples=35)
    
    print("\n=== 전체 데이터셋 처리 ===")
    # 전체 validation 세트 처리하고 싶다면:
    # create_coco_txt_files(coco_root, split='val2017')
    
    # 전체 training 세트 처리하고 싶다면:
    create_coco_txt_files_with_actual_captions(coco_root, split='val2017')