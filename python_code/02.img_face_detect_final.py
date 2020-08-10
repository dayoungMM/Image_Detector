#-*- coding: utf-8 -*- 
import os
import pathlib
import glob
import cv2
import settings

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 Path Pattern에 일치하는 파일 얻기
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지파일(파일명):{filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None:
            print(f"이미지파일({fullpath})을 읽을 수 없습니다.")
            continue
        name_images.append((filename, image))
        
    return name_images

def detect_image_face(file_path, image, cascade_filepath):
    # 이미지 파일의 Grayscale화
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert color
    # 캐스케이드 파일 읽기
    cascade = cv2.CascadeClassifier(cascade_filepath) #특징을 찾아내는 필터역할을 함
    print(file_path)
    # 얼굴인식
    faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1,minNeighbors=15, minSize=(128,128))
    if len(faces)==0:
        print("얼굴인식 실패")
        return
    print("얼굴인식 성공")

    # TO-DO 

    # 1개 이상의 얼굴인식
    # 이미지 찾아오면 좌표로 표시
    #[76 31 83 83] ->x_pos, y_pos, width, height
    face_count = 1
    for (x_pos, y_pos, width, height) in faces:
        face_image = image[y_pos:y_pos+height,x_pos:x_pos+width]  # y,x
        if face_image.shape[0] >128:
            face_image = cv2.resize(face_image, (128,128))
        print(face_image.shape)

        # 저장
        #00_001.jpg 에서 검출 -> 00_001_1.jpg
        #                     -> 00_001_2.jpg
        path = pathlib.Path(file_path)
        directory = str(path.parent.resolve())
        filename = path.stem
        extension = path.suffix

        output_path = os.path.join(directory,f"{filename}_{face_count:03}{extension}")
        print(f"출력파일(절대경로): {output_path}")
        try :
            cv2.imwrite(output_path, face_image)#이미지파일 생성
            face_count = face_count + 1
        except:
            print("Exception occured:{}".format(output_path))

    return RETURN_SUCCESS

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./origin_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_image"

def main():
    print("===================================================================")
    print("이미지 얼굴인식 OpenCV 이용")
    print("지정한 이미지 파일의 정면얼굴을 인식하고, 128x128 사이즈로 변경")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내의 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 이미지 파일 읽기
    # TO-DO 
    name_images =load_name_images(IMAGE_PATH_PATTERN)
    # 이미지별로 얼굴인식
    
    for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR,f"{name_image[0]}")
        image = name_image[1]
        cascade_filepath = settings.CASCADE_FILE_PATH
        detect_image_face(file_path, image, cascade_filepath)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()