import base64
import io
import cv2
import keras
import numpy as np  
from PIL import Image
from keras.backend import tensorflow_backend as backend
from django.conf import settings

def detect(upload_image):
    result_name = upload_image.name 
    result_list = []
    result_img =''

    cascade_file_path = settings.CASCADE_FILE_PATH  
    model_file_path = settings.MODEL_FILE_PATH  
    model = keras.models.load_model(model_file_path)
    image = np.asarray(Image.open(upload_image))

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gs = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
    #1)cascade 사용하기 위한 CascadeClassifier 생성
    cascade = cv2.CascadeClassifier(cascade_file_path)
    
    
    #2)OpenCV 이용해서 얼굴 인식 함수 호출 -> detectMultiScale()
    faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))
    #3)얼굴 인식 개수는?
    if len(faces)>0:
        count = 1
        for (xpos, ypos, width, height) in faces:
            face_image = image[ypos:ypos+height,xpos:xpos+width]  # y,x
        
            #64보다 작으면 무시
            if face_image.shape[0] < 64 or face_image.shape[1]<64:
                print("인식한 얼굴의 사이즈가 너무 작습니다")
                continue

            else:
                #64보다 크면 64로 resize
                face_image = cv2.resize(face_image, (64,64))
                #붉은색 사각형 표시

                #detect with face_img ->detect_who()
                cv2.rectangle(image_RGB,(xpos,ypos), (xpos+width,ypos+height),(255,0,0), thickness=2)
                print(face_image.shape)
                print(np.expand_dims(face_image, axis=0).shape)
                face_image = np.expand_dims(face_image, axis=0) #차원 늘려주기

                name ,result = detect_who(model, face_image)
                #인식된 얼굴의 이름 표시
                cv2.putText(image_RGB, name, (xpos, ypos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
                result_list.append(result)
                count = count+1
        #이미지 PNG 파일로 변환
    is_success, img_buffer = cv2.imencode(".png",image_RGB)
    if is_success:
        # 이미지: 메인 메로리의 바이너리 형태로 들어가 있는 형태 -> base 64로 변환 (문자형태로 정보 보내주고 받으니깐)
        io_buffer = io.BytesIO(img_buffer)
        result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'","")
    
    #tensorflow 에서 session() 닫히지 않는 문제 방지하기 위해
    backend.clear_session()



    return (result_list, result_name, result_img)





    return render(req, 'mydetector/index.html', self.params)

def detect_who(model, face_image):
    # 예측
    result = model.predict(face_image) #이미지는 배열형태여야함
    result_msg = f"송혜교일 가능성:{result[0][0]*100: .3f}% / 전지현일 가능성:{result[0][1]*100: .3f}%" #softmax한 값이라 *100
    name_number_label = np.argmax(result)
    if name_number_label == 0 :
        name = "SONG"
    else:
        name = "JEON"


    return (name, result_msg)
