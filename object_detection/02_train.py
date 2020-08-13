# import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.utils.vis_utils import plot_model
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import pickle


def label(x):
    #label data
    y_data.apend(x[1])
    #image data
    image = image = cv2.imread(x[0])
    # 이미지를 BGR 형식에서 RGB 형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지 배열(RGB 이미지)
    x_data.append(image)


# 파일 불러오기
def load_label_images(csv_path):
    data = pd.read_csv(csv_path)
    print(data.head())

    #x,y 데이터 만들기
    X = []
    y = []
    data.apply(label,axis=1)

    return (X, Y)
    

def main():
    #argument 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',default='./images/20200805_02_labeled.csv')
    parser.add_argument('--OUTPUT_MODEL_DIR',default='./model/')


    args = parser.parse_args()
    csv_path = args.csv_path
    OUTPUT_MODEL_DIR=args.OUTPUT_MODEL_DIR

    #X,Y 데이터 만들기
    X, Y=load_label_images(csv_path)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
    
    #test data 따로 보관하기 
    os.remove('X_test.bin')
    os.remove(('Y_test.bin'))
    with open('X_test.bin','wb') as f:
        pickle.dump(X_test,f)
    with open('Y_test.bin','wb') as f:
        pickle.dump(Y_test,f)

    ##LOAD 할때는 이렇게
    # with open('X_test.bin','rb') as f:
    #     pickle.load(f)
    # with open('Y_test.bin','rb') as f:
    #     pickle.load(f)

    # 모델정의
    model = Sequential()
    #CNN-1
    model.add(Conv2D(
        input_shape=(64,64,3),
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation='relu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #cnn-2
    model.add(Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation='relu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.02))
    #cnn-3
    model.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        activation='relu',
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.02))
    # fully-connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.summary()

    #compile
    model.compile(optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics= ['accuracy'],
    )

    plot_file_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_PLOT_FILE)
    plot_model(model, to_file=plot_file_path, show_shapes=True)
    history = model.fit(x_train, y_train, 
                    batch_size=batch_size, epochs=epochs, 
                    validation_data=(x_test, y_test),verbose=1)

    test_loss, test_acc = model.evaluate(x_train, y_train,batch_size=batch_size,verbose=0)
    print(f"validation loss:{test_loss}")
    print(f"validation accuracy:{test_acc}")
    # 모델 저장
    model_file_path = os.path.join(OUTPUT_MODEL_DIR, 'model.h5')
    model.save(model_file_path)

    


if __name__ == '__main__':
    main()