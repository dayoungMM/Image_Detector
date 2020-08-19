
import glob
import shutil
import os 
import pandas as pd
import re
from sklearn.utils import shuffle

# csv파일에 0열에는 파일경로가, 1열에는 라벨을 달았다면 불러오자
data=pd.read_csv('./images/total_img_list.csv')
# print(data.head())

# {라벨}_파일명 으로 이름을 바꾸고 labeled 폴더에 넣어보자
def read_label(x):
    if x[1]==1:
        label = "01"
    else:
        label = "00"
    # 정규식으로 파일명만 찾아내기
    m = re.search('.*\/(.+?)\.JPG',x[0])
    filename = m.group(1)
    new_filename = x[0].replace(filename, f"{label}_"+filename)
    print(new_filename)
    os.rename(x[0],new_filename)
    shutil.move(new_filename,'./labeled/')


# train, test 폴더로 이동하고 y 데이터 만들기
def classify_folder(x,origin_dir,new_dir):

    if x[1]==1:
        shutil.copy(origin_dir+x[0],new_dir+"1/")
    else:
        shutil.copy(origin_dir+x[0],new_dir+"0/")

## shuffle 해서 random으로 train test 나누기
def train_test(df, test_rate=0.3):

    #clear folder
    try:
        shutil.rmtree('./images/test/')
        os.mkdir('./images/test/')
        os.mkdir('./images/test/1')
        os.mkdir('./images/test/0')
        shutil.rmtree('./images/train/')
        os.mkdir('./images/train/')
        os.mkdir('./images/train/1')
        os.mkdir('./images/train/0')
    except:
        pass
    test_idx = round(len(df)*0.3)
    print(test_idx)
    df = shuffle(df)
    test_df = df.iloc[:test_idx+1]
    test_df.apply(lambda x: classify_folder(x,'./images/total_img/','./images/test/'),axis=1)
    test_df.to_csv('./images/test/test.csv')

    train_df = df.iloc[test_idx:]
    train_df.apply(lambda x: classify_folder(x,'./images/total_img/','./images/train/'),axis=1)
    train_df.to_csv('./images/test/train.csv')

    print(f"테스트 {len(test_df)}개 트레인 {len(train_df)}개")
    
    

train_test(data)
# data.apply(read_label, axis=1)