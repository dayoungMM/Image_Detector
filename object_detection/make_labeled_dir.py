
import glob
import shutil
import os 
import pandas as pd
import re

# csv파일에 0열에는 파일경로가, 1열에는 라벨을 달았다면 불러오자
data=pd.read_csv('./20200805_02_labeled.csv')
print(data.head())

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


data.apply(read_label, axis=1)