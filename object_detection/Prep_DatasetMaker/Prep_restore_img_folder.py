## 폴더가 다 날아갔을 때, csv 파일 리스트에 있는대로 복구하는 파일
## csv 파일 리스트에 있으면 남기고 없으면 삭제

import pandas as pd
import glob
import re 
import os



### 파일 리스트 가져오는 다른 방법
# import os
# path = './20200805_02'
# file_list = os.listdir(path)


#목록과 디렉토리내 파일 리스트 불러오기
data=pd.read_csv('./20200805_02_labeled.csv')
file_list = glob.glob('./20200805_02/*.JPG')


### 유용한 이미지만 남기고 삭제
# useful_data = list(data.iloc[:,0])
# for f in file_list:
#     if f in useful_data:
        
#         print(f)
#         print(data[data.iloc[:,0]==f].iloc[:,0])
#     else:
#         os.remove(f)

# file_list = glob.glob('./20200805_02/*.JPG')


## 확인
print(data.tail(1))
print(len(file_list))
