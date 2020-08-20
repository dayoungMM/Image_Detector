# 특정 디렉토리 내 파일 리스트 가져와서 csv 파일로 만들기
## 엑셀에서 편하게 0, 1 라벨링 하기 위함

# 방법1 : 현재폴더 기준으로 경로가 모두 나오게 write 됨
# import glob
# import pandas as pd

# path = './20200805_02/*JPG'
# folder_name = path.split("/")[1]

# file_list_img = glob.glob(path)
# file_df_img = pd.DataFrame(file_list_img)
# file_df_img.to_csv(f'./{folder_name}_list.csv',index=False)

# 방법2: 파일명만 저장됨
import os
import pandas as pd

path = './images/total_img/'
folder_name = path.split("/")[2]
file_list = os.listdir(path)
file_list_img = [file for file in file_list if file.endswith(".JPG")]
file_df_img = pd.DataFrame(file_list_img)
file_df_img.to_csv(f'./images/{folder_name}_list.csv',index=False)