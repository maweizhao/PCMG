

import os


def log_out_message(path,log_info):
    
    log_date_path=path+"train.txt"
    #if not os.path.isfile(log_date_path):  # 无文件时创建
    fd=open(log_date_path, mode="a", encoding="utf-8")
    fd.write(log_info+"\n")
    fd.close()
        
    print(log_info)