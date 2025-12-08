import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
    
    
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

folder_id = "1QZ-Z-C9MoLk1S9MovF3Tt6NjAEGbSJI3"

file_list = drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()
save_dir_vital = os.path.join("Segment_Files", "PulseDB_Vital")
os.makedirs(save_dir_vital, exist_ok=True)

for file in file_list:
    filename = file['title']
    filepath = os.path.join(save_dir_vital, filename)
    file.GetContentFile(filepath)

# folder_id = "1-X6832Gs2CViMofq_jJLKgzXNX-Sx8Oq"

# file_list = drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()
# save_dir_MIMIC = os.path.join("Segment_Files", "PulseDB_MIMIC")
# os.makedirs(save_dir_MIMIC, exist_ok=True)

# for file in file_list:
#     filename = file['title']
#     filepath = os.path.join(save_dir_MIMIC, filename)
#     file.GetContentFile(filepath)

