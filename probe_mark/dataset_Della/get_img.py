import os

import cv2
import numpy as np
import openpyxl

# import win32com.client as win32
from PIL import Image, ImageChops, ImageGrab


def create_folders(base_path, img_name_list):
    folder_paths = []

    for folder_name in img_name_list:
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        folder_paths.append(folder_path)
        print(f"資料夾 {folder_name} 已建立於 {folder_path}")

    return folder_paths


# def get_img(file_path, sheet_name, created_folders, img_name):
#     # 啟動Excel應用程式
#     excel = win32.Dispatch("Excel.Application")
#     workbook = excel.Workbooks.Open(os.path.abspath(file_path))
#     sheet = workbook.Sheets(sheet_name)

#     # 遍歷每個形狀並保存圖片到對應的資料夾
#     for i, shape in enumerate(sheet.Shapes):
#         if shape.Name.startswith("Picture"):
#             shape.Copy()
#             image = ImageGrab.grabclipboard()
#             if image:
#                 # 選擇對應資料夾來儲存圖片
#                 folder_index = i % len(created_folders)
#                 folder_path = created_folders[folder_index]
#                 image_path = os.path.join(folder_path, img_name + ".png")
#                 image.save(image_path)
#                 print(f"圖片已儲存到 {image_path}")

#     # 關閉Excel應用程式
#     excel.Quit()

#     print(f"Images have been extracted to folders in {os.path.dirname(file_path)}")


# 定義讀取和儲存圖片的靜態方法
def get_image(img_data):
    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


# 定義提取和儲存圖片的函數
def get_img(file_path, sheet_name, created_folders, img_name="img"):
    # 讀取Excel檔案
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook[sheet_name]

    # 提取圖片
    imgs_dict = {}
    for img_obj in sheet._images:
        row_idx = img_obj.anchor._from.row + 1
        img_name_in_sheet = sheet.cell(row_idx, 1).value

        if img_name_in_sheet not in imgs_dict:
            imgs_dict[img_name_in_sheet] = get_image(img_obj._data())

    # 檢查圖片數量和資料夾數量是否匹配
    if len(imgs_dict) != len(created_folders):
        raise ValueError("圖片數量和資料夾數量不匹配，請檢查資料來源。")

    # 儲存圖片
    for folder, (name, img) in zip(created_folders, imgs_dict.items()):
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{img_name}.png")
        cv2.imwrite(file_path, img)

    print("圖片提取和儲存完成。")


# # 使用範例
# folder_name = 'Artificial_VM_D80_MW120F-HS'
# file_path = os.path.join(folder_name, 'picture.xlsx')
# sheet_name = '主要'
# get_img(file_path, sheet_name, created_folders)


def make_folder(path, folder_name, date=None):
    if date:
        folder_name = f"{folder_name}_{date}"
    full_path = os.path.join(path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"已建立資料夾 {full_path}")
    else:
        print(f"資料夾已存在 {full_path}")
    return os.path.abspath(full_path)


def get_img_mask(image_path1, image_path2, save_path):
    try:
        # 讀取圖像並轉換為RGBA格式
        image1 = Image.open(image_path1).convert("RGBA")
        image2 = Image.open(image_path2).convert("RGBA")

        # 檢查兩個圖像的大小是否一致
        if image1.size != image2.size:
            raise ValueError("圖片大小不一致")

        # 將圖像轉換為numpy數組
        image1_array = np.array(image1)
        image2_array = np.array(image2)

        # 找到兩個圖像之間RGB不一樣的區域
        difference_array = np.any(
            image1_array[:, :, :3] != image2_array[:, :, :3], axis=-1
        )

        # 創建一個新圖像，將不同的區域變為白色，其他區域變為黑色
        highlight_image_array_black = np.where(
            difference_array[..., None], [255, 255, 255], [0, 0, 0]
        )

        # 將結果轉換回PIL圖像
        highlight_image_black = Image.fromarray(
            highlight_image_array_black.astype("uint8")
        )

        # 保存結果圖像
        highlight_image_black.save(save_path)
        print(f"Image saved to {save_path}")

    except ValueError as ve:
        print(f"{image_path1}和{image_path2}發生圖片大小不一致: {ve}")


def get_img_label(image_path1, image_path2, save_path):
    try:
        # 讀取圖像並轉換為RGBA格式
        image1 = Image.open(image_path1).convert("RGBA")
        image2 = Image.open(image_path2).convert("RGBA")

        # 檢查兩個圖像的大小是否一致
        if image1.size != image2.size:
            raise ValueError("圖片大小不一致")

        # 將圖像轉換為numpy數組
        image1_array = np.array(image1)
        image2_array = np.array(image2)

        # 找到兩個圖像之間RGB不一樣的區域
        difference_array = np.any(
            image1_array[:, :, :3] != image2_array[:, :, :3], axis=-1
        )

        # 創建一個新圖像，將不同的區域變為255，其他區域變為0
        label_image_array = np.where(difference_array, 1, 0).astype(np.uint8)

        # 將結果轉換回PIL圖像
        label_image = Image.fromarray(label_image_array, mode="L")

        # 保存結果圖像
        label_image.save(save_path)
        print(f"Image saved to {save_path}")

    except ValueError as ve:
        print(f"{image_path1}和{image_path2}發生圖片大小不一致: {ve}")


# for i in range(len(created_folders)):
#     image_path1 = created_folders[i] + '\\original.png'
#     image_path2 = created_folders[i] + '\\red_mark.png'
#     save_path = created_folders[i] + '\\label.png'
#     get_img_label(image_path1, image_path2, save_path)
