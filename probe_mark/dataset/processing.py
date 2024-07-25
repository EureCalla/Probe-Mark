import os

import cv2
import numpy as np
import openpyxl


class ImageExtractor:
    def __init__(self, data_folder, output_folder):
        # 定義工作表名稱與圖像類型的對應關係，避免opencv 無法存中文名稱的問題
        self.sheets = {"主要": "image", "體積面積量測": "label"}

        self.data_folder = data_folder
        self.folder_name = os.path.basename(data_folder)  # 獲取型號名稱

        self.file_path = os.path.join(data_folder, "picture.xlsx")  # 圖片excel檔案
        self.output_folder = os.path.join(output_folder, self.folder_name)
        # 載入Excel工作簿
        self.workbook = openpyxl.load_workbook(self.file_path, data_only=True)
        self.data_dict = self.extract_images()  # 提取圖片資料
        self.data_dict = self.delete_specific_data(self.data_dict)
        self.data_dict = self.check_data(self.data_dict)  # 檢查資料是否完整
        self.data_dict = self.export_segmentation_masks(self.data_dict)  # 產生分割遮罩
        self.save_images(self.data_dict)  # 儲存圖片

    def extract_images(self):
        data_dict = {}
        for sheet_name, img_type in self.sheets.items():
            sheet = self.workbook[sheet_name]
            for img_obj in sheet._images:
                row_idx = img_obj.anchor._from.row + 1  # 標題列有兩列，所以要加1
                img_name = sheet.cell(row_idx, 1).value  # 圖片名稱

                # 如果圖片名稱不存在，則新增一個字典
                if img_name not in data_dict:
                    data_dict[img_name] = {}
                data_dict[img_name][img_type] = self.get_image(img_obj._data())
        return data_dict

    @staticmethod
    def get_image(img_data):
        # 將圖像數據轉換成OpenCV可識別的格式
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def delete_specific_data(self, data_dict):
        # 刪除指定資料夾和圖片
        if self.folder_name == "Artificial_VM_D80_MW120F-HS":
            del data_dict["125-OD100-1"]
            del data_dict["125-OD100-2"]
            del data_dict["125-OD100-3"]

        return data_dict

    def check_data(self, data_dict):
        # 檢查每個圖像是否完整，如果不完整則刪除
        to_delete = []
        for img_name, img_data in data_dict.items():
            is_complete = True
            for img_type in self.sheets.values():
                if img_type not in img_data:
                    print(f"{img_type} is missing for {img_name}")
                    is_complete = False
            if not is_complete:
                to_delete.append(img_name)

        for img_name in to_delete:
            del data_dict[img_name]

        return data_dict

    def export_segmentation_masks(self, data_dict):
        # 產生圖像間的差異圖，用於語義分割的標籤
        for img_name, img_data in data_dict.items():
            keys = list(self.sheets.values())
            img_1 = img_data[keys[0]]
            img_2 = img_data[keys[1]]

            difference = cv2.absdiff(img_1, img_2)
            gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            img_data["ground_truth"] = cv2.threshold(
                gray_diff, 1, 1, cv2.THRESH_BINARY
            )[1]
            img_data["ground_truth_view"] = cv2.threshold(
                gray_diff, 1, 255, cv2.THRESH_BINARY
            )[1]
        return data_dict

    def save_images(self, data_dict):
        # 將所有圖像和產生的二值化圖像保存到指定文件夾
        os.makedirs(self.output_folder, exist_ok=True)
        for img_name, img_data in data_dict.items():
            folder_path = os.path.join(self.output_folder, img_name)
            os.makedirs(folder_path, exist_ok=True)
            for img_type, img in img_data.items():
                img_path = os.path.join(folder_path, f"{img_type}.png")
                cv2.imwrite(img_path, img)


if __name__ == "__main__":
    raw_data_folder = "data/raw"
    processed_folder = "data/processed"

    for folder in os.listdir(raw_data_folder):
        data_folder = os.path.join(raw_data_folder, folder)
        if os.path.isdir(data_folder):  # 確認該路徑是否為目錄
            extractor = ImageExtractor(
                data_folder=data_folder, output_folder=processed_folder
            )
