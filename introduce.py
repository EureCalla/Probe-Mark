class Introduce:
    def __init__(self):
        self.program = "Probe-Mark"
        self.version = "v2.19"
        self.developer = "Calla.Lin #6381"
        self.program_intro = (
            "探針痕跡影像辨識系統，使用 segmentation_models.pytorch（ResNet+FPN 等）"
            "進行二元分類與語意分割訓練。"
        )
        self.program_notice = (
            "需要 PyTorch 環境；訓練資料請放於 data/ 目錄；"
            "執行前確認 requirements.txt 已安裝完畢。"
        )
