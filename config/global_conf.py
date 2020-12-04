from l5kit.configs import load_config_data
import torch

class Global:
    DIR_INPUT = "/home/gowithrobo/prediction_lyft"
    SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/kaggle_summary/output/result/single_mode_sample_submission.csv"
    MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/kaggle_summary/output/result/multi_mode_sample_submission.csv"
    DEBUG = False
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def getConfig(name="topk"):
        if name == "topk":
            return load_config_data(f"{Global.DIR_INPUT}/kaggle_summary/config/topk_config.yaml")
        else:
            return "Error"

    @staticmethod
    def load_weight(model, W_PATH=None): 
        if W_PATH is not None:
            WEIGHT_FILE = f"{Global.DIR_INPUT}/kaggle_summary/output/param/{W_PATH}"
            model_state = torch.load(WEIGHT_FILE, map_location=Global.DEVICE)
            model.load_state_dict(model_state)