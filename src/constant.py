import os

class Path:

    DATA_PATH = "data"
    TRAIN_PATH = os.path.join(DATA_PATH, "seg_pred")
    TEST_PATH = os.path.join(DATA_PATH, "seg_test")
    PRED_PATH = os.path.join(DATA_PATH, "seg_pred")

    CONFIG = "config.yaml"

    MODEL_PATH = "model"
    MODEL_MOBILE = os.path.join(MODEL_PATH, "model_mobile.h5")
    MODEL_B0 = os.path.join(MODEL_PATH, "model_b0.h5")
    MODEL_B7 = os.path.join(MODEL_PATH, "model_b7.h5")

    LABEL = "src/labels.pkl"
    



