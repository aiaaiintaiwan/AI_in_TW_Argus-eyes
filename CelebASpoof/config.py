import os
import torch

DATASET_FOLDER = "../../../CelebA_Data/Data/train/"
# DATASET_FOLDER = "../../../CelebA_Spoof/Data/test/"

# training argument
GPU_ID = [0]
INPUT_SIZE = 224  
WARMUP_LR = 0.0
WARM_ITER = 2000
LR = 1e-3  # 学习率
BS = 200
PRE_EPOCH = 0
EPOCH = 150
IMG_PATH = "../../CelebA_Data/Data/trainAligned"
TXT_PATH = "../../CelebA_Data/metas/intra_test/train_label.txt"
TMP_TXT_PATH = "../../CelebA_Data/metas/intra_test/train_label_tmp.txt"

# testing arguments
TEST_IMG_PATH = "../../CelebA_Data/Data/testAligned/"
TEST_TXT_PATH = "../../CelebA_Data/metas/intra_test/test_label.txt"
TEST_BS = 512
TEST_NUM = 2048
TEST_ERR_IMG = "./txt/misclassified.txt"

# model setting
HEAD_NAME = "ArcFace"
MODEL_PATH = "./model/efficientModel"
IR_50_MODEL_PATH = "./model/ir50Model"
NET_OUT_FEATURES = 1000
ENSEMBLE_OUT_FEATURES = 2000

# training logs
PLOT_PATH = os.path.sep.join(["output", "plot.png"])
ACC_TXT_PATH = "./txt/acc.txt"
LOG_TXT_PATH = "./txt/log.txt"
BEST_TXT_PATH = "./txt/best_acc.txt"

# fine tuning Face.Evolve
BACKBONE_NAME = "IR_50"
BACKBONE_RESUME_ROOT = './backbone/backbone_ir50_ms1m_epoch63.pth' # the root to resume training from a saved checkpoint
HEAD_RESUME_ROOT = './' # the root to resume training from a saved checkpoint
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MULTI_GPU = True
FEATURE_EXTRACT = True