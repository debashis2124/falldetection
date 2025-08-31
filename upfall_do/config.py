
import os
import torch
import itertools


# Paths
DATA_ROOT = os.environ.get("UPFALL_DATA_ROOT", "/Users/debashis/Desktop/falldetection/UP_Fall_Detection_Dataset")
OUT_DIR   = os.environ.get("UPFALL_OUT_DIR",   "/Users/debashis/Desktop/falldetection/outputs_deep_only")

# Windowing
SEQ_LEN = 16
STEP    = 8

# Training
EPOCHS      = 10
ROUNDS      = 20
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Activities
FALL_ACTIVITIES = {'A06', 'A07', 'A08', 'A09', 'A10', 'A11'}
ADL_ACTIVITIES  = {'A01', 'A02', 'A03', 'A04', 'A05'}

# Columns / Modalities
ALL_COLUMNS = [
    "time", "helmet_raw",
    "belt_acc_x","belt_acc_y","belt_acc_z",
    "belt_ang_x","belt_ang_y","belt_ang_z","belt_luminosity",
    "neck_acc_x","neck_acc_y","neck_acc_z",
    "neck_ang_x","neck_ang_y","neck_ang_z","neck_luminosity",
    "pckt_acc_x","pckt_acc_y","pckt_acc_z",
    "pckt_ang_x","pckt_ang_y","pckt_ang_z","pckt_luminosity",
    "wrst_acc_x","wrst_acc_y","wrst_acc_z",
    "wrst_ang_x","wrst_ang_y","wrst_ang_z","wrst_luminosity",
    "ir_1","ir_2","ir_3","ir_4"
]



MODALITIES = {
    'eeg': ["helmet_raw"],
    'belt': ["belt_acc_x","belt_acc_y","belt_acc_z",
             "belt_ang_x","belt_ang_y","belt_ang_z","belt_luminosity"],
    'neck': ["neck_acc_x","neck_acc_y","neck_acc_z",
             "neck_ang_x","neck_ang_y","neck_ang_z","neck_luminosity"],
    'pocket': ["pckt_acc_x","pckt_acc_y","pckt_acc_z",
               "pckt_ang_x","pckt_ang_y","pckt_ang_z","pckt_luminosity"],
    'wrist': ["wrst_acc_x","wrst_acc_y","wrst_acc_z",
              "wrst_ang_x","wrst_ang_y","wrst_ang_z","wrst_luminosity"],
    'infrared': ["ir_1","ir_2","ir_3","ir_4"],
}


COMBINATIONS = {
    'all_sensors': list(set(itertools.chain.from_iterable(MODALITIES.values()))),
    'imu_only': list(set(MODALITIES['belt'] + MODALITIES['neck'] + MODALITIES['pocket'] + MODALITIES['wrist'])),
    'imu_plus_ir': list(set(MODALITIES['belt'] + MODALITIES['neck'] + MODALITIES['pocket'] + MODALITIES['wrist'] + MODALITIES['infrared'])),
    'eeg_plus_imu': list(set(MODALITIES['eeg'] + MODALITIES['belt'] + MODALITIES['neck'] + MODALITIES['pocket'] + MODALITIES['wrist'])),
}
