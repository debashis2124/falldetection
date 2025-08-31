from upfall_do.run import run_all_modalities, plot_all_metrics
import os

if __name__ == "__main__":
    out_dir = "/Users/debashis/Desktop/falldetection/outputs"

    modalities = ["all_sensors","imu_only","eeg","infrared"]

    results = run_all_modalities(modality_keys=modalities, out_dir=out_dir)

    # Create plots for accuracy, precision, recall, f1, specificity, roc_auc, loss
    plot_all_metrics(results, out_dir)



# from upfall_do.run import run_all_modalities, plot_all_metrics
# import os

# if __name__ == "__main__":
#     out_dir = "/Users/debashis/Desktop/falldetection/outputs"

#     modalities = ["all_sensors","imu_only","eeg","infrared"]

#     results = run_all_modalities(modality_keys=modalities, out_dir=out_dir)

#     plot_all_metrics(results, out_dir)
