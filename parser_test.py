import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch L2SKNet test")

    parser.add_argument("--model_name", default='L2SKNet_UNet', type=str,  # nargs='+',
                        help="model_name: 'L2SKNet_UNet', 'L2SKNet_FPN', "
                             "'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN'")
    parser.add_argument("--dataset_name", default='IRSTD-1K', type=str,  # nargs='+',
                        help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")
    parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
    parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
    parser.add_argument("--test_epo", type=str, default='best_miou', help="Number of epoch for test")

    opt = parser.parse_args()
    return opt