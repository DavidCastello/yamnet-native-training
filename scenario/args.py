import argparse


def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Sound_Classification')

    # params for downloading esc-50 dataset
    parser.add_argument('--dataset_name', type=str, default='esc-50')
    parser.add_argument('--metadata_path', type=str,
                        default='./raw-dataset-esc50/ESC-50-master/meta/esc50.csv')
    parser.add_argument('--data_path', type=str,
                        default='./raw-dataset-esc50/ESC-50-master/audio')
    
    # params for the working dir
    parser.add_argument('--dataset_path', type=str,
                        default='./dataset')

    # params for preparing dataset
    parser.add_argument('--train_dir', type=str,
                        default='./dataset/train')
    parser.add_argument('--test_dir', type=str,
                        default='./dataset/test')
    parser.add_argument('--test_data_ratio', type=float, default=0.2)
    parser.add_argument('--length_audio', type=int, default=5)

    # params for training model
    parser.add_argument('--train_data_ratio', type=float, default=0.8)
    parser.add_argument('--tflite_file_name', type=str,
                        default='model.tflite')
    parser.add_argument('--save_path', type=str, default='./model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)

    # params for scenario
    parser.add_argument('--scenario', type=str, default='train')

    args = parser.parse_args()

    return args
