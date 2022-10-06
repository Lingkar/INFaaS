import argparse
import os
import subprocess
import time


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir', type=str,
                    help='Path to the directory of the to be profiled models')
    ap.add_argument('data_set', type=str,
                    help='Name of the dataset used')
    ap.add_argument('task', type=str,
                    help='The task performed (e.g. classification)')
    ap.add_argument('im_dim', type=str,
                    help='The dimension of the input image')
    ap.add_argument('num_cpus', type=str,
                    help='Amount of cpus available to inference container')
    return ap.parse_args()


def find_path_file(path, file_name) -> list[str]:
    if not path[-1] == '/':  # Make sure the path ends with a slash
        path = path + '/'

    if not os.path.isfile(os.path.join(path, file_name)):
        raise Exception(f"File not found! {path}{file_name}")
    file_path = path + file_name
    return file_path


def find_model_files(path) -> list[str]:
    if not os.path.exists(path):
        raise Exception(f"Directory {path} not found!")
    if not path[-1] == '/':  # Make sure the path ends with a slash
        path = path + '/'

    file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return file_names


def get_accuracy(file_name: str):
    parts = file_name.split('_acc')
    if not len(parts) == 2 or not parts[1].endswith('.pt'):
        raise f"File name {file_name} incorrectly formatted"
    accuracy = parts[1].strip('.pt').strip('acc')
    return accuracy


def get_par_model(file_name: str):
    parts = file_name.split('_')
    return parts[0]


def main(args):
    path = args.model_dir

    model_files = find_model_files(path)
    for file_name in model_files:
        print(f"Running bash process for {file_name}")
        subprocess.call(
            ['./profile_model_torch.sh', find_path_file(path, file_name), get_accuracy(file_name), args.data_set,
             args.task, args.num_cpus, get_par_model(file_name), path, args.im_dim])
        print(f"Done for file {file_name}")
        time.sleep(4)


if __name__ == '__main__':
    main(get_args())
