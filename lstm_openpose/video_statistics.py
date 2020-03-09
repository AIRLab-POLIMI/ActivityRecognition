import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# import pose_estimation.build.human_pose_estimation_demo.python.chpe as chpe

ROUND_DECIMALS = False

START_MEASURING_FRAME = 300
REPETITIONS = 20

STATISTICS_FOLDER = 'video_statistics'

HPE_MODEL = './pose_estimation/human_pose_estimation_demo/intel_models/human-pose-estimation-0001/FP32/' \
            'human-pose-estimation-0001.xml'
TARGET_DEVICE = 'CPU'
SHOW_KEYPOINTS_IDS = False
PERFORMANCE_REPORT = False  # This is a flag from the original HPE Demo of OpenVino, not needed
SHOW_XS = False


def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, width, height


def compute_fps(inference_time):
    return int(1000.0 / inference_time * 100) / 100.0


def analyse_videos(args):
    for video_path in args.video_paths:

        print("os.path.exists(video_path):", os.path.exists(video_path))
        video_name = os.path.basename(video_path)
        no_ext_video_name = os.path.splitext(video_name)[0]
        frame_count, width, height = number_of_frames(video_path)

        target_path = os.path.join(STATISTICS_FOLDER, no_ext_video_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for n in range(0, REPETITIONS):
            print('----------------------------------------')
            print('Opening video', video_path, '...')
            print('Video name:', no_ext_video_name)
            print('Numer of frames:', frame_count)
            print('Width, height:', width, ',', height)
            print('----------------------------------------')

            statistics = []
            chpe_proxy = chpe.CameraHPEProxy(HPE_MODEL, TARGET_DEVICE, video_path, PERFORMANCE_REPORT,
                                             SHOW_KEYPOINTS_IDS, SHOW_XS)

            stop = False
            frame = 0
            for i in range(0, START_MEASURING_FRAME):
                chpe_proxy.estimate_poses()

            total_appending_time = 0
            while not stop:
                hpe_start_time = time.time()
                poses = chpe_proxy.estimate_poses()
                hpe_inference_time = time.time() - hpe_start_time

                num_detections = len(poses)
                fps = compute_fps(chpe_proxy.get_instant_inference_time())
                # print('FRAME #', frame, ': (DETS, FPS) = (', num_detections, ',', fps, ')')

                appending_start_time = time.time()
                statistics.append([num_detections, fps, hpe_inference_time])
                total_appending_time += time.time() - appending_start_time

                frame += 1
                # stop = chpe_proxy.render_poses()
                stop = not chpe_proxy.read() or stop

            npstats = np.array(statistics)
            np.save(os.path.join(target_path, 'iter_' + str(n)), npstats)

            model_statistics = {
                "appending_times": str(total_appending_time),
                "analysed_frames": str(frame_count - START_MEASURING_FRAME)
            }
            with open(os.path.join(target_path, 'other' + str(n) + '.json'), 'w') as f:
                json.dump(model_statistics, f, indent=2)


def load_statistics_data(video_path=None):
    if video_path:
        video_paths = [video_path]
    else:
        video_paths = [os.path.join(STATISTICS_FOLDER, folder) for folder in os.listdir(STATISTICS_FOLDER)
                       if os.path.isdir(os.path.join(STATISTICS_FOLDER, folder))]

    stats = np.empty([0, 3])

    for video_folder in video_paths:
        np_files = [os.path.join(video_folder, np_dump) for np_dump in os.listdir(video_folder)
                    if (os.path.isfile(os.path.join(video_folder, np_dump)) and 'json' not in np_dump)]

        # print('np_files:', np_files)

        for npf in np_files:
            stats = np.concatenate((stats, np.load(npf, allow_pickle=True)), axis=0)

    return stats


def plot(args):
    stats = load_statistics_data()
    min_people = int(stats[:, 0].min())
    max_people = int(stats[:, 0].max())

    fps_data = []
    infer_time_data = []

    for n in range(min_people, max_people+1):
        fps_for_n_people = stats[stats[:, 0] == n][:, 1]
        infer_time_n_people = stats[stats[:, 0] == n][:, 2]
        if ROUND_DECIMALS:
            fps_data.append(np.around(fps_for_n_people, decimals=1))
            infer_time_data.append(np.around(infer_time_n_people, decimals=2))
        else:
            fps_data.append(fps_for_n_people)
            infer_time_data.append(infer_time_n_people)

    fps_fig, ax = plt.subplots()

    flierprops = dict(markeredgecolor='r')

    ax.set_title('FPS for number of people detected')
    ax.boxplot(fps_data, flierprops=flierprops)
    ax.set_xlabel('Number of people in a frame')
    ax.set_ylabel("Frames per second [1/s]")

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    # plt.show()
    if args.plot_path:
        plt.savefig(os.path.join(args.plot_path, 'fps.png'), dpi=300, bbox_inches='tight')

    infer_fig, ax = plt.subplots()
    ax.set_title('Inference time for number of people detected')
    ax.boxplot(infer_time_data, flierprops=flierprops)
    ax.set_xlabel('Number of people in a frame')
    ax.set_ylabel("Inference time [s]")

    # plt.show()
    if args.plot_path:
        plt.savefig(os.path.join(args.plot_path, 'inference_time.png'), dpi=300, bbox_inches='tight')
    plt.close(fps_fig)
    plt.close(infer_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''HPE-to-HAR for robots - Video statistics utility''')

    subpars = parser.add_subparsers(help='Sub-commands of Video statistics utility')
    parser_analyse = subpars.add_parser('analyse', help='Analyse number of detected people and FPS of videos.')

    parser_analyse.add_argument('-v', dest='video_paths', type=str, nargs='*', help='List of path to videos to analyse.')
    parser_analyse.set_defaults(func=analyse_videos)

    parser_plot = subpars.add_parser('plot', help='Produce plots of the analysed videos.')
    parser_plot.add_argument('-o', dest='plot_path', type=str, help='Output path of plots.')
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)

