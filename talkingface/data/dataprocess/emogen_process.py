import argparse
import os
import subprocess
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import traceback
import audio
from hparams import hparams as hp

import face_detection

def modify_frame_rate(input_folder, output_folder, fps=25.0):
    # 修改视频的帧率
    os.makedirs(output_folder, exist_ok=True)
    fileList = []
    for root, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.mpg', '.mov', '.flv')):
                fileList.append(os.path.join(root, filename))

    for file in fileList:
        subprocess.run("ffmpeg -i {} -r {} -y {}".format(
            file, fps, os.path.join(output_folder, os.path.basename(file))), shell=True)

def process_video_file(vfile, args, gpu_id, fa):
    video_stream = cv2.VideoCapture(vfile)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    vidname = os.path.basename(vfile).split('.')[0]

    fulldir = os.path.join(args.preprocessed_root, vidname)
    os.makedirs(fulldir, exist_ok=True)

    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    i = -1
    for fb in batches:
        preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            x1, y1, x2, y2 = f
            cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    fulldir = os.path.join(args.preprocessed_root, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = os.path.join(fulldir, 'audio.wav')

    command = f"ffmpeg -loglevel panic -y -i {vfile} -strict -2 {wavpath}"
    subprocess.call(command, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help='Path to folder that contains original video files')
    parser.add_argument("--output_folder", type=str, help='Path to folder for storing modified videos', default='modified_videos/')
    parser.add_argument("--fps", type=float, help='Target FPS', default=25.0)
    parser.add_argument("--ngpu", type=int, help='Number of GPUs across which to run in parallel', default=1)
    parser.add_argument("--batch_size", type=int, help='Single GPU Face detection batch size', default=32)
    parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

    args = parser.parse_args()

    # 第一步：修改视频帧率
    modify_frame_rate(args.input_folder, args.output_folder, args.fps)

    # 配置面部检测
    fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                        device=f'cuda:{id}') for id in range(args.ngpu)]

    # 第二步：视频和音频的数据预处理
    filelist = glob(os.path.join(args.output_folder, '*.mp4'))

    print('Started processing videos')
    jobs = [(vfile, args, i % args.ngpu, fa[i % args.ngpu]) for i, vfile in enumerate(filelist)]
    with ThreadPoolExecutor(args.ngpu) as p:
        futures = [p.submit(process_video_file, *job) for job in jobs]
        _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')
    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()

if __name__ == '__main__':
    main()
