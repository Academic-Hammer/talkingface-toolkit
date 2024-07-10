import subprocess

def run_training_scripts():
    # 定义训练脚本的路径
    train_landmarks_script_path = "talkingface/model/audio_driven_talkingface/IPLAP/train_landmarks_generator.py"
    train_video_renderer_script_path = "talkingface/model/audio_driven_talkingface/IPLAP/train_video_renderer.py"

    # 运行 train_landmarks_generator.py 脚本
    print("Running train_landmarks_generator.py...")
    subprocess.run(["python", train_landmarks_script_path], check=True)

    # 运行 train_video_renderer.py 脚本
    print("Running train_video_renderer.py...")
    subprocess.run(["python", train_video_renderer_script_path], check=True)

    print("Training completed successfully.")

if __name__ == "__main__":
    run_training_scripts()
