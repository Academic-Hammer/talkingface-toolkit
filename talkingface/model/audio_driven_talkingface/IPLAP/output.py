import subprocess


def run_ffmpeg_command(input_audio, input_video, output_file):
    command = ['ffmpeg', '-y', '-i', input_audio, '-i', input_video, '-strict', '-2', '-q:v', '1', output_file]

    # 使用subprocess.Popen启动非阻塞子进程
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 读取标准输出和标准错误
    stdout, stderr = process.communicate()

    # 获取命令的返回码
    return_code = process.returncode

    # 打印标准输出和标准错误
    print("ffmpeg stdout:", stdout)
    print("ffmpeg stderr:", stderr)
    print("Command returned:", return_code)

    # 根据返回码判断命令执行是否成功
    if return_code == 0:
        print("Succeed output results to:", output_file)
    else:
        print("Failed to execute ffmpeg command")


# # 调用函数并传入输入音频、输入视频和输出文件路径
# input_audio = './test/template_video/audio2.wav'
# input_video = 'tempfile_of_test_result/result.avi'
# output_file = './test_result/test2result_N_25_Nl_15.mp4'
#
# run_ffmpeg_command(input_audio, input_video, output_file)
