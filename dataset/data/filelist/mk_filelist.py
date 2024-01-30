import glob

file_list = glob.glob("/home/hlj/danzi/talkingface-toolkit-main/dataset/vctk/data/*/*.wav")

with open("filelist.txt", 'w') as f:
    for item in file_list:
        wav_name, spk = item.split("/")[-1], item.split("/")[-2] 
        f.write(wav_name+"\n")