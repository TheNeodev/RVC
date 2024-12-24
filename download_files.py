import subprocess, os, sys

now_dir = os.getcwd()
sys.path.append(now_dir)

files = {
    "rmvpe.pt":"https://huggingface.co/theNeofr/rvc-base/resolve/main/rmvpe.pt",
    "hubert_base.pt":"https://huggingface.co/theNeofr/rvc-base/resolve/main/hubert_base.pt"
}
for file, link in files.items():
    file_path = os.path.join(now_dir, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', link, '-O', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")

