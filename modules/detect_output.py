import cv2
import os

def detect_out(detect_directory, frame, level):
    if not os.path.exists(detect_directory):
        os.makedirs(detect_directory)
    output_files = os.listdir(detect_directory)
    output_files = [file for file in output_files if file.startswith('detected-frame-') and file.endswith('.png')]
    output_files = sorted(output_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    if output_files:
        last_file = output_files[-1]
        print(last_file)
        last_file_number = int(last_file.split('-')[-1].split('.')[0])
        print(last_file)
        output_counter = last_file_number + 1
    else:
        output_counter = 1
    result = cv2.imwrite(os.path.join(detect_directory, f'detected-frame-{output_counter}.png'), frame)
    return result