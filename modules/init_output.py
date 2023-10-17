import cv2
import os

def init_out(output_directory, frame_width, frame_height):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_files = os.listdir(output_directory)
    output_files = [file for file in output_files if file.startswith('output-') and file.endswith('.mp4')]
    output_files.sort()
    if output_files:
        last_file = output_files[-1]
        last_file_number = int(last_file.split('-')[-1].split('.')[0])
        output_counter = last_file_number + 1
    else:
        output_counter = 1
    output_path = os.path.join(output_directory, f'output-{output_counter}.mp4')
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))
    return writer