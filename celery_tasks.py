from celery import Celery
from inpaint.main import ImageProcessor
from PIL import Image
import os
from celery.signals import task_success


celery = Celery(
    "dap391_project",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

processor = None
BASE_RESULT_FOLDER = os.path.join('inpaint', 'result')

@celery.task
def load_model():
    global processor
    if not processor:
        processor = ImageProcessor()
        print('LOAD MODEL DONE!')


@celery.task
def celery_remove_anything(session_id, filename, input_img_path, point_coords, dilate_kernel_size):
    global processor

    user_result_folder = os.path.join(BASE_RESULT_FOLDER, session_id)
    if not os.path.exists(user_result_folder):
        os.makedirs(user_result_folder)

    img_inpainted_array_list, name_list = processor.remove_anything(filename, input_img_path, point_coords, dilate_kernel_size)

    for i in range(len(img_inpainted_array_list)):
        img_inpainted_path = os.path.join(user_result_folder, name_list[i])
        Image.fromarray(img_inpainted_array_list[i]).save(img_inpainted_path)

    return name_list, session_id


@celery.task
def celery_replace_anything(session_id, filename, input_img_path, point_coords, text_prompt, mask_index):
    global processor

    user_result_folder = os.path.join(BASE_RESULT_FOLDER, session_id)
    if not os.path.exists(user_result_folder):
        os.makedirs(user_result_folder)

    img_inpainted, filename = processor.replace_anything(filename, input_img_path, point_coords, text_prompt, mask_index)

    file_inpainted_path = os.path.join(user_result_folder, filename)
    Image.fromarray(img_inpainted).save(file_inpainted_path)

    return filename, session_id


@celery.task
def celery_fill_anything(session_id, filename, input_img_path, point_coords, text_prompt, dilate_kernel_size, mask_index):
    global processor

    user_result_folder = os.path.join(BASE_RESULT_FOLDER, session_id)
    if not os.path.exists(user_result_folder):
        os.makedirs(user_result_folder)

    img_inpainted, filename = processor.fill_anything(filename, input_img_path, point_coords, text_prompt, dilate_kernel_size, mask_index)

    file_inpainted_path = os.path.join(user_result_folder, filename)
    Image.fromarray(img_inpainted).save(file_inpainted_path)

    return filename, session_id


@task_success.connect
def task_completed_handler(sender=None, **kwargs):
    with open('queue.txt', 'r') as f:
        lines = f.readlines()
    if lines:
        with open('queue.txt', 'w') as f:
            f.writelines(lines[1:])
