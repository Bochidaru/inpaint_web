from flask import Flask, request, render_template, session, url_for, send_from_directory
from PIL import Image
import os
import uuid
import shutil
from threading import Thread
import time
from datetime import datetime
from celery_config import celery
from celery_tasks import celery_remove_anything, celery_replace_anything, celery_fill_anything, load_model


app = Flask('dap391_project')
app.secret_key = 'minhphuong04'

app.config['CELERY_BROKER_URL'] = "redis://localhost:6379/0"
app.config['CELERY_RESULT_BACKEND'] = "redis://localhost:6379/0"

celery.conf.update(app.config)


BASE_RESULT_FOLDER = os.path.join('inpaint', 'result')

last_access_times = {}

# load the model in celery
load_model.delay()

with open('queue.txt', 'w') as f:
    pass 


def queue_task_id(task_id):
    with open('queue.txt', 'a') as f:
        f.write(f'{task_id}\n')


def read_task_ids_as_list():
    with open('queue.txt', 'r') as f:
        task_id = f.read().splitlines()
    return task_id


def delete_session_folder(session_id):
    """Delete the session folder if users dont access after 15 minutes."""
    while True:
        time.sleep(60)  # Kiểm tra mỗi phút
        current_time = datetime.now()
        last_access_time = last_access_times.get(session_id)
        if last_access_time and (current_time - last_access_time).total_seconds() > 900:
            user_result_folder = os.path.join(BASE_RESULT_FOLDER, session_id)
            if os.path.exists(user_result_folder):
                if os.path.exists(user_result_folder):
                    shutil.rmtree(user_result_folder)


@app.route('/')
def home():
    global call_loaded_model_func

    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    last_access_times[session_id] = datetime.now()

    user_result_folder = os.path.join(BASE_RESULT_FOLDER, session_id)
    if not os.path.exists(user_result_folder):
        os.makedirs(user_result_folder)
    
    Thread(target=delete_session_folder, args=(session_id,), daemon=True).start()

    return render_template('index.html', session_id=session_id)


@app.route('/remove-object', methods=['POST'])
def remove_object():  
    image = request.files['image']
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    
    filename = image.filename

    session_id = session.get('session_id')

    last_access_times[session_id] = datetime.now()
    
    task_list = read_task_ids_as_list()

    if len(task_list) >= 20:
        return {'message': 'Queue is full now (20 tasks), please wait a moment and try again.'}, 504

    temp_image_path = os.path.join(BASE_RESULT_FOLDER, session_id, filename)
    image.save(temp_image_path)

    task = celery_remove_anything.delay(session_id, filename, temp_image_path, [[x, y]], 15)

    queue_task_id(task.id)

    return {'taskID': task.id}, 200


@app.route('/replace-background', methods=['POST'])
def replace_background():
    image = request.files['image']
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    mask_number = int(request.form.get('mask-number'))
    prompt = request.form.get('prompt')
    filename = image.filename

    session_id = session.get('session_id')
    
    last_access_times[session_id] = datetime.now()

    task_list = read_task_ids_as_list()

    if len(task_list) >= 20:
        return {'message': 'Queue is full now (20 tasks), please wait a moment and try again.'}, 504

    temp_image_path = os.path.join(BASE_RESULT_FOLDER, session_id, filename)
    image.save(temp_image_path)

    task = celery_replace_anything.delay(session_id, filename, temp_image_path, [[x, y]], prompt, mask_number)

    queue_task_id(task.id)

    return {'taskID': task.id}, 200


@app.route('/replace-object', methods=['POST'])
def replace_object():
    image = request.files['image']
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    mask_number = int(request.form.get('mask-number'))
    prompt = request.form.get('prompt')
    filename = image.filename

    session_id = session.get('session_id')
    
    last_access_times[session_id] = datetime.now()

    task_list = read_task_ids_as_list()

    if len(task_list) >= 20:
        return {'message': 'Queue is full now (20 tasks), please wait a moment and try again.'}, 504

    temp_image_path = os.path.join(BASE_RESULT_FOLDER, session_id, filename)
    image.save(temp_image_path)

    task = celery_fill_anything.delay(session_id, filename, temp_image_path, [[x, y]], prompt, 15, mask_number)

    queue_task_id(task.id)

    return {'taskID': task.id}, 200


@app.route('/remove-object-result/<task_id>')
def get_remove_object_result(task_id):
    task = celery_remove_anything.AsyncResult(task_id)
    if task.state == 'PENDING':
        task_list = read_task_ids_as_list()
        queue_number = task_list.index(task_id)
        return {'task': 'REMOVE_OBJECT','state': 'PENDING', 'queue_number': queue_number}
    elif task.state == 'SUCCESS':
        namelist = task.result[0]
        session_id = task.result[1]
        image_urls = [url_for('send_inpainted_image', session_id=session_id, filename=name) for name in namelist]
        return {'task': 'REMOVE_OBJECT', 'state': 'SUCCESS', 'image_urls': image_urls}
    else:
        return {'task': 'REMOVE_OBJECT', 'state': 'FAILURE', 'result': str(task.info)}


@app.route('/change-background-result/<task_id>')
def get_change_background_result(task_id):
    task = celery_replace_anything.AsyncResult(task_id)
    if task.state == 'PENDING':
        task_list = read_task_ids_as_list()
        queue_number = task_list.index(task_id)
        return {'task': 'CHANGE_BACKGROUND', 'state': 'PENDING', 'queue_number': queue_number}
    elif task.state == 'SUCCESS':
        filename = task.result[0]
        session_id = task.result[1]
        image_url = url_for('send_inpainted_image', session_id=session_id, filename=filename)
        return {'task': 'CHANGE_BACKGROUND', 'state': 'SUCCESS', 'image_url': image_url}
    else:
        return {'task': 'CHANGE_BACKGROUND', 'state': 'FAILURE', 'result': str(task.info)}
    

@app.route('/change-object-result/<task_id>')
def get_change_object_result(task_id):
    task = celery_replace_anything.AsyncResult(task_id)
    if task.state == 'PENDING':
        task_list = read_task_ids_as_list()
        queue_number = task_list.index(task_id)
        return {'task': 'CHANGE_OBJECT', 'state': 'PENDING', 'queue_number': queue_number}
    elif task.state == 'SUCCESS':
        filename = task.result[0]
        session_id = task.result[1]
        image_url = url_for('send_inpainted_image', session_id=session_id, filename=filename)
        return {'task': 'CHANGE_OBJECT', 'state': 'SUCCESS', 'image_url': image_url}
    else:
        return {'task': 'CHANGE_OBJECT', 'state': 'FAILURE', 'result': str(task.info)}


@app.route('/inpaint/result/<session_id>/<filename>')
def send_inpainted_image(session_id, filename):
    return send_from_directory(os.path.join(BASE_RESULT_FOLDER, session_id), filename)


if __name__ == '__main__':
    app.run(port=5000)
