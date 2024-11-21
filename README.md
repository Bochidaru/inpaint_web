How to run this program?
1. Download pretrained_models folder and put it in inpaint folder from this link: https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg
2. Install redis and enable, this web use redis as broker for celery, enable redis, 'sudo apt install redis-server && sudo apt install redis-server' (for ubuntu/debian), use 'sudo systemctl status redis' to check the status of redis, if you use Windows, you can run Redis-x64-3.0.504/redis-server.exe
3. setup firewall, install requirements package in file 'r.txt' by running: 'pip install --no-cache-dir -r r.txt'
4. run this command: 'celery -A celery_config worker --loglevel=info --pool=solo' (for windows), 'celery -A celery_config worker --loglevel=info --concurrency=1' (for linux/debian)
5. run flask through this command: 'python app.py' or run through gunicorn: 'gunicorn -w 4 -b 128.0.0.1:5000 app:app'
6. if you want a reverse proxy, i highly recommend you to use nginx, it's easy to config
