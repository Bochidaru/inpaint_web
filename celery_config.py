from celery_tasks import *

celery = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery.conf.update({
    'broker_connection_retry_on_startup': True
})
