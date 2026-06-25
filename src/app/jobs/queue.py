from redis import Redis
from rq import Queue

from src.app.core.config import settings

redis_conn = Redis.from_url(settings.redis_url)
insight_queue = Queue("insights", connection=redis_conn)
