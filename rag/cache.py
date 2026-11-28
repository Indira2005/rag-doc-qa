"""Redis answer cache. If Redis is missing or down, the app keeps working without it."""
import hashlib
import json
import logging
import redis.asyncio as redis
from rag import config

log = logging.getLogger(__name__)
client = redis.from_url(config.REDIS_URL, decode_responses=True,
                        socket_connect_timeout=1) if config.REDIS_URL else None


def key(*parts) -> str:
    return "qa:" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()


async def get(k):
    if client is None:
        return None
    try:
        value = await client.get(k)
        return json.loads(value) if value else None
    except redis.RedisError as e:
        log.warning("cache read failed: %s", e)
        return None


async def put(k, value, ttl=3600):
    if client is None:
        return
    try:
        await client.set(k, json.dumps(value), ex=ttl)
    except redis.RedisError as e:
        log.warning("cache write failed: %s", e)


async def clear():
    """Called after an upload so old answers are not served for changed documents."""
    if client is None:
        return
    try:
        async for k in client.scan_iter("qa:*"):
            await client.delete(k)
    except redis.RedisError as e:
        log.warning("cache clear failed: %s", e)
