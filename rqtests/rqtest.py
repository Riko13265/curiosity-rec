

from redis import Redis
from rq import Queue


queue = Queue(connection=Redis())


from asdf import asdfg
job = queue.enqueue(asdfg, 1234, result_ttl=8640000)

