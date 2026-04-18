from concurrent.futures import Future
from typing import Dict, Optional

from nano_inference.core.request import GenerateOutput


class ResponseManager:
    def __init__(self):
        self._futures: Dict[str, Future[GenerateOutput]] = {}

    def create_future(self, request_id: str) -> Future[GenerateOutput]:
        future = Future()
        self._futures[request_id] = future
        return future

    def complete(self, request_id: str, output: GenerateOutput) -> None:
        if request_id not in self._futures:
            return
        future = self._futures.pop(request_id)
        future.set_result(output)

    def fail(self, request_id: str, exception: Exception) -> None:
        if request_id not in self._futures:
            return
        future = self._futures.pop(request_id)
        future.set_exception(exception)

    def has_future(self, request_id: str) -> bool:
        return request_id in self._futures
