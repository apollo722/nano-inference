import argparse
import json
import threading
import time

import requests


def send_one_request(url, model, prompt, max_tokens, request_id):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        print(f"Request {request_id} finished in {end_time - start_time:.2f}s")
    except Exception as e:
        print(f"Request {request_id} failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1/completions"
    model = "qwen3-0.6b"
    prompt = "Who are you?"

    threads = []
    print(f"Firing {args.concurrency} requests simultaneously...")

    # Pre-create threads to minimize start-up skew
    for i in range(args.concurrency):
        t = threading.Thread(
            target=send_one_request, args=(url, model, prompt, args.max_tokens, i)
        )
        threads.append(t)

    # Start all threads as close together as possible
    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
