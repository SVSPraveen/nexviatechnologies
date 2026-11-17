"""kafka_producer.py: Milestone 5 - Transaction streaming replay producer.

Replays transactions from the PaySim dataset into a Kafka topic ('transactions.v1').
Includes a seamless local file/queue fallback if a live Kafka broker is not present.
"""

import os
import time
import json
import argparse
import pandas as pd

TOPIC_NAME = "transactions.v1"

def create_kafka_producer(bootstrap_servers: str = "localhost:9092"):
    try:
        from confluent_kafka import Producer
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": "aegis-prototype-producer",
            "socket.timeout.ms": 2000,
        }
        producer = Producer(conf)
        # Test connectivity
        metadata = producer.list_topics(timeout=2.0)
        print(f"Connected to Kafka broker at {bootstrap_servers}. Active topics: {list(metadata.topics.keys())}")
        return producer, "kafka"
    except Exception as e:
        print(f"Notice: Live Kafka broker at {bootstrap_servers} unavailable ({e}). Using local stream spool fallback.")
        return None, "fallback"

def run_replay(dataset_path: str, rate_per_sec: float = 10.0, max_events: int = 500, bootstrap_servers: str = "localhost:9092"):
    print(f"[Milestone 5] Loading transaction replay dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    producer, mode = create_kafka_producer(bootstrap_servers)
    spool_file = os.path.join(os.path.dirname(dataset_path), "stream_spool.jsonl")

    print(f"Starting transaction replay at {rate_per_sec} events/sec (Mode: {mode.upper()})...")
    events_sent = 0
    sleep_interval = 1.0 / max(1.0, rate_per_sec)

    try:
        for idx, row in df.iterrows():
            if events_sent >= max_events:
                break

            event = {
                "transaction_id": f"tx_{idx:08d}",
                "step": int(row["step"]),
                "account_id": str(row["nameOrig"]),
                "amount": float(row["amount"]),
                "type": str(row["type"]),
                "nameDest": str(row["nameDest"]),
                "device_id": str(row["device_id"]),
                "card_id": str(row["card_id"]),
                "timestamp": time.time(),
            }

            payload = json.dumps(event)

            if mode == "kafka" and producer:
                producer.produce(
                    topic=TOPIC_NAME,
                    key=event["account_id"].encode("utf-8"),
                    value=payload.encode("utf-8"),
                )
                producer.poll(0)
            else:
                with open(spool_file, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")

            events_sent += 1
            if events_sent % 50 == 0:
                print(f"Replayed {events_sent}/{max_events} transactions (Latest: {event['transaction_id']}, Amount: ${event['amount']:,.2f})")

            time.sleep(sleep_interval)

    except KeyboardInterrupt:
        print("\nReplay stopped by user.")

    if mode == "kafka" and producer:
        producer.flush(timeout=5.0)

    print(f"[Milestone 5] Finished replaying {events_sent} transactions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay transactions into Kafka or local spool")
    parser.add_argument("--rate", type=float, default=20.0, help="Replay rate (events/second)")
    parser.add_argument("--max", type=int, default=200, help="Max events to replay")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_csv = os.path.join(base_dir, "..", "data", "raw", "paysim_sample.csv")
    run_replay(data_csv, rate_per_sec=args.rate, max_events=args.max)

