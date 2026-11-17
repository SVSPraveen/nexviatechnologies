"""features.py: Feast Entity and FeatureView definitions for the Fraud Detection Prototype."""

import os
from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# File source pointing to the processed features parquet
base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, "..", "data", "processed", "features.parquet")

account_source = FileSource(
    path=parquet_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Entity
account = Entity(
    name="account_id",
    value_type=ValueType.STRING,
    description="Unique customer account identifier",
)

# Feature View
account_fraud_features = FeatureView(
    name="account_fraud_features",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="velocity_1h", dtype=Int64),
        Field(name="velocity_24h", dtype=Int64),
        Field(name="amount_to_user_avg_ratio", dtype=Float32),
        Field(name="device_changed_flag", dtype=Int64),
        Field(name="location_delta_km", dtype=Float32),
    ],
    online=True,
    source=account_source,
    tags={"team": "fraud_intelligence", "tier": "critical"},
)
