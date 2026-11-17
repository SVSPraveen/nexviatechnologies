"""init_feature_store.py: Initializes Feast feature store and materializes features to SQLite online store using native Python API."""

import os
import sys
from datetime import datetime, timezone
from feast import FeatureStore

def initialize_and_materialize():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(base_dir, "..", "feature_repo")
    
    # Add repo_path to sys.path so we can import features
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        
    import features as feast_defs
    
    print(f"[Milestone 2] Initializing Feast feature store in {repo_path}...")
    store = FeatureStore(repo_path=repo_path)
    
    print("Applying Feast definitions (Entity + FeatureView) via native Python API...")
    store.apply([feast_defs.account, feast_defs.account_fraud_features])
    
    # Materialize features into online store
    end_date = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    start_date = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    print(f"Materializing features from {start_date} to {end_date} into online store...")
    store.materialize(start_date=start_date, end_date=end_date)
    
    print("[Milestone 2] Feast feature store initialized and materialized successfully!")

if __name__ == "__main__":
    initialize_and_materialize()
