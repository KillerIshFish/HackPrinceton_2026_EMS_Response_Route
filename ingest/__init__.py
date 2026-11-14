"""Data ingestion. v0: synthetic building generator + stubs for the real datasets."""

from ingest.matterport import generate_synthetic_office, list_buildings, load_building

__all__ = ["generate_synthetic_office", "list_buildings", "load_building"]
