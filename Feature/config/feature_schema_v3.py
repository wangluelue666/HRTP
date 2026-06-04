#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    fields: List[str]

    @property
    def dim(self) -> int:
        return len(self.fields)


class FeatureSchemaV3:
    """
    Unified node feature schema for my_method_v3.

    Design principles
    -----------------
    1. Features must be task-aligned with stock / flow definitions.
    2. Avoid weak semantic placeholders such as raw topic_id or venue hash as numeric signals.
    3. Keep features compact, interpretable, and stable across datasets.
    4. All downstream modules must use this schema as the single source of truth.
    """

    def __init__(self) -> None:
        # Target author feature used in ring0
        self.target = FeatureGroup(
            name="target",
            fields=[
                # Stock-aligned cumulative signals
                "cum_papers",
                "cum_citations",
                "cum_hindex",
                # Flow-aligned short/mid-term signals
                "papers_last_1y",
                "citations_last_1y",
                "hindex_delta_last_1y",
                "papers_last_3y",
                "citations_last_3y",
                "hindex_delta_last_3y",
                # Activity profile
                "active_years",
                "recent_activity_ratio",
                # Collaboration profile
                "cum_coauthors",
                "coauthors_last_3y",
            ],
        )

        # Coauthor / author node feature
        self.author = FeatureGroup(
            name="author",
            fields=[
                # Relation strength to target author
                "collab_count_with_target",
                "collab_count_with_target_last_3y",
                "years_since_last_collab_with_target",
                "collab_ratio_to_author_total",
                # Author cumulative impact state
                "cum_papers",
                "cum_citations",
                "cum_hindex",
                # Author recent growth state
                "papers_last_1y",
                "citations_last_1y",
                "hindex_delta_last_1y",
                "papers_last_3y",
                "citations_last_3y",
                "hindex_delta_last_3y",
                # Activity profile
                "active_years",
                "recent_activity_ratio",
                # Entry / freshness marker in local neighborhood
                "is_new_to_target_recent_3y",
            ],
        )

        # Paper node feature
        self.paper = FeatureGroup(
            name="paper",
            fields=[
                # Time position
                "paper_age",
                "paper_age_norm",
                "is_recent_1y",
                "is_recent_3y",
                "is_same_year",
                "years_since_author_first_paper",
                # Structural participation
                "coauthor_count",
                "has_topic",
                "has_venue",
                # Expansion / novelty signals
                "introduces_new_coauthor",
                "introduces_new_topic",
                "introduces_new_venue",
                # Diversity / bridge-style local signals
                "coauthor_diversity",
                "topic_diversity",
                # View-aligned score hooks
                "stock_score",
                "flow_score",
            ],
        )

        # Topic node feature
        self.topic = FeatureGroup(
            name="topic",
            fields=[
                "topic_count_total",
                "topic_count_last_3y",
                "topic_recent_ratio",
                "is_new_topic",
                "topic_span_years",
                "years_since_last_topic_use",
            ],
        )

        # Venue node feature
        self.venue = FeatureGroup(
            name="venue",
            fields=[
                "venue_count_total",
                "venue_count_last_3y",
                "venue_recent_ratio",
                "is_new_venue",
                "venue_span_years",
                "years_since_last_venue_use",
            ],
        )

        self._groups: Dict[str, FeatureGroup] = {
            "target": self.target,
            "author": self.author,
            "paper": self.paper,
            "topic": self.topic,
            "venue": self.venue,
        }

    def get_group(self, name: str) -> FeatureGroup:
        key = name.lower().strip()
        if key not in self._groups:
            raise KeyError(f"Unsupported feature group: {name}")
        return self._groups[key]

    def get_fields(self, name: str) -> List[str]:
        return list(self.get_group(name).fields)

    def get_dim(self, name: str) -> int:
        return self.get_group(name).dim

    def as_dict(self) -> Dict[str, List[str]]:
        return {k: list(v.fields) for k, v in self._groups.items()}

    def dim_dict(self) -> Dict[str, int]:
        return {k: v.dim for k, v in self._groups.items()}

    def pretty_print(self) -> str:
        lines: List[str] = []
        for name in ["target", "author", "paper", "topic", "venue"]:
            group = self.get_group(name)
            lines.append(f"[{group.name}] dim={group.dim}")
            for i, field in enumerate(group.fields):
                lines.append(f"  {i:02d}  {field}")
        return "\n".join(lines)


FEATURE_SCHEMA = FeatureSchemaV3()


if __name__ == "__main__":
    print(FEATURE_SCHEMA.pretty_print())
    print()
    print("dim_dict =", FEATURE_SCHEMA.dim_dict())