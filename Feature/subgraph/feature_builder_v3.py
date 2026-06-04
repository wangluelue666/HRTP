#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from config.feature_schema_v3 import FEATURE_SCHEMA
from utils.resource_manager_v3 import ResourceManagerV3


class FeatureBuilderV3:
    """
    Build task-aligned node features for my_method_v3.

    All feature fields and orders must follow FEATURE_SCHEMA exactly.
    """

    def __init__(self, resource_manager: ResourceManagerV3) -> None:
        self.rm = resource_manager

    # ------------------------------------------------------------------
    # Public builders
    # ------------------------------------------------------------------
    def build_target_feature(self, target_author_id: str, year: int) -> List[float]:
        vec = {
            "cum_papers": self.rm.get_author_pub_count_until(target_author_id, year),
            "cum_citations": self.rm.get_author_citation_count_until(target_author_id, year),
            "cum_hindex": self.rm.get_author_hindex_until(target_author_id, year),
            "papers_last_1y": self.rm.get_author_pub_count_window(target_author_id, year, 1),
            "citations_last_1y": self.rm.get_author_citation_count_window(target_author_id, year, 1),
            "hindex_delta_last_1y": self.rm.get_author_hindex_delta_window(target_author_id, year, 1),
            "papers_last_3y": self.rm.get_author_pub_count_window(target_author_id, year, 3),
            "citations_last_3y": self.rm.get_author_citation_count_window(target_author_id, year, 3),
            "hindex_delta_last_3y": self.rm.get_author_hindex_delta_window(target_author_id, year, 3),
            "active_years": self.rm.get_author_active_years_until(target_author_id, year),
            "recent_activity_ratio": self.rm.get_author_recent_activity_ratio(target_author_id, year, 3),
            "cum_coauthors": self.rm.get_total_coauthors_until(target_author_id, year),
            "coauthors_last_3y": self.rm.get_recent_coauthors_count(target_author_id, year, 3),
        }
        return self._ordered_feature("target", vec)

    def build_author_feature(
        self,
        target_author_id: str,
        coauthor_id: str,
        year: int,
    ) -> List[float]:
        total_coauthors = max(self.rm.get_total_coauthors_until(coauthor_id, year), 1)
        collab_total = self.rm.get_collab_count_until(target_author_id, coauthor_id, year)
        collab_recent = self.rm.get_collab_count_window(target_author_id, coauthor_id, year, 3)

        vec = {
            "collab_count_with_target": collab_total,
            "collab_count_with_target_last_3y": collab_recent,
            "years_since_last_collab_with_target": self.rm.get_years_since_last_collab(
                target_author_id, coauthor_id, year
            ),
            "collab_ratio_to_author_total": float(collab_total / total_coauthors),
            "cum_papers": self.rm.get_author_pub_count_until(coauthor_id, year),
            "cum_citations": self.rm.get_author_citation_count_until(coauthor_id, year),
            "cum_hindex": self.rm.get_author_hindex_until(coauthor_id, year),
            "papers_last_1y": self.rm.get_author_pub_count_window(coauthor_id, year, 1),
            "citations_last_1y": self.rm.get_author_citation_count_window(coauthor_id, year, 1),
            "hindex_delta_last_1y": self.rm.get_author_hindex_delta_window(coauthor_id, year, 1),
            "papers_last_3y": self.rm.get_author_pub_count_window(coauthor_id, year, 3),
            "citations_last_3y": self.rm.get_author_citation_count_window(coauthor_id, year, 3),
            "hindex_delta_last_3y": self.rm.get_author_hindex_delta_window(coauthor_id, year, 3),
            "active_years": self.rm.get_author_active_years_until(coauthor_id, year),
            "recent_activity_ratio": self.rm.get_author_recent_activity_ratio(coauthor_id, year, 3),
            "is_new_to_target_recent_3y": self.rm.is_new_coauthor_for_author(
                target_author_id, coauthor_id, year, 3
            ),
        }
        return self._ordered_feature("author", vec)

    def build_paper_feature(
        self,
        target_author_id: str,
        paper_id: str,
        year: int,
        stock_score: float = 0.0,
        flow_score: float = 0.0,
    ) -> List[float]:
        paper_year = self.rm.get_paper_year(paper_id)
        if paper_year is None:
            paper_year = year

        age = max(year - paper_year, 0)
        author_first_year = self.rm.get_author_first_year(target_author_id)
        if author_first_year is None:
            author_first_year = paper_year

        topics = self.rm.get_paper_topics(paper_id)
        venue = self.rm.get_paper_venue(paper_id)
        authors = self.rm.get_paper_authors(paper_id)

        coauthor_ids = [aid for aid in authors if aid != target_author_id]
        coauthor_count = len(coauthor_ids)

        # novelty signals
        introduces_new_coauthor = int(
            any(self.rm.is_new_coauthor_for_author(target_author_id, aid, year, 3) > 0 for aid in coauthor_ids)
        )
        introduces_new_topic = int(
            any(self.rm.is_new_topic_for_author(target_author_id, tp, year, 3) > 0 for tp in topics)
        )
        introduces_new_venue = int(self.rm.is_new_venue_for_author(target_author_id, venue, year, 3) > 0)

        # diversity signals
        coauthor_diversity = self._safe_ratio(coauthor_count, max(self.rm.get_total_coauthors_until(target_author_id, year), 1))
        topic_diversity = self._safe_ratio(len(set(topics)), max(len(topics), 1))

        vec = {
            "paper_age": age,
            "paper_age_norm": self._safe_ratio(age, max(year - 2000 + 1, 1)),
            "is_recent_1y": int(age <= 0),
            "is_recent_3y": int(age <= 2),
            "is_same_year": int(paper_year == year),
            "years_since_author_first_paper": max(paper_year - author_first_year, 0),
            "coauthor_count": coauthor_count,
            "has_topic": int(len(topics) > 0),
            "has_venue": int(venue != "unknown"),
            "introduces_new_coauthor": introduces_new_coauthor,
            "introduces_new_topic": introduces_new_topic,
            "introduces_new_venue": introduces_new_venue,
            "coauthor_diversity": coauthor_diversity,
            "topic_diversity": topic_diversity,
            "stock_score": float(stock_score),
            "flow_score": float(flow_score),
        }
        return self._ordered_feature("paper", vec)

    def build_topic_feature(
        self,
        target_author_id: str,
        topic: str,
        year: int,
    ) -> List[float]:
        total = self.rm.get_topic_count_until(target_author_id, topic, year)
        recent = self.rm.get_topic_count_window(target_author_id, topic, year, 3)

        vec = {
            "topic_count_total": total,
            "topic_count_last_3y": recent,
            "topic_recent_ratio": self._safe_ratio(recent, max(total, 1)),
            "is_new_topic": self.rm.is_new_topic_for_author(target_author_id, topic, year, 3),
            "topic_span_years": self.rm.get_topic_span_years_until(target_author_id, topic, year),
            "years_since_last_topic_use": self.rm.get_years_since_last_topic_use(target_author_id, topic, year),
        }
        return self._ordered_feature("topic", vec)

    def build_venue_feature(
        self,
        target_author_id: str,
        venue: str,
        year: int,
    ) -> List[float]:
        total = self.rm.get_venue_count_until(target_author_id, venue, year)
        recent = self.rm.get_venue_count_window(target_author_id, venue, year, 3)

        vec = {
            "venue_count_total": total,
            "venue_count_last_3y": recent,
            "venue_recent_ratio": self._safe_ratio(recent, max(total, 1)),
            "is_new_venue": self.rm.is_new_venue_for_author(target_author_id, venue, year, 3),
            "venue_span_years": self.rm.get_venue_span_years_until(target_author_id, venue, year),
            "years_since_last_venue_use": self.rm.get_years_since_last_venue_use(target_author_id, venue, year),
        }
        return self._ordered_feature("venue", vec)

    # ------------------------------------------------------------------
    # Batch-like helpers for node dict enrichment
    # ------------------------------------------------------------------
    def enrich_target_node(
        self,
        target_author_id: str,
        year: int,
        node_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = {} if node_dict is None else dict(node_dict)
        out["feature"] = self.build_target_feature(target_author_id, year)
        return out

    def enrich_author_node(
        self,
        target_author_id: str,
        author_node: Dict[str, Any],
        year: int,
    ) -> Dict[str, Any]:
        out = dict(author_node)
        aid = str(out.get("id", "")).strip()
        out["feature"] = self.build_author_feature(target_author_id, aid, year)
        return out

    def enrich_paper_node(
        self,
        target_author_id: str,
        paper_node: Dict[str, Any],
        year: int,
        stock_score: Optional[float] = None,
        flow_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        out = dict(paper_node)
        pid = str(out.get("id", "")).strip()

        if stock_score is None:
            stock_score = float(out.get("stock_score", 0.0))
        if flow_score is None:
            flow_score = float(out.get("flow_score", 0.0))

        out["feature"] = self.build_paper_feature(
            target_author_id=target_author_id,
            paper_id=pid,
            year=year,
            stock_score=stock_score,
            flow_score=flow_score,
        )
        return out

    def enrich_topic_node(
        self,
        target_author_id: str,
        topic_node: Dict[str, Any],
        year: int,
    ) -> Dict[str, Any]:
        out = dict(topic_node)
        topic = str(out.get("name", "") or out.get("id", "")).strip().lower()
        out["feature"] = self.build_topic_feature(target_author_id, topic, year)
        return out

    def enrich_venue_node(
        self,
        target_author_id: str,
        venue_node: Dict[str, Any],
        year: int,
    ) -> Dict[str, Any]:
        out = dict(venue_node)
        venue = str(out.get("name", "") or out.get("id", "")).strip().lower()
        out["feature"] = self.build_venue_feature(target_author_id, venue, year)
        return out

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_feature_dims(self) -> Dict[str, int]:
        return FEATURE_SCHEMA.dim_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ordered_feature(self, group_name: str, value_map: Dict[str, Any]) -> List[float]:
        fields = FEATURE_SCHEMA.get_fields(group_name)
        feat = [self._to_float(value_map.get(k, 0.0)) for k in fields]
        return feat

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        den = max(float(den), 1e-12)
        return float(num) / den

    @staticmethod
    def _to_float(x: Any) -> float:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        try:
            return float(x)
        except Exception:
            return 0.0


if __name__ == "__main__":
    from utils.resource_manager_v3 import ResourceManagerV3

    rm = ResourceManagerV3("aps")
    fb = FeatureBuilderV3(rm)

    # simple smoke tests
    target_id = rm.author_ids_history[0]
    year = 2014

    t = fb.build_target_feature(target_id, year)
    print("target_dim =", len(t), "expected =", FEATURE_SCHEMA.get_dim("target"))

    papers = rm.get_author_papers_until(target_id, year)
    if papers:
        p = fb.build_paper_feature(target_id, papers[0], year, stock_score=1.2, flow_score=0.8)
        print("paper_dim  =", len(p), "expected =", FEATURE_SCHEMA.get_dim("paper"))

    print("dim_dict =", fb.validate_feature_dims())