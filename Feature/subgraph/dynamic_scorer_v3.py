#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from utils.resource_manager_v3 import ResourceManagerV3


class DynamicScorerV3:
    """
    Stage-2 task-aligned re-scoring for stock / flow candidates.

    Design goals
    ------------
    1. Refine stage-1 candidates into stronger ring1-ready nodes.
    2. Stock view should emphasize stable backbone and accumulated influence.
    3. Flow view should emphasize recent momentum, novelty, and expansion.
    4. Keep the scorer deterministic and efficient, but more structured than V2.
    """

    def __init__(self, resource_manager: ResourceManagerV3) -> None:
        self.rm = resource_manager

        # refined ring1 candidate limits
        self.refined_stock_papers = 48
        self.refined_flow_papers = 48

        self.refined_stock_authors = 36
        self.refined_flow_authors = 36

        self.refined_stock_topics = 24
        self.refined_flow_topics = 24

        self.refined_stock_venues = 16
        self.refined_flow_venues = 16

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refine(
        self,
        target_author_id: str,
        year: int,
        candidates: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        return {
            "stock": self._refine_view(target_author_id, year, candidates["stock"], view="stock"),
            "flow": self._refine_view(target_author_id, year, candidates["flow"], view="flow"),
        }

    # ------------------------------------------------------------------
    # Per-view refinement
    # ------------------------------------------------------------------
    def _refine_view(
        self,
        target_author_id: str,
        year: int,
        view_candidates: Dict[str, List[Dict[str, Any]]],
        view: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        papers = self._refine_papers(target_author_id, year, view_candidates.get("papers", []), view)
        authors = self._refine_authors(target_author_id, year, view_candidates.get("authors", []), view)
        topics = self._refine_topics(target_author_id, year, view_candidates.get("topics", []), view)
        venues = self._refine_venues(target_author_id, year, view_candidates.get("venues", []), view)

        return {
            "papers": papers,
            "authors": authors,
            "topics": topics,
            "venues": venues,
        }

    # ------------------------------------------------------------------
    # Paper refinement
    # ------------------------------------------------------------------
    def _refine_papers(
        self,
        target_author_id: str,
        year: int,
        paper_nodes: List[Dict[str, Any]],
        view: str,
    ) -> List[Dict[str, Any]]:
        limit = self.refined_stock_papers if view == "stock" else self.refined_flow_papers
        if not paper_nodes:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for node in paper_nodes:
            pid = str(node["id"]).strip()
            py = self.rm.get_paper_year(pid)
            if py is None or py > year:
                continue

            age = max(year - py, 0)
            cited = self.rm.get_paper_citations_until(pid, year)
            authors = self.rm.get_paper_authors(pid)
            topics = self.rm.get_paper_topics(pid)
            venue = self.rm.get_paper_venue(pid)

            coauthor_ids = [aid for aid in authors if aid != target_author_id]
            coauthor_count = len(coauthor_ids)

            new_coauthor_count = sum(
                self.rm.is_new_coauthor_for_author(target_author_id, aid, year, 3) for aid in coauthor_ids
            )
            new_topic_count = sum(
                self.rm.is_new_topic_for_author(target_author_id, tp, year, 3) for tp in topics
            )
            new_venue_flag = self.rm.is_new_venue_for_author(target_author_id, venue, year, 3)

            collab_backbone = sum(
                self.rm.get_collab_count_until(target_author_id, aid, year) for aid in coauthor_ids
            )
            topic_recurrence = sum(
                self.rm.get_topic_count_until(target_author_id, tp, year) for tp in topics
            )
            venue_recurrence = self.rm.get_venue_count_until(target_author_id, venue, year)

            base_score = float(node.get("score", 0.0))

            if view == "stock":
                score = (
                    1.4 * base_score
                    + 1.8 * self._log1p(cited)
                    + 1.0 * self._log1p(collab_backbone)
                    + 0.7 * self._log1p(topic_recurrence)
                    + 0.7 * self._log1p(venue_recurrence)
                    + 0.3 * self._log1p(coauthor_count)
                    - 0.10 * age
                )
                
            else:
                recent_momentum = self._recent_momentum(age)
                expansion_strength = (
                    1.3 * self._safe_ratio(new_coauthor_count, max(coauthor_count, 1))
                    + 1.1 * self._safe_ratio(new_topic_count, max(len(topics), 1))
                    + 0.9 * float(new_venue_flag)
                )
                score = (
                    1.5 * base_score
                    + 1.8 * recent_momentum
                    + 1.4 * expansion_strength
                    + 0.3 * self._log1p(cited)
                    + 0.2 * self._log1p(coauthor_count)
                )
                

            out = dict(node)
            out["refined_score"] = float(score)
            out["stock_score"] = float(score) if view == "stock" else float(node.get("stock_score", 0.0))
            out["flow_score"] = float(score) if view == "flow" else float(node.get("flow_score", 0.0))
            scored.append((score, out))

        if view == "stock":
            scored.sort(key=lambda x: (-x[0], x[1].get("year", 9999), x[1]["id"]))
        else:
            scored.sort(key=lambda x: (-x[0], -x[1].get("year", -1), x[1]["id"]))

        return [x[1] for x in scored[:limit]]

    # ------------------------------------------------------------------
    # Author refinement
    # ------------------------------------------------------------------
    def _refine_authors(
        self,
        target_author_id: str,
        year: int,
        author_nodes: List[Dict[str, Any]],
        view: str,
    ) -> List[Dict[str, Any]]:
        limit = self.refined_stock_authors if view == "stock" else self.refined_flow_authors
        if not author_nodes:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for node in author_nodes:
            aid = str(node["id"]).strip()
            base_score = float(node.get("score", 0.0))

            collab_total = self.rm.get_collab_count_until(target_author_id, aid, year)
            collab_recent = self.rm.get_collab_count_window(target_author_id, aid, year, 3)
            years_since_last = self.rm.get_years_since_last_collab(target_author_id, aid, year)

            cum_h = self.rm.get_author_hindex_until(aid, year)
            recent_papers = self.rm.get_author_pub_count_window(aid, year, 3)
            recent_citations = self.rm.get_author_citation_count_window(aid, year, 3)
            is_new = self.rm.is_new_coauthor_for_author(target_author_id, aid, year, 3)

            if view == "stock":
                score = (
                    1.4 * base_score
                    + 1.8 * self._log1p(collab_total)
                    + 0.9 * self._log1p(cum_h)
                    + 0.4 * self._log1p(recent_papers)
                    - 0.05 * min(years_since_last, 20)
                )
                
            else:
                score = (
                    1.5 * base_score
                    + 1.5 * self._log1p(collab_recent)
                    + 0.9 * self._log1p(recent_papers)
                    + 0.8 * self._log1p(recent_citations)
                    + 0.8 * float(is_new)
                    - 0.03 * min(years_since_last, 20)
                )
                

            out = dict(node)
            out["refined_score"] = float(score)
            scored.append((score, out))

        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [x[1] for x in scored[:limit]]

    # ------------------------------------------------------------------
    # Topic refinement
    # ------------------------------------------------------------------
    def _refine_topics(
        self,
        target_author_id: str,
        year: int,
        topic_nodes: List[Dict[str, Any]],
        view: str,
    ) -> List[Dict[str, Any]]:
        limit = self.refined_stock_topics if view == "stock" else self.refined_flow_topics
        if not topic_nodes:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for node in topic_nodes:
            topic = str(node.get("name", "") or node.get("id", "")).strip().lower()
            base_score = float(node.get("score", 0.0))

            total = self.rm.get_topic_count_until(target_author_id, topic, year)
            recent = self.rm.get_topic_count_window(target_author_id, topic, year, 3)
            span = self.rm.get_topic_span_years_until(target_author_id, topic, year)
            gap = self.rm.get_years_since_last_topic_use(target_author_id, topic, year)
            is_new = self.rm.is_new_topic_for_author(target_author_id, topic, year, 3)

            if view == "stock":
                score = (
                    1.3 * base_score
                    + 1.4 * self._log1p(total)
                    + 0.7 * self._log1p(span)
                    + 0.4 * self._log1p(recent)
                    - 0.03 * min(gap, 20)
                )
                
            else:
                score = (
                    1.4 * base_score
                    + 1.2 * self._log1p(recent)
                    + 1.0 * float(is_new)
                    - 0.02 * min(gap, 20)
                )
                

            out = dict(node)
            out["refined_score"] = float(score)
            scored.append((score, out))

        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [x[1] for x in scored[:limit]]

    # ------------------------------------------------------------------
    # Venue refinement
    # ------------------------------------------------------------------
    def _refine_venues(
        self,
        target_author_id: str,
        year: int,
        venue_nodes: List[Dict[str, Any]],
        view: str,
    ) -> List[Dict[str, Any]]:
        limit = self.refined_stock_venues if view == "stock" else self.refined_flow_venues
        if not venue_nodes:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for node in venue_nodes:
            venue = str(node.get("name", "") or node.get("id", "")).strip().lower()
            base_score = float(node.get("score", 0.0))

            total = self.rm.get_venue_count_until(target_author_id, venue, year)
            recent = self.rm.get_venue_count_window(target_author_id, venue, year, 3)
            span = self.rm.get_venue_span_years_until(target_author_id, venue, year)
            gap = self.rm.get_years_since_last_venue_use(target_author_id, venue, year)
            is_new = self.rm.is_new_venue_for_author(target_author_id, venue, year, 3)

            if view == "stock":
                score = (
                    1.3 * base_score
                    + 1.4 * self._log1p(total)
                    + 0.7 * self._log1p(span)
                    + 0.4 * self._log1p(recent)
                    - 0.03 * min(gap, 20)
                )
                
            else:
                score = (
                    1.4 * base_score
                    + 1.1 * self._log1p(recent)
                    + 1.0 * float(is_new)
                    - 0.02 * min(gap, 20)
                )
                

            out = dict(node)
            out["refined_score"] = float(score)
            scored.append((score, out))

        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [x[1] for x in scored[:limit]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        den = max(float(den), 1e-12)
        return float(num) / den

    @staticmethod
    def _log1p(x: float) -> float:
        import math
        return math.log1p(max(float(x), 0.0))

    @staticmethod
    def _recent_momentum(age: int) -> float:
        if age <= 0:
            return 1.0
        if age == 1:
            return 0.8
        if age == 2:
            return 0.55
        if age == 3:
            return 0.25
        return 0.0


if __name__ == "__main__":
    from utils.resource_manager_v3 import ResourceManagerV3
    from subgraph.candidate_builder_v3 import CandidateBuilderV3

    rm = ResourceManagerV3("aps")
    cb = CandidateBuilderV3(rm)
    ds = DynamicScorerV3(rm)

    aid = rm.author_ids_history[0]
    cands = cb.build_candidates(aid, 2014)
    refined = ds.refine(aid, 2014, cands)

    print("stock refined papers :", len(refined["stock"]["papers"]))
    print("stock refined authors:", len(refined["stock"]["authors"]))
    print("stock refined topics :", len(refined["stock"]["topics"]))
    print("stock refined venues :", len(refined["stock"]["venues"]))

    print("flow refined papers  :", len(refined["flow"]["papers"]))
    print("flow refined authors :", len(refined["flow"]["authors"]))
    print("flow refined topics  :", len(refined["flow"]["topics"]))
    print("flow refined venues  :", len(refined["flow"]["venues"]))