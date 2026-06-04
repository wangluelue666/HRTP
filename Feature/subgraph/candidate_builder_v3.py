#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

from utils.resource_manager_v3 import ResourceManagerV3


class CandidateBuilderV3:
    """
    Build stock / flow candidates for a target author at a given year.

    Design goals
    ------------
    1. Stock view emphasizes accumulation, stability, and long-term backbone.
    2. Flow view emphasizes recent activity, expansion, and novelty.
    3. Candidate stage should be task-aligned but still lightweight.
    4. No neural scorer is used here. This stage provides stronger structured
       candidates for later dynamic re-scoring.
    """

    def __init__(self, resource_manager: ResourceManagerV3) -> None:
        self.rm = resource_manager

        # candidate size controls
        self.max_stock_papers = 80
        self.max_flow_papers = 80

        self.max_stock_authors = 60
        self.max_flow_authors = 60

        self.max_stock_topics = 40
        self.max_flow_topics = 40

        self.max_stock_venues = 30
        self.max_flow_venues = 30

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_candidates(self, target_author_id: str, year: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Return format
        -------------
        {
          "stock": {
              "papers":  [ ... ],
              "authors": [ ... ],
              "topics":  [ ... ],
              "venues":  [ ... ],
          },
          "flow": {
              "papers":  [ ... ],
              "authors": [ ... ],
              "topics":  [ ... ],
              "venues":  [ ... ],
          }
        }
        """
        stock_papers = self._build_stock_papers(target_author_id, year)
        flow_papers = self._build_flow_papers(target_author_id, year)

        stock_authors, stock_topics, stock_venues = self._collect_side_nodes(
            target_author_id=target_author_id,
            year=year,
            paper_nodes=stock_papers,
            view="stock",
        )
        flow_authors, flow_topics, flow_venues = self._collect_side_nodes(
            target_author_id=target_author_id,
            year=year,
            paper_nodes=flow_papers,
            view="flow",
        )

        return {
            "stock": {
                "papers": stock_papers,
                "authors": stock_authors,
                "topics": stock_topics,
                "venues": stock_venues,
            },
            "flow": {
                "papers": flow_papers,
                "authors": flow_authors,
                "topics": flow_topics,
                "venues": flow_venues,
            },
        }

    # ------------------------------------------------------------------
    # Paper candidate construction
    # ------------------------------------------------------------------
    def _build_stock_papers(self, target_author_id: str, year: int) -> List[Dict[str, Any]]:
        papers = self.rm.get_author_papers_until(target_author_id, year)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for pid in papers:
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

            # stability / accumulation oriented
            long_term_citation_strength = cited
            collaboration_backbone = sum(
                self.rm.get_collab_count_until(target_author_id, aid, year) for aid in coauthor_ids
            )
            topic_recurrence = sum(
                self.rm.get_topic_count_until(target_author_id, tp, year) for tp in topics
            )
            venue_recurrence = self.rm.get_venue_count_until(target_author_id, venue, year)

            score = (
                2.2 * self._log1p(long_term_citation_strength)
                + 1.3 * self._log1p(collaboration_backbone)
                + 0.8 * self._log1p(topic_recurrence)
                + 0.8 * self._log1p(venue_recurrence)
                + 0.4 * self._log1p(coauthor_count)
                - 0.12 * age
            )

            scored.append(
                (
                    score,
                    {
                        "id": pid,
                        "year": py,
                        "score": float(score),
                        "stock_score": float(score),
                        "flow_score": 0.0,
                        "paper_citations": int(cited),
                        "coauthor_count": int(coauthor_count),
                        "topics": list(topics),
                        "venue": venue,
                    },
                )
            )

        scored.sort(key=lambda x: (-x[0], x[1]["year"], x[1]["id"]))
        return [x[1] for x in scored[: self.max_stock_papers]]

    def _build_flow_papers(self, target_author_id: str, year: int) -> List[Dict[str, Any]]:
        papers = self.rm.get_author_papers_until(target_author_id, year)
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for pid in papers:
            py = self.rm.get_paper_year(pid)
            if py is None or py > year:
                continue

            age = max(year - py, 0)
            authors = self.rm.get_paper_authors(pid)
            topics = self.rm.get_paper_topics(pid)
            venue = self.rm.get_paper_venue(pid)

            coauthor_ids = [aid for aid in authors if aid != target_author_id]
            coauthor_count = len(coauthor_ids)

            recent_bonus = 1.0 if age <= 0 else (0.8 if age <= 1 else (0.5 if age <= 2 else 0.0))

            new_coauthor_count = sum(
                self.rm.is_new_coauthor_for_author(target_author_id, aid, year, 3) for aid in coauthor_ids
            )
            new_topic_count = sum(
                self.rm.is_new_topic_for_author(target_author_id, tp, year, 3) for tp in topics
            )
            new_venue_flag = self.rm.is_new_venue_for_author(target_author_id, venue, year, 3)

            # recent growth / expansion oriented
            score = (
                2.0 * recent_bonus
                + 1.5 * self._safe_ratio(new_coauthor_count, max(coauthor_count, 1))
                + 1.2 * self._safe_ratio(new_topic_count, max(len(topics), 1))
                + 0.9 * float(new_venue_flag)
                + 0.35 * self._log1p(coauthor_count)
            )

            scored.append(
                (
                    score,
                    {
                        "id": pid,
                        "year": py,
                        "score": float(score),
                        "stock_score": 0.0,
                        "flow_score": float(score),
                        "paper_citations": int(self.rm.get_paper_citations_until(pid, year)),
                        "coauthor_count": int(coauthor_count),
                        "topics": list(topics),
                        "venue": venue,
                    },
                )
            )

        scored.sort(key=lambda x: (-x[0], -x[1]["year"], x[1]["id"]))
        return [x[1] for x in scored[: self.max_flow_papers]]

    # ------------------------------------------------------------------
    # Side node collection
    # ------------------------------------------------------------------
    def _collect_side_nodes(
        self,
        target_author_id: str,
        year: int,
        paper_nodes: List[Dict[str, Any]],
        view: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        author_scores: DefaultDict[str, float] = defaultdict(float)
        topic_scores: DefaultDict[str, float] = defaultdict(float)
        venue_scores: DefaultDict[str, float] = defaultdict(float)

        used_papers: Set[str] = set()

        for p in paper_nodes:
            pid = p["id"]
            used_papers.add(pid)
            base_score = float(p.get("stock_score", 0.0) if view == "stock" else p.get("flow_score", 0.0))

            authors = self.rm.get_paper_authors(pid)
            topics = self.rm.get_paper_topics(pid)
            venue = self.rm.get_paper_venue(pid)

            for aid in authors:
                if aid == target_author_id:
                    continue
                author_scores[aid] += self._score_author_side_node(
                    target_author_id=target_author_id,
                    coauthor_id=aid,
                    year=year,
                    base_score=base_score,
                    view=view,
                )

            for tp in topics:
                topic_scores[tp] += self._score_topic_side_node(
                    target_author_id=target_author_id,
                    topic=tp,
                    year=year,
                    base_score=base_score,
                    view=view,
                )

            venue_scores[venue] += self._score_venue_side_node(
                target_author_id=target_author_id,
                venue=venue,
                year=year,
                base_score=base_score,
                view=view,
            )

        author_limit = self.max_stock_authors if view == "stock" else self.max_flow_authors
        topic_limit = self.max_stock_topics if view == "stock" else self.max_flow_topics
        venue_limit = self.max_stock_venues if view == "stock" else self.max_flow_venues

        author_nodes = [
            {"id": aid, "score": float(score)}
            for aid, score in sorted(author_scores.items(), key=lambda x: (-x[1], x[0]))[:author_limit]
        ]
        topic_nodes = [
            {"id": tp, "name": tp, "score": float(score)}
            for tp, score in sorted(topic_scores.items(), key=lambda x: (-x[1], x[0]))[:topic_limit]
        ]
        venue_nodes = [
            {"id": vn, "name": vn, "score": float(score)}
            for vn, score in sorted(venue_scores.items(), key=lambda x: (-x[1], x[0]))[:venue_limit]
        ]

        return author_nodes, topic_nodes, venue_nodes

    def _score_author_side_node(
        self,
        target_author_id: str,
        coauthor_id: str,
        year: int,
        base_score: float,
        view: str,
    ) -> float:
        collab_total = self.rm.get_collab_count_until(target_author_id, coauthor_id, year)
        collab_recent = self.rm.get_collab_count_window(target_author_id, coauthor_id, year, 3)
        coauthor_cum_h = self.rm.get_author_hindex_until(coauthor_id, year)
        coauthor_recent_papers = self.rm.get_author_pub_count_window(coauthor_id, year, 3)

        if view == "stock":
            score = (
                1.5 * base_score
                + 1.6 * self._log1p(collab_total)
                + 0.9 * self._log1p(coauthor_cum_h)
                + 0.4 * self._log1p(coauthor_recent_papers)
            )
        else:
            is_new = self.rm.is_new_coauthor_for_author(target_author_id, coauthor_id, year, 3)
            score = (
                1.4 * base_score
                + 1.2 * self._log1p(collab_recent)
                + 1.0 * float(is_new)
                + 0.7 * self._log1p(coauthor_recent_papers)
            )
        return float(score)

    def _score_topic_side_node(
        self,
        target_author_id: str,
        topic: str,
        year: int,
        base_score: float,
        view: str,
    ) -> float:
        total = self.rm.get_topic_count_until(target_author_id, topic, year)
        recent = self.rm.get_topic_count_window(target_author_id, topic, year, 3)
        is_new = self.rm.is_new_topic_for_author(target_author_id, topic, year, 3)

        if view == "stock":
            score = 1.4 * base_score + 1.3 * self._log1p(total) + 0.5 * self._log1p(recent)
        else:
            score = 1.3 * base_score + 0.8 * self._log1p(recent) + 1.1 * float(is_new)
        return float(score)

    def _score_venue_side_node(
        self,
        target_author_id: str,
        venue: str,
        year: int,
        base_score: float,
        view: str,
    ) -> float:
        total = self.rm.get_venue_count_until(target_author_id, venue, year)
        recent = self.rm.get_venue_count_window(target_author_id, venue, year, 3)
        is_new = self.rm.is_new_venue_for_author(target_author_id, venue, year, 3)

        if view == "stock":
            score = 1.4 * base_score + 1.2 * self._log1p(total) + 0.5 * self._log1p(recent)
        else:
            score = 1.3 * base_score + 0.9 * self._log1p(recent) + 1.0 * float(is_new)
        return float(score)

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


if __name__ == "__main__":
    from utils.resource_manager_v3 import ResourceManagerV3

    rm = ResourceManagerV3("aps")
    builder = CandidateBuilderV3(rm)

    aid = rm.author_ids_history[0]
    out = builder.build_candidates(aid, 2014)

    print("stock papers :", len(out["stock"]["papers"]))
    print("stock authors:", len(out["stock"]["authors"]))
    print("stock topics :", len(out["stock"]["topics"]))
    print("stock venues :", len(out["stock"]["venues"]))

    print("flow papers  :", len(out["flow"]["papers"]))
    print("flow authors :", len(out["flow"]["authors"]))
    print("flow topics  :", len(out["flow"]["topics"]))
    print("flow venues  :", len(out["flow"]["venues"]))