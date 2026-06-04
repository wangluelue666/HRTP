#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from subgraph.feature_builder_v3 import FeatureBuilderV3
from utils.resource_manager_v3 import ResourceManagerV3


class RingConstructorV3:
    """
    Construct paired stock / flow subgraphs with:
      - ring0: target author
      - ring1: refined core papers/authors/topics/venues + memberships
      - ring2: lightweight structural context + memberships

    Notes
    -----
    1. ring1 is the main structured layer.
    2. ring2 is not a full expansion graph, but it is no longer just a weak set.
    3. All node features are precomputed here and written into the subgraph dict.
    """

    def __init__(self, resource_manager: ResourceManagerV3) -> None:
        self.rm = resource_manager
        self.fb = FeatureBuilderV3(resource_manager)

        # ring2 size controls
        self.max_ring2_papers = 20
        self.max_ring2_authors = 24
        self.max_ring2_topics = 16
        self.max_ring2_venues = 12

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def construct(
        self,
        target_author_id: str,
        year: int,
        refined: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Dict[str, Any]]:
        return {
            "stock": self._construct_single_view(
                target_author_id=target_author_id,
                year=year,
                refined_view=refined["stock"],
                view="stock",
            ),
            "flow": self._construct_single_view(
                target_author_id=target_author_id,
                year=year,
                refined_view=refined["flow"],
                view="flow",
            ),
        }

    # ------------------------------------------------------------------
    # Per-view constructor
    # ------------------------------------------------------------------
    def _construct_single_view(
        self,
        target_author_id: str,
        year: int,
        refined_view: Dict[str, List[Dict[str, Any]]],
        view: str,
    ) -> Dict[str, Any]:
        ring0 = self._build_ring0(target_author_id, year)

        ring1 = self._build_ring1(
            target_author_id=target_author_id,
            year=year,
            refined_view=refined_view,
            view=view,
        )

        ring2 = self._build_ring2(
            target_author_id=target_author_id,
            year=year,
            ring1=ring1,
            view=view,
        )

        return {
            "target_author_id": target_author_id,
            "year": int(year),
            "view": view,
            "ring0": ring0,
            "ring1": ring1,
            "ring2": ring2,
        }

    # ------------------------------------------------------------------
    # Ring0
    # ------------------------------------------------------------------
    def _build_ring0(self, target_author_id: str, year: int) -> Dict[str, Any]:
        node = {
            "id": target_author_id,
            "type": "target",
        }
        node = self.fb.enrich_target_node(target_author_id, year, node)
        return {
            "target": node,
        }

    # ------------------------------------------------------------------
    # Ring1
    # ------------------------------------------------------------------
    def _build_ring1(
        self,
        target_author_id: str,
        year: int,
        refined_view: Dict[str, List[Dict[str, Any]]],
        view: str,
    ) -> Dict[str, Any]:
        paper_nodes_raw = refined_view.get("papers", [])
        author_nodes_raw = refined_view.get("authors", [])
        topic_nodes_raw = refined_view.get("topics", [])
        venue_nodes_raw = refined_view.get("venues", [])

        # ----- enrich nodes -----
        papers = []
        for p in paper_nodes_raw:
            papers.append(
                self.fb.enrich_paper_node(
                    target_author_id=target_author_id,
                    paper_node=p,
                    year=year,
                    stock_score=float(p.get("stock_score", 0.0)),
                    flow_score=float(p.get("flow_score", 0.0)),
                )
            )

        authors = []
        for a in author_nodes_raw:
            authors.append(
                self.fb.enrich_author_node(
                    target_author_id=target_author_id,
                    author_node=a,
                    year=year,
                )
            )

        topics = []
        for t in topic_nodes_raw:
            topics.append(
                self.fb.enrich_topic_node(
                    target_author_id=target_author_id,
                    topic_node=t,
                    year=year,
                )
            )

        venues = []
        for v in venue_nodes_raw:
            venues.append(
                self.fb.enrich_venue_node(
                    target_author_id=target_author_id,
                    venue_node=v,
                    year=year,
                )
            )

        # ----- build id -> local index -----
        paper_idx = {x["id"]: i for i, x in enumerate(papers)}
        author_idx = {x["id"]: i for i, x in enumerate(authors)}
        topic_idx = {x["id"]: i for i, x in enumerate(topics)}
        venue_idx = {x["id"]: i for i, x in enumerate(venues)}

        # ----- memberships -----
        paper_author_edges: List[List[int]] = []
        paper_topic_edges: List[List[int]] = []
        paper_venue_edges: List[List[int]] = []

        for p in papers:
            pid = p["id"]

            for aid in self.rm.get_paper_authors(pid):
                if aid == target_author_id:
                    continue
                if aid in author_idx:
                    paper_author_edges.append([paper_idx[pid], author_idx[aid]])

            for tp in self.rm.get_paper_topics(pid):
                if tp in topic_idx:
                    paper_topic_edges.append([paper_idx[pid], topic_idx[tp]])

            venue = self.rm.get_paper_venue(pid)
            if venue in venue_idx:
                paper_venue_edges.append([paper_idx[pid], venue_idx[venue]])

        return {
            "papers": papers,
            "authors": authors,
            "topics": topics,
            "venues": venues,
            "paper_author_edges": paper_author_edges,
            "paper_topic_edges": paper_topic_edges,
            "paper_venue_edges": paper_venue_edges,
        }

    # ------------------------------------------------------------------
    # Ring2
    # ------------------------------------------------------------------
    def _build_ring2(
        self,
        target_author_id: str,
        year: int,
        ring1: Dict[str, Any],
        view: str,
    ) -> Dict[str, Any]:
        ring1_pids = [x["id"] for x in ring1["papers"]]
        ring1_authors = {x["id"] for x in ring1["authors"]}
        ring1_topics = {x["id"] for x in ring1["topics"]}
        ring1_venues = {x["id"] for x in ring1["venues"]}

        ring2_paper_scores: Dict[str, float] = {}
        ring2_author_scores: Dict[str, float] = {}
        ring2_topic_scores: Dict[str, float] = {}
        ring2_venue_scores: Dict[str, float] = {}

        # expand from ring1 papers through references and shared local structure
        for pid in ring1_pids:
            base_score = self._get_paper_node_score(ring1["papers"], pid, view)

            # referenced papers as extra context
            for ref_pid in self.rm.get_paper_references(pid):
                py = self.rm.get_paper_year(ref_pid)
                if py is None or py > year:
                    continue
                if ref_pid in ring1_pids:
                    continue
                ring2_paper_scores[ref_pid] = ring2_paper_scores.get(ref_pid, 0.0) + 0.7 * base_score

            # one-hop side context from ring1 papers
            for aid in self.rm.get_paper_authors(pid):
                if aid == target_author_id or aid in ring1_authors:
                    continue
                ring2_author_scores[aid] = ring2_author_scores.get(aid, 0.0) + 1.0 * base_score

            for tp in self.rm.get_paper_topics(pid):
                if tp in ring1_topics:
                    continue
                ring2_topic_scores[tp] = ring2_topic_scores.get(tp, 0.0) + 0.9 * base_score

            venue = self.rm.get_paper_venue(pid)
            if venue not in ring1_venues:
                ring2_venue_scores[venue] = ring2_venue_scores.get(venue, 0.0) + 0.8 * base_score

        # enrich ring2 paper candidates with local task-aligned bonuses
        ranked_ring2_papers = []
        for pid, score in ring2_paper_scores.items():
            py = self.rm.get_paper_year(pid)
            if py is None:
                continue
            age = max(year - py, 0)
            recent_bonus = self._recent_bonus(age)
            cited = self.rm.get_paper_citations_until(pid, year)
            final_score = score + 0.4 * recent_bonus + 0.25 * self._log1p(cited)
            ranked_ring2_papers.append((final_score, pid))

        ranked_ring2_authors = []
        for aid, score in ring2_author_scores.items():
            recent = self.rm.get_author_pub_count_window(aid, year, 3)
            final_score = score + 0.35 * self._log1p(recent)
            ranked_ring2_authors.append((final_score, aid))

        ranked_ring2_topics = []
        for tp, score in ring2_topic_scores.items():
            recent = self.rm.get_topic_count_window(target_author_id, tp, year, 3)
            final_score = score + 0.3 * self._log1p(recent)
            ranked_ring2_topics.append((final_score, tp))

        ranked_ring2_venues = []
        for vn, score in ring2_venue_scores.items():
            recent = self.rm.get_venue_count_window(target_author_id, vn, year, 3)
            final_score = score + 0.3 * self._log1p(recent)
            ranked_ring2_venues.append((final_score, vn))

        ranked_ring2_papers.sort(key=lambda x: (-x[0], x[1]))
        ranked_ring2_authors.sort(key=lambda x: (-x[0], x[1]))
        ranked_ring2_topics.sort(key=lambda x: (-x[0], x[1]))
        ranked_ring2_venues.sort(key=lambda x: (-x[0], x[1]))

        ring2_papers_raw = ranked_ring2_papers[: self.max_ring2_papers]
        ring2_authors_raw = ranked_ring2_authors[: self.max_ring2_authors]
        ring2_topics_raw = ranked_ring2_topics[: self.max_ring2_topics]
        ring2_venues_raw = ranked_ring2_venues[: self.max_ring2_venues]

        # ----- enrich ring2 nodes -----
        papers = []
        for score, pid in ring2_papers_raw:
            paper_node = {
                "id": pid,
                "year": self.rm.get_paper_year(pid),
                "score": float(score),
                "stock_score": float(score) if view == "stock" else 0.0,
                "flow_score": float(score) if view == "flow" else 0.0,
            }
            papers.append(
                self.fb.enrich_paper_node(
                    target_author_id=target_author_id,
                    paper_node=paper_node,
                    year=year,
                    stock_score=float(paper_node["stock_score"]),
                    flow_score=float(paper_node["flow_score"]),
                )
            )

        authors = []
        for score, aid in ring2_authors_raw:
            author_node = {"id": aid, "score": float(score)}
            authors.append(
                self.fb.enrich_author_node(
                    target_author_id=target_author_id,
                    author_node=author_node,
                    year=year,
                )
            )

        topics = []
        for score, tp in ring2_topics_raw:
            topic_node = {"id": tp, "name": tp, "score": float(score)}
            topics.append(
                self.fb.enrich_topic_node(
                    target_author_id=target_author_id,
                    topic_node=topic_node,
                    year=year,
                )
            )

        venues = []
        for score, vn in ring2_venues_raw:
            venue_node = {"id": vn, "name": vn, "score": float(score)}
            venues.append(
                self.fb.enrich_venue_node(
                    target_author_id=target_author_id,
                    venue_node=venue_node,
                    year=year,
                )
            )

        # ----- ring2 memberships -----
        paper_idx = {x["id"]: i for i, x in enumerate(papers)}
        author_idx = {x["id"]: i for i, x in enumerate(authors)}
        topic_idx = {x["id"]: i for i, x in enumerate(topics)}
        venue_idx = {x["id"]: i for i, x in enumerate(venues)}

        paper_author_edges: List[List[int]] = []
        paper_topic_edges: List[List[int]] = []
        paper_venue_edges: List[List[int]] = []

        for p in papers:
            pid = p["id"]

            for aid in self.rm.get_paper_authors(pid):
                if aid in author_idx:
                    paper_author_edges.append([paper_idx[pid], author_idx[aid]])

            for tp in self.rm.get_paper_topics(pid):
                if tp in topic_idx:
                    paper_topic_edges.append([paper_idx[pid], topic_idx[tp]])

            venue = self.rm.get_paper_venue(pid)
            if venue in venue_idx:
                paper_venue_edges.append([paper_idx[pid], venue_idx[venue]])

        return {
            "papers": papers,
            "authors": authors,
            "topics": topics,
            "venues": venues,
            "paper_author_edges": paper_author_edges,
            "paper_topic_edges": paper_topic_edges,
            "paper_venue_edges": paper_venue_edges,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log1p(x: float) -> float:
        import math
        return math.log1p(max(float(x), 0.0))

    @staticmethod
    def _recent_bonus(age: int) -> float:
        if age <= 0:
            return 1.0
        if age == 1:
            return 0.8
        if age == 2:
            return 0.5
        if age == 3:
            return 0.2
        return 0.0

    @staticmethod
    def _get_paper_node_score(paper_nodes: List[Dict[str, Any]], pid: str, view: str) -> float:
        for node in paper_nodes:
            if node["id"] == pid:
                if view == "stock":
                    return float(node.get("stock_score", node.get("refined_score", node.get("score", 0.0))))
                return float(node.get("flow_score", node.get("refined_score", node.get("score", 0.0))))
        return 0.0


if __name__ == "__main__":
    from subgraph.candidate_builder_v3 import CandidateBuilderV3
    from subgraph.dynamic_scorer_v3 import DynamicScorerV3
    from utils.resource_manager_v3 import ResourceManagerV3

    rm = ResourceManagerV3("aps")
    cb = CandidateBuilderV3(rm)
    ds = DynamicScorerV3(rm)
    rc = RingConstructorV3(rm)

    aid = rm.author_ids_history[0]
    year = 2014

    cands = cb.build_candidates(aid, year)
    refined = ds.refine(aid, year, cands)
    paired = rc.construct(aid, year, refined)

    for view in ["stock", "flow"]:
        sg = paired[view]
        print(f"[{view}]")
        print("  ring1 papers :", len(sg["ring1"]["papers"]))
        print("  ring1 authors:", len(sg["ring1"]["authors"]))
        print("  ring1 topics :", len(sg["ring1"]["topics"]))
        print("  ring1 venues :", len(sg["ring1"]["venues"]))
        print("  ring2 papers :", len(sg["ring2"]["papers"]))
        print("  ring2 authors:", len(sg["ring2"]["authors"]))
        print("  ring2 topics :", len(sg["ring2"]["topics"]))
        print("  ring2 venues :", len(sg["ring2"]["venues"]))
        print("  ring1 pa edges:", len(sg["ring1"]["paper_author_edges"]))
        print("  ring1 pt edges:", len(sg["ring1"]["paper_topic_edges"]))
        print("  ring1 pv edges:", len(sg["ring1"]["paper_venue_edges"]))
        print("  ring2 pa edges:", len(sg["ring2"]["paper_author_edges"]))
        print("  ring2 pt edges:", len(sg["ring2"]["paper_topic_edges"]))
        print("  ring2 pv edges:", len(sg["ring2"]["paper_venue_edges"]))