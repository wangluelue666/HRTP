#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional

from subgraph.candidate_builder_v3 import CandidateBuilderV3
from subgraph.dynamic_scorer_v3 import DynamicScorerV3
from subgraph.ring_constructor_v3 import RingConstructorV3
from utils.resource_manager_v3 import ResourceManagerV3


class AuthorSubgraphSamplerV3:
    """
    End-to-end paired subgraph sampler for my_method_v3.

    Pipeline
    --------
    target author-year
        -> stage-1 candidates
        -> stage-2 refined candidates
        -> paired stock/flow subgraphs

    This class is the main entry point for later cache building.
    """

    def __init__(self, resource_manager: ResourceManagerV3) -> None:
        self.rm = resource_manager
        self.candidate_builder = CandidateBuilderV3(resource_manager)
        self.dynamic_scorer = DynamicScorerV3(resource_manager)
        self.ring_constructor = RingConstructorV3(resource_manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self, target_author_id: str, year: int) -> Dict[str, Any]:
        """
        Build a paired stock/flow subgraph for one (author, year).

        Return format
        -------------
        {
            "author_id": ...,
            "year": ...,
            "stock": {...},
            "flow": {...},
            "meta": {...},
        }
        """
        candidates = self.candidate_builder.build_candidates(target_author_id, year)
        refined = self.dynamic_scorer.refine(target_author_id, year, candidates)
        paired = self.ring_constructor.construct(target_author_id, year, refined)

        return {
            "author_id": target_author_id,
            "year": int(year),
            "stock": paired["stock"],
            "flow": paired["flow"],
            "meta": self._build_meta(target_author_id, year, candidates, refined, paired),
        }

    def sample_by_author_index(self, author_idx: int, year: int) -> Dict[str, Any]:
        author_id = self.rm.author_ids_history[author_idx]
        return self.sample(author_id, year)

    # ------------------------------------------------------------------
    # Meta / debug info
    # ------------------------------------------------------------------
    def _build_meta(
        self,
        target_author_id: str,
        year: int,
        candidates: Dict[str, Dict[str, Any]],
        refined: Dict[str, Dict[str, Any]],
        paired: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "target_author_id": target_author_id,
            "year": int(year),
            "candidate_sizes": {
                "stock": self._view_sizes(candidates["stock"]),
                "flow": self._view_sizes(candidates["flow"]),
            },
            "refined_sizes": {
                "stock": self._view_sizes(refined["stock"]),
                "flow": self._view_sizes(refined["flow"]),
            },
            "paired_sizes": {
                "stock": self._paired_view_sizes(paired["stock"]),
                "flow": self._paired_view_sizes(paired["flow"]),
            },
        }

    @staticmethod
    def _view_sizes(view_dict: Dict[str, Any]) -> Dict[str, int]:
        return {
            "papers": len(view_dict.get("papers", [])),
            "authors": len(view_dict.get("authors", [])),
            "topics": len(view_dict.get("topics", [])),
            "venues": len(view_dict.get("venues", [])),
        }

    @staticmethod
    def _paired_view_sizes(view_dict: Dict[str, Any]) -> Dict[str, int]:
        ring0 = view_dict.get("ring0", {})
        ring1 = view_dict.get("ring1", {})
        ring2 = view_dict.get("ring2", {})
        return {
            "ring0_target": int("target" in ring0),
            "ring1_papers": len(ring1.get("papers", [])),
            "ring1_authors": len(ring1.get("authors", [])),
            "ring1_topics": len(ring1.get("topics", [])),
            "ring1_venues": len(ring1.get("venues", [])),
            "ring1_pa_edges": len(ring1.get("paper_author_edges", [])),
            "ring1_pt_edges": len(ring1.get("paper_topic_edges", [])),
            "ring1_pv_edges": len(ring1.get("paper_venue_edges", [])),
            "ring2_papers": len(ring2.get("papers", [])),
            "ring2_authors": len(ring2.get("authors", [])),
            "ring2_topics": len(ring2.get("topics", [])),
            "ring2_venues": len(ring2.get("venues", [])),
            "ring2_pa_edges": len(ring2.get("paper_author_edges", [])),
            "ring2_pt_edges": len(ring2.get("paper_topic_edges", [])),
            "ring2_pv_edges": len(ring2.get("paper_venue_edges", [])),
        }


if __name__ == "__main__":
    from utils.resource_manager_v3 import ResourceManagerV3

    rm = ResourceManagerV3("aps")
    sampler = AuthorSubgraphSamplerV3(rm)

    aid = rm.author_ids_history[0]
    year = 2014
    out = sampler.sample(aid, year)

    print("author_id =", out["author_id"])
    print("year      =", out["year"])
    print("meta      =", out["meta"])