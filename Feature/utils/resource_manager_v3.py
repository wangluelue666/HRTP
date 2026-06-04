#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np

from config.path_config_v3 import PATHS, DatasetPaths


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class ResourceManagerV3:
    """
    Unified resource manager for my_method_v3.

    Responsibilities
    ----------------
    1. Load fixed experiment inputs:
       - raw dataset json
       - history/eval labels
       - split file
       - v1 shared resources root paths

    2. Build reusable in-memory statistics for V3 subgraph construction:
       - paper -> metadata
       - author -> papers
       - author-year activity stats
       - collaboration stats
       - topic usage stats
       - venue usage stats
       - citation counts by time cutoff

    3. Provide clean query interfaces for feature_builder / candidate_builder.
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset.lower().strip()
        self.dp: DatasetPaths = PATHS.get_dataset_paths(self.dataset)
        PATHS.validate_required_inputs(self.dataset)

        # Fixed experiment artifacts
        self.author_ids_history: List[str] = []
        self.author_id_to_idx: Dict[str, int] = {}

        self.history_mask: Optional[np.ndarray] = None
        self.history_stock: Optional[np.ndarray] = None
        self.history_flow: Optional[np.ndarray] = None
        self.history_hs: Optional[np.ndarray] = None
        self.history_hf: Optional[np.ndarray] = None
        self.history_cls: Optional[np.ndarray] = None
        self.eval_cls: Optional[np.ndarray] = None
        self.splits: Optional[SplitIndices] = None

        # Raw parsed data
        self.records: List[Dict[str, Any]] = []

        # Paper-level metadata
        self.paper_year: Dict[str, int] = {}
        self.paper_authors: Dict[str, List[str]] = {}
        self.paper_topics: Dict[str, List[str]] = {}
        self.paper_venue: Dict[str, str] = {}
        self.paper_title: Dict[str, str] = {}
        self.paper_references: Dict[str, List[str]] = {}

        # Author-level structures
        self.author_papers: DefaultDict[str, List[str]] = defaultdict(list)
        self.author_year_papers: DefaultDict[str, DefaultDict[int, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.author_first_year: Dict[str, int] = {}
        self.author_last_year: Dict[str, int] = {}

        # Author cumulative / yearly stats
        self.author_year_pub_count: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.author_year_citation_count: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Collaboration / topic / venue stats
        self.author_coauthor_counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.author_topic_counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.author_venue_counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self.author_topic_years: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.author_venue_years: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.author_coauthor_years: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Citation event year approximation:
        # cited paper receives one citation at citing paper's publication year
        self.paper_cited_by_years: DefaultDict[str, List[int]] = defaultdict(list)

        # h-index related preparation
        # For each author, store paper ids to later compute h-index by cutoff year
        self._author_unique_papers: DefaultDict[str, Set[str]] = defaultdict(set)

        self._load_all()

    # ---------------------------------------------------------------------
    # Public loading
    # ---------------------------------------------------------------------
    def _load_all(self) -> None:
        self._load_author_ids()
        self._load_labels()
        self._load_splits()
        self._load_raw_records()
        self._build_core_indices()
        self._validate_alignment()

    def _load_author_ids(self) -> None:
        with open(self.dp.author_ids_history, "r", encoding="utf-8") as f:
            self.author_ids_history = json.load(f)
        self.author_id_to_idx = {aid: i for i, aid in enumerate(self.author_ids_history)}

    def _load_labels(self) -> None:
        self.history_mask = np.load(self.dp.history_mask)
        self.history_stock = np.load(self.dp.history_stock)
        self.history_flow = np.load(self.dp.history_flow)
        self.history_hs = np.load(self.dp.history_hs)
        self.history_hf = np.load(self.dp.history_hf)
        self.history_cls = np.load(self.dp.history_cls)
        self.eval_cls = np.load(self.dp.eval_cls)

    def _load_splits(self) -> None:
        with open(self.dp.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        self.splits = SplitIndices(
            train=np.array(split_data["train"], dtype=np.int64),
            val=np.array(split_data["val"], dtype=np.int64),
            test=np.array(split_data["test"], dtype=np.int64),
        )

    def _load_raw_records(self) -> None:
        records: List[Dict[str, Any]] = []
        with open(self.dp.raw_json, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        self.records = records

    # ---------------------------------------------------------------------
    # Index building
    # ---------------------------------------------------------------------
    def _build_core_indices(self) -> None:
        # Pass 1: paper metadata and author-paper links
        for rec in self.records:
            paper_id = str(rec.get("id", "")).strip()
            if not paper_id:
                continue

            year = self._safe_year(rec.get("year", None))
            if year is None:
                continue

            authors = rec.get("authors", []) or []
            author_ids = []
            for a in authors:
                aid = str(a.get("id", "")).strip()
                if aid:
                    author_ids.append(aid)

            raw_keywords = rec.get("keywords", []) or []
            topics = [self._normalize_token(x) for x in raw_keywords if self._normalize_token(x)]

            venue = self._normalize_token(rec.get("venue", "")) or "unknown"
            title = str(rec.get("title", "")).strip()
            references = [str(x).strip() for x in (rec.get("references", []) or []) if str(x).strip()]

            self.paper_year[paper_id] = year
            self.paper_authors[paper_id] = author_ids
            self.paper_topics[paper_id] = topics
            self.paper_venue[paper_id] = venue
            self.paper_title[paper_id] = title
            self.paper_references[paper_id] = references

            # author-paper structures
            for aid in author_ids:
                self.author_papers[aid].append(paper_id)
                self.author_year_papers[aid][year].append(paper_id)
                self.author_year_pub_count[aid][year] += 1
                self._author_unique_papers[aid].add(paper_id)

                if aid not in self.author_first_year or year < self.author_first_year[aid]:
                    self.author_first_year[aid] = year
                if aid not in self.author_last_year or year > self.author_last_year[aid]:
                    self.author_last_year[aid] = year

            # coauthor / topic / venue counters by author
            unique_author_ids = list(dict.fromkeys(author_ids))
            for aid in unique_author_ids:
                coauthors = [x for x in unique_author_ids if x != aid]
                for coaid in coauthors:
                    self.author_coauthor_counter[aid][coaid] += 1
                    self.author_coauthor_years[aid][coaid].append(year)

                for topic in topics:
                    self.author_topic_counter[aid][topic] += 1
                    self.author_topic_years[aid][topic].append(year)

                self.author_venue_counter[aid][venue] += 1
                self.author_venue_years[aid][venue].append(year)

        # Pass 2: citation event years
        # If citing paper published in year Y references paper P, then P gets one citation event at year Y.
        for citing_pid, refs in self.paper_references.items():
            citing_year = self.paper_year.get(citing_pid, None)
            if citing_year is None:
                continue
            for cited_pid in refs:
                if cited_pid in self.paper_year:
                    self.paper_cited_by_years[cited_pid].append(citing_year)

        # Pass 3: author yearly citation counts
        for aid, papers in self.author_papers.items():
            for pid in papers:
                cited_years = self.paper_cited_by_years.get(pid, [])
                for cy in cited_years:
                    self.author_year_citation_count[aid][cy] += 1

        # Sort all year lists once for stable later queries
        for aid in self.author_papers:
            self.author_papers[aid].sort(key=lambda pid: (self.paper_year.get(pid, 9999), pid))
            for y in self.author_year_papers[aid]:
                self.author_year_papers[aid][y].sort()

        for d in (self.author_topic_years, self.author_venue_years, self.author_coauthor_years):
            for aid in d:
                for key in d[aid]:
                    d[aid][key].sort()

        for pid in self.paper_cited_by_years:
            self.paper_cited_by_years[pid].sort()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def _validate_alignment(self) -> None:
        n = len(self.author_ids_history)

        assert self.history_mask is not None
        assert self.history_stock is not None
        assert self.history_flow is not None
        assert self.history_hs is not None
        assert self.history_hf is not None
        assert self.history_cls is not None
        assert self.eval_cls is not None
        assert self.splits is not None

        assert self.history_mask.shape[0] == n
        assert self.history_stock.shape[0] == n
        assert self.history_flow.shape[0] == n
        assert self.history_hs.shape[0] == n
        assert self.history_hf.shape[0] == n
        assert self.history_cls.shape[0] == n
        assert self.eval_cls.shape[0] == n

        all_idx = np.concatenate([self.splits.train, self.splits.val, self.splits.test], axis=0)
        unique_idx = np.unique(all_idx)
        if len(unique_idx) != n:
            raise ValueError(
                f"Split union size mismatch: got {len(unique_idx)}, expected {n}"
            )

    # ---------------------------------------------------------------------
    # Basic experiment inputs
    # ---------------------------------------------------------------------
    def get_split_indices(self, split: str) -> np.ndarray:
        assert self.splits is not None
        split = split.lower().strip()
        if split == "train":
            return self.splits.train
        if split == "val":
            return self.splits.val
        if split == "test":
            return self.splits.test
        raise ValueError(f"Unsupported split: {split}")

    def get_history_labels(self) -> Dict[str, np.ndarray]:
        return {
            "mask": self.history_mask,
            "stock": self.history_stock,
            "flow": self.history_flow,
            "hs": self.history_hs,
            "hf": self.history_hf,
            "cls": self.history_cls,
        }

    def get_eval_labels(self) -> np.ndarray:
        assert self.eval_cls is not None
        return self.eval_cls

    # ---------------------------------------------------------------------
    # Paper-level queries
    # ---------------------------------------------------------------------
    def get_paper_year(self, paper_id: str) -> Optional[int]:
        return self.paper_year.get(paper_id)

    def get_paper_authors(self, paper_id: str) -> List[str]:
        return list(self.paper_authors.get(paper_id, []))

    def get_paper_topics(self, paper_id: str) -> List[str]:
        return list(self.paper_topics.get(paper_id, []))

    def get_paper_venue(self, paper_id: str) -> str:
        return self.paper_venue.get(paper_id, "unknown")

    def get_paper_references(self, paper_id: str) -> List[str]:
        return list(self.paper_references.get(paper_id, []))

    def get_paper_citations_until(self, paper_id: str, year: int) -> int:
        years = self.paper_cited_by_years.get(paper_id, [])
        if not years:
            return 0
        return int(sum(1 for y in years if y <= year))

    # ---------------------------------------------------------------------
    # Author-level queries
    # ---------------------------------------------------------------------
    def get_author_papers_until(self, author_id: str, year: int) -> List[str]:
        papers = self.author_papers.get(author_id, [])
        return [pid for pid in papers if self.paper_year.get(pid, 9999) <= year]

    def get_author_recent_papers(self, author_id: str, year: int, window: int = 3) -> List[str]:
        start_year = year - window + 1
        papers = self.author_papers.get(author_id, [])
        return [
            pid for pid in papers
            if start_year <= self.paper_year.get(pid, -9999) <= year
        ]

    def get_author_first_year(self, author_id: str) -> Optional[int]:
        return self.author_first_year.get(author_id)

    def get_author_last_year(self, author_id: str) -> Optional[int]:
        return self.author_last_year.get(author_id)

    def get_author_pub_count_until(self, author_id: str, year: int) -> int:
        yearly = self.author_year_pub_count.get(author_id, {})
        return int(sum(v for y, v in yearly.items() if y <= year))

    def get_author_pub_count_window(self, author_id: str, end_year: int, window: int) -> int:
        start_year = end_year - window + 1
        yearly = self.author_year_pub_count.get(author_id, {})
        return int(sum(v for y, v in yearly.items() if start_year <= y <= end_year))

    def get_author_citation_count_until(self, author_id: str, year: int) -> int:
        yearly = self.author_year_citation_count.get(author_id, {})
        return int(sum(v for y, v in yearly.items() if y <= year))

    def get_author_citation_count_window(self, author_id: str, end_year: int, window: int) -> int:
        start_year = end_year - window + 1
        yearly = self.author_year_citation_count.get(author_id, {})
        return int(sum(v for y, v in yearly.items() if start_year <= y <= end_year))

    def get_author_active_years_until(self, author_id: str, year: int) -> int:
        yearly = self.author_year_pub_count.get(author_id, {})
        active_years = [y for y, v in yearly.items() if y <= year and v > 0]
        return len(active_years)

    def get_author_recent_activity_ratio(self, author_id: str, year: int, recent_window: int = 3) -> float:
        total = self.get_author_pub_count_until(author_id, year)
        if total <= 0:
            return 0.0
        recent = self.get_author_pub_count_window(author_id, year, recent_window)
        return float(recent / max(total, 1))

    def get_author_hindex_until(self, author_id: str, year: int) -> int:
        papers = self.get_author_papers_until(author_id, year)
        if not papers:
            return 0
        cts = [self.get_paper_citations_until(pid, year) for pid in papers]
        cts.sort(reverse=True)
        h = 0
        for i, c in enumerate(cts, start=1):
            if c >= i:
                h = i
            else:
                break
        return h

    def get_author_hindex_delta_window(self, author_id: str, end_year: int, window: int) -> int:
        prev_year = end_year - window
        h_now = self.get_author_hindex_until(author_id, end_year)
        h_prev = self.get_author_hindex_until(author_id, prev_year)
        return int(h_now - h_prev)

    # ---------------------------------------------------------------------
    # Collaboration / topic / venue queries
    # ---------------------------------------------------------------------
    def get_collab_count_until(self, author_id: str, coauthor_id: str, year: int) -> int:
        years = self.author_coauthor_years.get(author_id, {}).get(coauthor_id, [])
        return int(sum(1 for y in years if y <= year))

    def get_collab_count_window(self, author_id: str, coauthor_id: str, end_year: int, window: int) -> int:
        start_year = end_year - window + 1
        years = self.author_coauthor_years.get(author_id, {}).get(coauthor_id, [])
        return int(sum(1 for y in years if start_year <= y <= end_year))

    def get_years_since_last_collab(self, author_id: str, coauthor_id: str, year: int) -> int:
        years = [y for y in self.author_coauthor_years.get(author_id, {}).get(coauthor_id, []) if y <= year]
        if not years:
            return 999
        return int(year - max(years))

    def get_total_coauthors_until(self, author_id: str, year: int) -> int:
        cnt = 0
        for coaid in self.author_coauthor_years.get(author_id, {}):
            if self.get_collab_count_until(author_id, coaid, year) > 0:
                cnt += 1
        return cnt

    def get_recent_coauthors_count(self, author_id: str, year: int, window: int = 3) -> int:
        cnt = 0
        for coaid in self.author_coauthor_years.get(author_id, {}):
            if self.get_collab_count_window(author_id, coaid, year, window) > 0:
                cnt += 1
        return cnt

    def is_new_coauthor_for_author(self, author_id: str, coauthor_id: str, year: int, recent_window: int = 3) -> int:
        prev_cnt = self.get_collab_count_until(author_id, coauthor_id, year - recent_window)
        recent_cnt = self.get_collab_count_window(author_id, coauthor_id, year, recent_window)
        return int(prev_cnt == 0 and recent_cnt > 0)

    def get_topic_count_until(self, author_id: str, topic: str, year: int) -> int:
        years = self.author_topic_years.get(author_id, {}).get(topic, [])
        return int(sum(1 for y in years if y <= year))

    def get_topic_count_window(self, author_id: str, topic: str, end_year: int, window: int) -> int:
        start_year = end_year - window + 1
        years = self.author_topic_years.get(author_id, {}).get(topic, [])
        return int(sum(1 for y in years if start_year <= y <= end_year))

    def get_topic_span_years_until(self, author_id: str, topic: str, year: int) -> int:
        years = [y for y in self.author_topic_years.get(author_id, {}).get(topic, []) if y <= year]
        if not years:
            return 0
        return int(max(years) - min(years) + 1)

    def get_years_since_last_topic_use(self, author_id: str, topic: str, year: int) -> int:
        years = [y for y in self.author_topic_years.get(author_id, {}).get(topic, []) if y <= year]
        if not years:
            return 999
        return int(year - max(years))

    def is_new_topic_for_author(self, author_id: str, topic: str, year: int, recent_window: int = 3) -> int:
        prev_cnt = self.get_topic_count_until(author_id, topic, year - recent_window)
        recent_cnt = self.get_topic_count_window(author_id, topic, year, recent_window)
        return int(prev_cnt == 0 and recent_cnt > 0)

    def get_venue_count_until(self, author_id: str, venue: str, year: int) -> int:
        years = self.author_venue_years.get(author_id, {}).get(venue, [])
        return int(sum(1 for y in years if y <= year))

    def get_venue_count_window(self, author_id: str, venue: str, end_year: int, window: int) -> int:
        start_year = end_year - window + 1
        years = self.author_venue_years.get(author_id, {}).get(venue, [])
        return int(sum(1 for y in years if start_year <= y <= end_year))

    def get_venue_span_years_until(self, author_id: str, venue: str, year: int) -> int:
        years = [y for y in self.author_venue_years.get(author_id, {}).get(venue, []) if y <= year]
        if not years:
            return 0
        return int(max(years) - min(years) + 1)

    def get_years_since_last_venue_use(self, author_id: str, venue: str, year: int) -> int:
        years = [y for y in self.author_venue_years.get(author_id, {}).get(venue, []) if y <= year]
        if not years:
            return 999
        return int(year - max(years))

    def is_new_venue_for_author(self, author_id: str, venue: str, year: int, recent_window: int = 3) -> int:
        prev_cnt = self.get_venue_count_until(author_id, venue, year - recent_window)
        recent_cnt = self.get_venue_count_window(author_id, venue, year, recent_window)
        return int(prev_cnt == 0 and recent_cnt > 0)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _safe_year(x: Any) -> Optional[int]:
        try:
            y = int(x)
            if 1900 <= y <= 2100:
                return y
            return None
        except Exception:
            return None

    @staticmethod
    def _normalize_token(x: Any) -> str:
        s = str(x).strip().lower()
        if not s or s == "none":
            return ""
        return s

    # ---------------------------------------------------------------------
    # Quick summary
    # ---------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_author_history": len(self.author_ids_history),
            "n_records": len(self.records),
            "n_papers": len(self.paper_year),
            "n_authors_with_papers": len(self.author_papers),
            "n_topics": sum(len(v) for v in self.author_topic_counter.values()),
            "n_venues": sum(len(v) for v in self.author_venue_counter.values()),
        }


if __name__ == "__main__":
    for ds in ("aps",):
        rm = ResourceManagerV3(ds)
        print(json.dumps(rm.summary(), indent=2, ensure_ascii=False))