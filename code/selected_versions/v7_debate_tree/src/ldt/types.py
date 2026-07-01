from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepProposal:
    """One agent's single-hop proposal for expanding a tree node."""

    proposal_id: int
    agent_idx: int
    text: str
    answer: str = ""
    is_final: bool = False
    confidence: float = 0.5
    raw: str = ""
    parse_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "agent_idx": self.agent_idx,
            "text": self.text,
            "answer": self.answer,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "raw": self.raw,
            "parse_ok": self.parse_ok,
        }


@dataclass
class CandidateNode:
    """A merged next-hop candidate before it is added to the reasoning tree."""

    candidate_id: str
    text: str
    depth: int
    parent_id: str
    support_agents: list[int] = field(default_factory=list)
    proposals: list[StepProposal] = field(default_factory=list)
    confidence: float = 0.5
    score: float = 0.5
    status: str = "uncertain"
    is_final: bool = False
    answer: str = ""
    merge_notes: str = ""
    history: dict[str, Any] = field(default_factory=dict)

    @property
    def support(self) -> int:
        return len(set(self.support_agents))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "support_agents": list(self.support_agents),
            "support": self.support,
            "confidence": self.confidence,
            "score": self.score,
            "status": self.status,
            "is_final": self.is_final,
            "answer": self.answer,
            "merge_notes": self.merge_notes,
            "proposals": [p.to_dict() for p in self.proposals],
            "history": self.history,
        }


@dataclass
class TreeNode:
    """A committed node in the shared layerwise debate tree."""

    node_id: str
    text: str
    depth: int
    parent_id: str | None = None
    support_agents: list[int] = field(default_factory=list)
    score: float = 1.0
    status: str = "accepted"
    is_final: bool = False
    answer: str = ""
    source_candidate_id: str = ""
    history: dict[str, Any] = field(default_factory=dict)

    @property
    def support(self) -> int:
        return len(set(self.support_agents))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "support_agents": list(self.support_agents),
            "support": self.support,
            "score": self.score,
            "status": self.status,
            "is_final": self.is_final,
            "answer": self.answer,
            "source_candidate_id": self.source_candidate_id,
            "history": self.history,
        }


class ReasoningTree:
    """Mutable tree plus path scoring helpers for LDT."""

    def __init__(self, question: str):
        self.root_id = "n0"
        self.nodes: dict[str, TreeNode] = {
            self.root_id: TreeNode(
                node_id=self.root_id,
                text=question,
                depth=0,
                parent_id=None,
                score=1.0,
                status="root",
            )
        }
        self.children: dict[str, list[str]] = {self.root_id: []}

    def add_child(self, parent_id: str, cand: CandidateNode) -> TreeNode:
        node_id = f"n{len(self.nodes)}"
        node = TreeNode(
            node_id=node_id,
            text=cand.text,
            depth=cand.depth,
            parent_id=parent_id,
            support_agents=list(cand.support_agents),
            score=cand.score,
            status=cand.status,
            is_final=cand.is_final,
            answer=cand.answer,
            source_candidate_id=cand.candidate_id,
            history=cand.history,
        )
        self.nodes[node_id] = node
        self.children.setdefault(parent_id, []).append(node_id)
        self.children.setdefault(node_id, [])
        return node

    def path_to(self, node_id: str, include_root: bool = False) -> list[TreeNode]:
        out = []
        cur = self.nodes[node_id]
        while cur is not None:
            if include_root or cur.node_id != self.root_id:
                out.append(cur)
            if cur.parent_id is None:
                break
            cur = self.nodes[cur.parent_id]
        out.reverse()
        return out

    def prefix_text(self, node_id: str) -> str:
        path = self.path_to(node_id, include_root=False)
        if not path:
            return "(no prior reasoning hops yet)"
        return "\n".join(f"{i}. {n.text}" for i, n in enumerate(path, start=1))

    def path_score(self, node_id: str) -> float:
        path = self.path_to(node_id, include_root=False)
        if not path:
            return 1.0
        return min(n.score for n in path)

    def leaf_ids(self) -> list[str]:
        return [nid for nid, kids in self.children.items() if nid != self.root_id and not kids]

    def final_ids(self) -> list[str]:
        return [nid for nid, node in self.nodes.items() if nid != self.root_id and node.is_final]

    def ranked_terminal_ids(self) -> list[str]:
        terminals = self.final_ids() or self.leaf_ids()
        return sorted(terminals, key=lambda nid: (self.path_score(nid), self.nodes[nid].score), reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "children": {pid: list(kids) for pid, kids in self.children.items()},
        }
