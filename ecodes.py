# E-code generator (corrected per Hyde–Kjos-Hanssen, Def. 7 & 9; Lemma 11 form)
# Alphabet: { '0', '[', ']', '+' }  with reduced labeling (all non-initial edges labeled '0').

from functools import lru_cache
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import sys

def gen_ecodes(max_length: int):
    """
    Enumerate all valid reduced E-codes of length <= max_length.

    Rules (Definition 9, reduced):
        1) E -> a E
        2) E -> a '[' E ']' '0' E
        3) E -> a '+' E
        4) E -> a '[+' E ']' '0' E
       where a = '' if i==0 else '0'.  (Only the very first edge may be ε; all later edges are '0'.)
       In (2) and (4), the symbol after ']' is a_j = '0' (since j>i).  [Lemma 11 proof form]

    Returns: list[str] sorted by (length, lexicographic).
    """

    @lru_cache(maxsize=None)
    def E(i: int, m: int, room: int):
        """
        Generate all reduced E(i, n) strings using exactly m remaining edges (from e_i onward),
        with output length <= room.  Base: if m==0, return {''}.
        """
        out = set()
        if room < 0:
            return tuple()
        if m == 0:
            out.add("")
            return tuple(sorted(out))

        # a_i options: only i==0 may use ε; otherwise '0'
        a_opts = [""] if i == 0 else []
        a_opts.append("0")

        # Rule 1 / 5: a E(i+1, ...)
        for a in a_opts:
            la = len(a)
            if la <= room:
                for t in E(i + 1, m - 1, room - la):
                    out.add(a + t)

        # Rule 3: a '+' E(i+1, ...)
        for a in a_opts:
            la = len(a)
            if la + 1 <= room:
                for t in E(i + 1, m - 1, room - la - 1):
                    out.add(a + "+" + t)

        # Bracket rules:
        # Split m edges as: 1 (ei) + k (inside) + 1 (ej) + rest (outside)
        # so k ∈ [0, m-2], outside = m-2-k ≥ 0.
        for a in a_opts:
            la = len(a)
            for k in range(0, m - 1):
                inside_m = k
                outside_m = m - 2 - k

                # Rule 2: a '[' E(i+1, i+k) ']' '0' E(j+1, n)
                # Overhead chars (no inside/outside): la + '[' + ']' + '0' = la + 3
                if la + 3 <= room:
                    for u in E(i + 1, inside_m, room - la - 3):
                        used = la + 3 + len(u)
                        if used <= room:
                            for v in E(i + (1 + inside_m) + 1, outside_m, room - used):
                                out.add(a + "[" + u + "]" + "0" + v)

                # Rule 4: a '[+' E(i+1, i+k) ']' '0' E(...)
                # Overhead chars: la + '[' + '+' + ']' + '0' = la + 4
                if la + 4 <= room:
                    for u in E(i + 1, inside_m, room - la - 4):
                        used = la + 4 + len(u)
                        if used <= room:
                            for v in E(i + (1 + inside_m) + 1, outside_m, room - used):
                                out.add(a + "[+" + u + "]" + "0" + v)

        return tuple(sorted(out))

    # Collect all codes for feasible edge counts m.
    # Since |E(0,m)| <= 2m (Lemma 11), codes of length <= L come from m in [1, L+1].
    seen = set()
    for m in range(1, max_length + 2):
        for code in E(0, m, max_length):
            if code and len(code) <= max_length:
                seen.add(code)
    return sorted(seen, key=lambda s: (len(s), s))


@dataclass
class NFA:
    start: int
    accept: Set[int]
    # transitions[state][symbol] -> set(next_states)
    transitions: Dict[int, Dict[str, Set[int]]]

    def add_edge(self, u: int, v: int, a: str) -> None:
        self.transitions[u].setdefault(a, set()).add(v)

def _bit(b: str) -> str:
    if b not in ("0", "1"):
        raise ValueError("Input word must be binary.")
    return b

def _bit_complement(b: str) -> str:
    return "1" if b == "0" else "0"

def decode_e_to_nfa(e: str, x: str) -> NFA:
    """
    Decode an E-code into an NFA tied to the specific input word x.

    Tokens:
      '0' : one edge along the abbreviated accepting path (consumes next bit of x)
      '[' : marks the *current* vertex as the target of a future backedge
      ']' : adds a backedge from the *current* vertex to the most recent '[' target
      '+' : indicates the current vertex has out-degree 2; the extra edge:
            - if inside a '[' ... ']' block, it skips that block and lands at the
              first vertex created right after the matching ']' (on the next '0');
            - if not inside any block, it lands at the very next vertex created (on
              the next '0').
    Labeling:
      - The edge on the accepting path is labeled by the next bit of x.
      - Every “extra” edge created by '+' or ']' is labeled with the bit-complement
        of the bit that would be consumed at that step, making it impossible to
        follow that edge while reading x, which enforces uniqueness for (E, x).

    Returns an NFA with no ε-transitions and a single accept state.
    Raises ValueError if the E-code is ill-formed w.r.t. brackets/placement or if
    the number of '0' tokens differs from len(x).
    """
    allowed = {"0", "[", "]", "+"}
    if any(ch not in allowed for ch in e):
        raise ValueError("E-code contains invalid characters.")

    # Quick length check: each '0' consumes one symbol of x.
    zeros = e.count("0")
    if zeros != len(x):
        raise ValueError(f"E-code has {zeros} '0' edges but |x|={len(x)}.")

    # Bracket stack: each item stores (target_vertex, plus_flag, plus_label_index)
    # target_vertex is the vertex marked by '['
    # plus_flag indicates whether we saw a '+' associated to this '['
    # plus_label_index is the input index at which that '+' occurs (to label the extra edge)
    bracket_stack: List[Tuple[int, bool, int]] = []

    # A '+' outside any bracket: connect FROM current vertex TO the *next* created vertex.
    # Store tuples (from_vertex, label_index)
    pending_plus_to_next: List[Tuple[int, int]] = []

    # A '+' inside a bracket: connect FROM the '['-vertex TO the first vertex created after
    # the matching ']' (i.e., the next '0' processed after we close that bracket).
    # We can’t connect immediately at '+', so when we see ']' we enqueue it here;
    # each item is (from_vertex, label_index)
    pending_plus_after_bracket: List[Tuple[int, int]] = []

    # NFA under construction
    nfa = NFA(start=0, accept=set(), transitions=defaultdict(dict))
    current = 0
    next_state = 1
    i = 0  # how many bits of x we have consumed

    # Single dead state used only if needed (not actually necessary under this decoding,
    # but left here for completeness)
    dead_state = None

    # Helper: when we create the next main-path state on a '0', connect any waiting
    # “to next” edges to that newly created state.
    def flush_pending_plus(to_state: int):
        nonlocal pending_plus_to_next, pending_plus_after_bracket
        for (frm, lab_idx) in pending_plus_to_next:
            nfa.add_edge(frm, to_state, _bit_complement(_bit(x[lab_idx])))
        pending_plus_to_next.clear()
        for (frm, lab_idx) in pending_plus_after_bracket:
            nfa.add_edge(frm, to_state, _bit_complement(_bit(x[lab_idx])))
        pending_plus_after_bracket.clear()

    # Track whether '+' occurs “inside a bracket” or not: bracket_stack nonempty => inside
    for ch in e:
        if ch == "[":
            # Mark the current vertex as the future backedge target.
            bracket_stack.append((current, False, i))  # (target, plus_flag, plus_label_idx_placeholder)
        elif ch == "+":
            if bracket_stack:
                # Mark '+' for the *innermost* open '['; record the input index for labeling.
                tgt, has_plus, _ = bracket_stack.pop()
                bracket_stack.append((tgt, True, i))
            else:
                # '+' outside brackets -> extra edge to the very next vertex created
                pending_plus_to_next.append((current, i))
        elif ch == "]":
            if not bracket_stack:
                raise ValueError("Unmatched ']' in E-code.")
            tgt, has_plus, plus_lab_idx = bracket_stack.pop()
            # Backedge from *current* vertex back to tgt, labeled with complement of next bit of x.
            # (If i == len(x) here, this edge is irrelevant for x and labeling is moot.)
            if i < len(x):
                nfa.add_edge(current, tgt, _bit_complement(_bit(x[i])))
            else:
                # Edge won’t be usable while reading x; give it any label (say '0')
                nfa.add_edge(current, tgt, "0")
            # If this '[' had a '+', queue an extra edge from tgt to the first vertex created
            # *after* this bracket closes (i.e., on the next '0' we process).
            if has_plus:
                pending_plus_after_bracket.append((tgt, plus_lab_idx))
        elif ch == "0":
            # Create the next main-path edge consuming x[i]
            if i >= len(x):
                # Should not happen due to earlier length check, but guard anyway
                raise ValueError("Too many '0' tokens for the length of x.")
            # Before creating the next vertex, wire any waiting “to next” edges to it.
            to_state = next_state
            flush_pending_plus(to_state)
            # Now add the main-path edge labeled with x[i]
            nfa.add_edge(current, to_state, _bit(x[i]))
            current = to_state
            next_state += 1
            i += 1
        else:
            # unreachable due to allowed set check
            raise AssertionError("Unexpected token.")

    if bracket_stack:
        raise ValueError("Unmatched '[' in E-code.")
    if pending_plus_to_next or pending_plus_after_bracket:
        raise ValueError("Dangling '+' with no following '0' to connect to.")

    # Finalize accept state
    nfa.accept = {current}
    return nfa

def count_accepting_paths(nfa: NFA, x: str, early_stop_at_two: bool = True) -> int:
    """
    Count the number of distinct accepting paths of length |x| that read x.
    Early-exits at 2 if early_stop_at_two is True.
    """
    from collections import Counter
    cur = Counter({nfa.start: 1})
    for c in x:
        nxt = Counter()
        for s, ways in cur.items():
            for t in nfa.transitions.get(s, {}).get(c, ()):
                nxt[t] += ways
                if early_stop_at_two and nxt[t] >= 2:
                    # We can already tell non-uniqueness
                    return 2
        cur = nxt
        if not cur:
            return 0
    total = sum(ways for s, ways in cur.items() if s in nfa.accept)
    return total

def uniquely_accepts_from_e(e: str, x: str) -> bool:
    """
    Top-level helper: build NFA from (E, x) and test unique acceptance of x.
    """
    nfa = decode_e_to_nfa(e, x)
    return count_accepting_paths(nfa, x) == 1

e = "0[0+]0"
x = "010"
print(uniquely_accepts_from_e(e, x))
sys.exit()

L = 10
codes = gen_ecodes(L)
counter = Counter(map(len,codes))
print(counter)
