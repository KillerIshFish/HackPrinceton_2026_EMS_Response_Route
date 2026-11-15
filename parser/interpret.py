"""Read one transcript turn into Evidence, Facts and an incident-type suggestion.

Two readers (docs/BUILD_PLAN.md section 6, steps C1 and C2):

* Mode "rules" (``_rules_path``): deterministic dialog rules from ``parser/dialog.py``
  (yes / no answers to the question just asked, numbers and names printed on rooms,
  corrections) run before the vocabulary extractor (``parser.extract``), and the results
  are merged. No network.
* Modes "llm" and "agent": one and the same reader, the grounded Claude Haiku interpreter
  ``parser.agent.interpret``, which sees the building's room list and records what the
  caller's words point at. Both names are accepted so older clients keep working; the
  console shows it as "Claude Haiku". It falls back to the rules path with source
  ``"rules (agent unavailable: <reason>)"`` when the model cannot be reached. The older
  single-utterance vocabulary extractor (``parser.extract.extract_clues_llm``) is no
  longer a console mode; ``_rules_path(use_llm=True)`` still runs it for comparison.

Conventions every path keeps:

* Only caller turns produce evidence. Other speakers return an empty Interpretation.
* A specific statement of where the caller is (a room, unit or suite number, a room or
  tenant name, a floor) carries the stated share as its confidence: 0.40 to 0.95 depending
  on how sure and how exact the caller sounds (``core.fusion.stated_share``). In mode
  "rules" fixed word patterns rate it (``parser.statement_rating.rate_statement``); in the
  model modes Claude Haiku does. A floor that is about someone or something else ("he went
  up to the third floor", "the fire is on four") is not the caller's location and is not
  recorded as one.
* Evidence ids are ``evidence_id(index, k)`` = ``"t<index>-e<k>"`` (k counts from 0 within
  the turn), and fact ids ``fact_id(index, k)`` = ``"t<index>-f<k>"``, so ids stay stable
  when a turn is re-read and a dispatcher's reject survives a parser switch.
* ``Interpretation.source`` becomes ``TranscriptTurn.parser``: "rules", or the model mode
  that read the turn ("llm" / "agent"), or "rules (<mode> unavailable: <reason>)".
"""

from __future__ import annotations

from typing import Sequence

from core.schema import Building, Clue, Evidence, Fact, Interpretation, TranscriptTurn
from parser.dialog import (
    DialogStatement,
    rating_note,
    read_dialog,
    someone_elses_floor,
    someone_elses_place,
    superseded_ids,
)
from parser.extract import extract_clues_with_source
from parser.statement_rating import Rating, rate_statement

PARSER_MODES = ("rules", "llm", "agent")
# Modes read by the grounded Claude Haiku interpreter (parser.agent). "llm" is the name the
# console uses; "agent" is kept as an alias.
MODEL_MODES = ("llm", "agent")


def evidence_id(turn_index: int, k: int) -> str:
    return f"t{turn_index}-e{k}"


def fact_id(turn_index: int, k: int) -> str:
    return f"t{turn_index}-f{k}"


def wrap_clue(clue: Clue, turn_index: int, k: int, source: str) -> Evidence:
    """A vocabulary Clue as ``kind="vocab"`` Evidence (same confidence and span)."""
    return Evidence(
        id=evidence_id(turn_index, k),
        kind="vocab",
        clue=clue,
        confidence=clue.confidence,
        span=clue.span,
        turn=turn_index,
        source="llm" if source == "llm" else "rules",
    )


def interpret_turn(
    building: Building,
    transcript: Sequence[TranscriptTurn],
    index: int,
    mode: str = "rules",
    *,
    prior_evidence: Sequence[Evidence] = (),
    vocab: dict | None = None,
    prior_facts: Sequence[Fact] = (),
    incident_type: str | None = None,
) -> Interpretation:
    """Read ``transcript[index]`` in the context of the turns before it.

    * ``transcript``      every turn so far (index ``index`` is the one to read; earlier
                          turns resolve "yes", "it", corrections).
    * ``mode``            "rules", or "llm" / "agent" (the same grounded Claude Haiku reader).
    * ``prior_evidence``  evidence from earlier turns (ids a correction may supersede).
    * ``prior_facts``, ``incident_type``  what earlier turns already put on the console; the
                          model reader is shown them so it does not repeat them.
    """
    turn = transcript[index]
    if turn.speaker != "caller":
        return Interpretation(source="")

    if mode in MODEL_MODES:
        # AGENT HOOK (step C2). parser.agent falls back to the rules path itself when the model fails.
        from parser.agent import interpret as agent_interpret
        return agent_interpret(building, transcript, index, prior_evidence=prior_evidence, vocab=vocab,
                               prior_facts=prior_facts, incident_type=incident_type, source=mode)

    return _rules_path(building, transcript, index, use_llm=False, prior_evidence=prior_evidence, vocab=vocab)


def _rules_path(
    building: Building,
    transcript: Sequence[TranscriptTurn],
    index: int,
    *,
    use_llm: bool,
    prior_evidence: Sequence[Evidence] = (),
    vocab: dict | None = None,
) -> Interpretation:
    """Mode "rules" (and, with ``use_llm``, the older vocabulary extractor for comparison).

    1. Dialog rules (``parser.dialog.read_dialog``): the yes / no answer to the question
       just asked (or the Yes / No / Not sure buttons), numbers and names printed on rooms,
       and where a correction cue starts.
    2. The single-utterance vocabulary extractor (rules, or Claude Haiku when ``use_llm``).
    3. Merge: dialog statements first, then extractor clues; an extractor clue saying the
       same thing as a dialog clue (same type, value, polarity) is dropped, keeping the
       higher confidence. A room-type clue for a place that is not the caller's ("next to
       the boardroom", "the fire's in the kitchen") is dropped, and so is a floor that is
       about someone or something else ("he went up to the third floor"). A floor the
       caller states gets the stated share for how they said it. Ids are ``t<index>-e<k>``
       in that order.
    4. Corrections: earlier evidence of a slot restated after the cue (previous turns via
       ``prior_evidence``, and this turn before the cue) goes into
       ``Interpretation.supersedes``; this turn's own superseded items also carry
       ``status="superseded"``.
    """
    turn = transcript[index]
    text = turn.text or ""
    reading = read_dialog(building, transcript, index, vocab)
    clues, source = extract_clues_with_source(text, vocab=vocab, use_llm=use_llm)
    ev_source = "llm" if source == "llm" else "rules"

    # (statement-or-clue, text offset); dialog first.
    items: list[tuple[DialogStatement | Clue, int]] = [(s, s.start) for s in reading.statements]
    by_key: dict[tuple[str, str, bool], int] = {}
    for pos, (s, _) in enumerate(items):
        if isinstance(s, DialogStatement) and s.clue is not None:
            by_key.setdefault((s.clue.type, s.clue.value, s.clue.polarity), pos)
    cursor = 0
    floor_ratings: dict[tuple[str, int], Rating] = {}  # (floor value, offset) -> the rule rating behind its share
    for clue in clues:
        start = text.find(clue.span, cursor) if clue.span else -1
        if start < 0 and clue.span:
            start = text.find(clue.span)
        if start >= 0:
            cursor = start + len(clue.span)
        if clue.type == "room_type" and clue.polarity and start >= 0 and someone_elses_place(text, start):
            # "next to the boardroom", "just off the lobby", "the fire's in the kitchen": the
            # room named is not the one the caller is in, so its type says nothing about theirs.
            continue
        if clue.type == "floor" and clue.polarity and start >= 0 and source != "llm":
            if someone_elses_floor(text, start):
                continue  # "he went up to the third floor": not where the caller is
            # the caller's own statement of their floor: a specific statement (core.fusion)
            rating = rate_statement(text, start, start + len(clue.span))
            clue = clue.model_copy(update={"confidence": rating.share()})
            floor_ratings[(clue.value, start)] = rating
        key = (clue.type, clue.value, clue.polarity)
        if key in by_key:
            pos = by_key[key]
            existing = items[pos][0]
            if isinstance(existing, DialogStatement) and clue.confidence > existing.confidence and existing.source != "dispatcher":
                items[pos] = (DialogStatement(
                    confidence=clue.confidence, span=clue.span, clue=clue, source=ev_source,
                    start=start, end=start + len(clue.span) if start >= 0 else -1,
                ), start)
            continue
        by_key[key] = len(items)
        items.append((clue, start))

    evidence: list[Evidence] = []
    offsets: list[int] = []
    for k, (item, start) in enumerate(items):
        if isinstance(item, Clue):
            e = wrap_clue(item, index, k, source)
            rated = floor_ratings.get((item.value, start)) if item.type == "floor" else None
            if rated is not None:
                e = e.model_copy(update={"rationale": rating_note(rated, "rules")})
        else:
            rationale = item.rationale
            if item.rating is not None:
                rationale = "; ".join(x for x in (rationale, rating_note(item.rating, "rules")) if x)
            e = Evidence(
                id=evidence_id(index, k), kind=item.kind, clue=item.clue, selector=item.selector,
                default_match=0.0, confidence=item.confidence, span=item.span, turn=index,
                source=item.source, rationale=rationale,
            )
        evidence.append(e)
        offsets.append(start)

    supersedes = superseded_ids(list(zip(evidence, offsets)), prior_evidence, reading.correction_at)
    own = {e.id for e in evidence}
    if own & set(supersedes):
        evidence = [e.model_copy(update={"status": "superseded"}) if e.id in supersedes else e for e in evidence]
    return Interpretation(evidence=evidence, source=source, supersedes=supersedes)
