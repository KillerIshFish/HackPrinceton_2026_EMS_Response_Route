import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import {
  Ban,
  BriefcaseMedical,
  Check,
  ChevronRight,
  CircleQuestionMark,
  CornerDownLeft,
  DoorOpen,
  Flame,
  Headset,
  HeartPulse,
  ListTree,
  MessageSquare,
  Phone,
  ShieldAlert,
  Smartphone,
  Users,
  X,
} from "lucide-react";
import type {
  AgentTraceEntry,
  Answer,
  AnswerValue,
  Building,
  Evidence,
  Fact,
  FactCategory,
  ParserMode,
  PhoneFix,
  Question,
  Room,
  RoomSelector,
  TranscriptTurn,
} from "../types";
import { titleCase } from "../lib";
import { Panel } from "./Panel";
import { EvidenceChip, SourceMark, clueNegated, describeSelector, evidenceType, evidenceValue } from "./EvidenceChip";
import { QuestionList } from "./QuestionList";
import { MicButton } from "./MicButton";
import { INCIDENT_LABEL } from "./SituationBar";

type Speaker = "caller" | "dispatcher";

interface Props {
  transcript: TranscriptTurn[];
  /** every evidence item, whatever its status */
  evidence: Evidence[];
  facts: Fact[];
  /** top three suggested questions */
  questions: Question[];
  agentTrace: AgentTraceEntry[];
  building: Building;
  /** policy.show_incident_board: no caller-contact prompts */
  activeThreat: boolean;
  turn: number;
  totalTurns: number;
  /** which interpreter the next caller turn goes through */
  parserMode: ParserMode;
  /** the request in flight ("say", "ask", "step", "evidence:<id>", ...) or null */
  busy: string | null;
  onSay: (text: string, speaker: Speaker, answers?: Answer | null) => void;
  onAsk: (q: Question) => void;
  onRejectEvidence: (id: string) => void;
  onRestoreEvidence: (id: string) => void;
  /** rooms to outline on the map (null = none) */
  onHoverRooms: (roomIds: string[] | null) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parserLabel(parser: string | null | undefined): { text: string; warn: boolean } | null {
  if (!parser) return null;
  if (parser === "llm") return { text: "Read by Claude", warn: false };
  if (parser === "rules") return { text: "Read by keyword rules", warn: false };
  if (parser === "agent") return { text: "Read by Claude", warn: false };
  const cutShort = /^agent \((.*)\)$/.exec(parser);
  if (cutShort) return { text: `Read by Claude: ${cutShort[1]}`, warn: true };
  const fallback = /^rules \((.*)\)$/.exec(parser);
  if (fallback) return { text: `Read by keyword rules: ${fallback[1]}`, warn: true };
  return { text: `Read by: ${parser}`, warn: false };
}

const PARSER_NOTE: Record<ParserMode, { short: string; long: string }> = {
  rules: { short: "Keyword rules read it", long: "Caller turns are read by keyword rules." },
  llm: {
    short: "Claude reads it",
    long: "Caller turns are read by Claude, which reasons over this building's layout: hallways, door order, room sizes, corners, stairs and elevators (one request per turn, cached).",
  },
  agent: {
    short: "Claude reads it",
    long: "Caller turns are read by Claude, which reasons over this building's layout: hallways, door order, room sizes, corners, stairs and elevators (one request per turn, cached).",
  },
};

function clock(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function dateText(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

function fixSummary(fix: PhoneFix): string {
  const parts: string[] = [];
  if (fix.z_m !== null) parts.push(`z ${fix.z_m.toFixed(1)} m (σ ${fix.z_sigma_m.toFixed(2)} m)`);
  if (fix.x_m !== null && fix.y_m !== null)
    parts.push(
      `x ${fix.x_m.toFixed(1)}, y ${fix.y_m.toFixed(1)} m${fix.horizontal_sigma_m !== null ? ` (σ ${fix.horizontal_sigma_m.toFixed(0)} m)` : ""}`,
    );
  else parts.push("no horizontal fix");
  return parts.join(" - ");
}

interface Mark {
  span: string;
  color: string;
  title: string;
  struck: boolean;
}

/** Split `text` into plain and highlighted segments, one per span (first free occurrence, no overlaps). */
function highlight(text: string, items: Mark[]): ReactNode[] {
  const marks: { start: number; end: number; m: Mark }[] = [];
  // longer spans first so "his office, 405" wins over "office"
  for (const m of [...items].sort((a, b) => b.span.length - a.span.length)) {
    if (!m.span) continue;
    let from = 0;
    for (;;) {
      const start = text.indexOf(m.span, from);
      if (start < 0) break;
      const end = start + m.span.length;
      if (!marks.some((x) => start < x.end && end > x.start)) {
        marks.push({ start, end, m });
        break;
      }
      from = start + 1;
    }
  }
  marks.sort((a, b) => a.start - b.start);
  const out: ReactNode[] = [];
  let cursor = 0;
  marks.forEach(({ start, end, m }, i) => {
    if (start > cursor) out.push(text.slice(cursor, start));
    out.push(
      <mark
        key={`${start}-${i}`}
        title={m.title}
        className={`rounded-[3px] px-0.5 ${m.struck ? "text-ink-mute" : "text-ink"}`}
        style={{
          background: m.struck ? "transparent" : `${m.color}2e`,
          boxShadow: `inset 0 -2px 0 ${m.struck ? `${m.color}66` : m.color}`,
          textDecoration: m.struck ? "line-through" : undefined,
        }}
      >
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  });
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
}

const ANSWER_WORD: Record<AnswerValue, string> = { yes: "Yes", no: "No", unsure: "Not sure" };

const FACT_ORDER: FactCategory[] = ["threat", "hazard", "patient", "people", "access", "resource"];
const FACT_STYLE: Record<FactCategory, { label: string; icon: ReactNode; tone: string }> = {
  threat: { label: "Threat", icon: <ShieldAlert size={12} aria-hidden />, tone: "text-alert" },
  hazard: { label: "Hazard", icon: <Flame size={12} aria-hidden />, tone: "text-alert" },
  patient: { label: "Patient", icon: <HeartPulse size={12} aria-hidden />, tone: "text-ink-dim" },
  people: { label: "People", icon: <Users size={12} aria-hidden />, tone: "text-ink-dim" },
  access: { label: "Access", icon: <DoorOpen size={12} aria-hidden />, tone: "text-ink-dim" },
  resource: { label: "Resource", icon: <BriefcaseMedical size={12} aria-hidden />, tone: "text-ink-dim" },
};

/** One agent step in words. */
function describeStep(e: AgentTraceEntry, rooms: ReadonlyMap<string, Room>, evidence: Evidence[]): string {
  const a = e.args;
  const str = (k: string) => (typeof a[k] === "string" ? (a[k] as string) : null);
  const span = str("span");
  const quoted = span ? `: "${span}"` : "";
  switch (e.tool) {
    case "search_rooms": {
      const q = str("query");
      const fl = str("floor_label");
      return `Looked up rooms${q ? ` matching "${q}"` : ""}${fl ? ` on floor ${fl}` : ""}`;
    }
    case "record_location": {
      const sel = (a.selector ?? null) as Partial<RoomSelector> | null;
      return `Placed the caller at ${describeSelector(sel, rooms)}${quoted}`;
    }
    case "record_fact":
      return `Noted a ${str("category") ?? "response"} fact${quoted}`;
    case "supersede": {
      const target = evidence.find((x) => x.id === str("evidence_id"));
      return `Marked ${target ? `"${target.span}"` : (str("evidence_id") ?? "an earlier statement")} as corrected${quoted}`;
    }
    case "suggest_incident_type": {
      const t = str("type");
      return `Suggested the incident type ${t && t in INCIDENT_LABEL ? INCIDENT_LABEL[t as keyof typeof INCIDENT_LABEL] : (t ?? "")}${quoted}`;
    }
    default: {
      const rest = Object.entries(a)
        .filter(([k]) => k !== "span")
        .map(([k, v]) => `${k.replace(/_/g, " ")} ${typeof v === "string" ? v : JSON.stringify(v)}`)
        .join(", ");
      return `${titleCase(e.tool)}${rest ? ` (${rest})` : ""}${quoted}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export function TranscriptPanel({
  transcript,
  evidence,
  facts,
  questions,
  agentTrace,
  building,
  activeThreat,
  turn,
  totalTurns,
  parserMode,
  busy,
  onSay,
  onAsk,
  onRejectEvidence,
  onRestoreEvidence,
  onHoverRooms,
}: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState("");
  const [interim, setInterim] = useState("");
  const [speaker, setSpeaker] = useState<Speaker>("caller");
  /** what was just sent, shown as the last turn with "Reading..." until the server answers */
  const [pending, setPending] = useState<{ text: string; speaker: Speaker } | null>(null);
  const [tab, setTab] = useState<"questions" | "evidence">(() => {
    try {
      return window.localStorage.getItem("console.transcriptTab") === "evidence" ? "evidence" : "questions";
    } catch {
      return "questions";
    }
  });
  const rememberTab = (v: "questions" | "evidence") => {
    setTab(v);
    try {
      window.localStorage.setItem("console.transcriptTab", v);
    } catch {
      /* storage unavailable */
    }
  };

  useEffect(() => {
    if (busy === null) setPending(null);
  }, [busy]);

  const showPending = pending !== null && (busy === "say" || busy === "ask");
  const reading = busy === "step";
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [transcript.length, showPending, reading]);

  const rooms = useMemo(() => new Map(building.rooms.map((r) => [r.id, r])), [building]);
  const evidenceById = useMemo(() => new Map(evidence.map((e) => [e.id, e])), [evidence]);
  const factsById = useMemo(() => new Map(facts.map((f) => [f.id, f])), [facts]);

  /** evidence, facts and agent steps per transcript turn */
  const perTurn = useMemo(() => {
    const m = new Map<number, { ev: Evidence[]; facts: Fact[]; trace: AgentTraceEntry[] }>();
    const get = (t: number) => {
      let v = m.get(t);
      if (!v) m.set(t, (v = { ev: [], facts: [], trace: [] }));
      return v;
    };
    for (const t of transcript) {
      const v = get(t.turn);
      for (const id of t.evidence_ids ?? []) {
        const e = evidenceById.get(id);
        if (e && !v.ev.includes(e)) v.ev.push(e);
      }
      for (const id of t.fact_ids ?? []) {
        const f = factsById.get(id);
        if (f && !v.facts.includes(f)) v.facts.push(f);
      }
    }
    for (const e of evidence) if (!get(e.turn).ev.includes(e)) get(e.turn).ev.push(e);
    for (const f of facts) if (f.turn !== null && !get(f.turn).facts.includes(f)) get(f.turn).facts.push(f);
    for (const a of agentTrace) get(a.turn).trace.push(a);
    return m;
  }, [transcript, evidence, facts, agentTrace, evidenceById, factsById]);

  function send(text: string, who: Speaker, answers: Answer | null = null) {
    setPending({ text, speaker: who });
    onSay(text, who, answers);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy !== null) return;
    send(text, speaker);
    setDraft("");
  }

  function ask(q: Question) {
    setPending({ text: q.question, speaker: "dispatcher" });
    onAsk(q);
    if (q.kind === "open") {
      setSpeaker("caller");
      inputRef.current?.focus();
    }
  }

  const chipProps = {
    rooms,
    busy,
    onReject: onRejectEvidence,
    onRestore: onRestoreEvidence,
    onHoverRooms,
  };

  const activeCount = evidence.filter((e) => e.status === "active").length;
  const asideCount = evidence.length - activeCount;
  const factGroups = FACT_ORDER.map((c) => ({ c, items: facts.filter((f) => f.category === c) })).filter((g) => g.items.length > 0);

  return (
    <Panel
      title="Call transcript"
      icon={<MessageSquare size={15} aria-hidden />}
      actions={
        <span className="num text-xs text-ink-mute">
          turn {Math.min(turn + 1, totalTurns)}/{totalTurns}
        </span>
      }
      className="h-full"
      bodyClassName="flex flex-col"
    >
      <ol className="scroll-thin min-h-40 flex-1 space-y-2.5 overflow-y-auto p-3" aria-live="polite" aria-label="Transcript">
        {transcript.map((t, index) => {
          const own = perTurn.get(t.turn) ?? { ev: [], facts: [], trace: [] };
          const answeredBy = t.question
            ? transcript
                .slice(index + 1)
                .find((u) => u.answer && u.answer.clue_type === t.question?.clue_type && u.answer.value === t.question?.value)
            : undefined;
          const askedTurn = t.answer
            ? [...transcript.slice(0, index)].reverse().find((u) => u.question && u.question.clue_type === t.answer?.clue_type && u.question.value === t.answer?.value)
            : undefined;
          return (
            <TurnView
              key={`${t.turn}-${index}`}
              t={t}
              own={own}
              answeredBy={answeredBy}
              askedTurn={askedTurn}
              chipProps={chipProps}
              evidence={evidence}
              busy={busy}
              onAnswer={(q, value) =>
                q.clue_type && q.value
                  ? send(ANSWER_WORD[value], "caller", { clue_type: q.clue_type, value: q.value, answer: value })
                  : undefined
              }
              onHoverRooms={onHoverRooms}
            />
          );
        })}
        {showPending && pending ? (
          <li className="text-[13px] opacity-80" aria-busy="true">
            <TurnHeader label={pending.speaker === "caller" ? "Caller" : "Dispatcher"} sub="sending" />
            <div
              className={
                pending.speaker === "caller"
                  ? "rounded-md rounded-tl-none border border-line-strong bg-raised px-3 py-2 text-[14px] text-ink"
                  : "border-l-2 border-line-strong pl-2.5 text-[13px] text-ink-dim"
              }
            >
              {pending.speaker === "caller" ? `“${pending.text}”` : pending.text}
              <Reading />
            </div>
          </li>
        ) : null}
        {reading ? (
          <li className="text-[13px]" aria-busy="true">
            <Reading text="Reading the next turn…" />
          </li>
        ) : null}
        <div ref={endRef} />
      </ol>

      <form onSubmit={submit} className="shrink-0 border-t border-line px-3 py-2">
        <div className="mb-1.5 flex items-center gap-2">
          <ToggleGroup.Root
            type="single"
            aria-label="Who said it"
            value={speaker}
            onValueChange={(v) => {
              if (v === "caller" || v === "dispatcher") setSpeaker(v);
            }}
            className="flex h-7 overflow-hidden rounded border border-line-strong"
          >
            {(
              [
                ["caller", "Caller", <Phone key="i" size={12} aria-hidden />],
                ["dispatcher", "Dispatcher", <Headset key="i" size={12} aria-hidden />],
              ] as const
            ).map(([v, label, icon]) => (
              <ToggleGroup.Item
                key={v}
                value={v}
                className="flex items-center gap-1 border-l border-line-strong px-2 text-[12px] first:border-l-0 data-[state=off]:text-ink-dim data-[state=off]:hover:bg-raised data-[state=off]:hover:text-ink data-[state=on]:bg-ink data-[state=on]:font-semibold data-[state=on]:text-bg"
              >
                {icon}
                {label}
              </ToggleGroup.Item>
            ))}
          </ToggleGroup.Root>
          <label htmlFor="turn-text" className="sr-only">
            {speaker === "caller" ? "Caller said" : "Dispatcher said"}
          </label>
          <span className="ml-auto truncate text-[10.5px] text-ink-mute" title={PARSER_NOTE[parserMode].long}>
            {PARSER_NOTE[parserMode].short}
          </span>
        </div>
        <div className="flex flex-wrap gap-x-2 gap-y-1">
          <div className="relative min-w-0 flex-1">
            <input
              id="turn-text"
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={600}
              autoComplete="off"
              placeholder={interim ? "" : speaker === "caller" ? "Type what the caller just said" : "Type what you said to the caller"}
              className="w-full rounded border border-line-strong bg-bg px-2.5 py-1.5 text-[13px] text-ink placeholder:text-ink-mute focus:border-accent focus:outline-none"
            />
            {/* dictation: words not yet final, greyed after whatever was typed (MicButton) */}
            {interim ? (
              <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center overflow-hidden whitespace-pre border border-transparent px-2.5 text-[13px]">
                <span className="invisible">{draft}</span>
                <span className="italic text-ink-mute">{draft && !/\s$/.test(draft) ? " " : ""}{interim}</span>
              </div>
            ) : null}
          </div>
          <MicButton speaker={speaker} busy={busy !== null} onFinal={(text, who) => send(text, who)} onInterim={setInterim} />
          <button
            type="submit"
            disabled={busy !== null || draft.trim().length === 0}
            className="flex shrink-0 items-center gap-1 rounded border border-line-strong px-2.5 py-1.5 text-xs font-medium text-ink hover:bg-raised disabled:cursor-not-allowed disabled:opacity-50"
          >
            Add
            <CornerDownLeft size={13} aria-hidden />
          </button>
        </div>
      </form>

      <Tabs.Root
        value={tab}
        onValueChange={(v) => {
          if (v === "questions" || v === "evidence") rememberTab(v);
        }}
        className="flex max-h-[36vh] min-h-0 shrink-0 flex-col border-t border-line xl:max-h-[38%]"
      >
        <Tabs.List aria-label="Questions and evidence" className="flex shrink-0 border-b border-line px-1.5">
          {(
            [
              [
                "questions",
                <CircleQuestionMark key="i" size={13} aria-hidden />,
                [["Questions", questions.length]],
                `Questions: ${questions.length}`,
              ],
              [
                "evidence",
                <ListTree key="i" size={13} aria-hidden />,
                [
                  ["Evidence", evidence.length],
                  ["Facts", facts.length],
                ],
                `Evidence: ${activeCount} in use, ${asideCount} set aside. Facts: ${facts.length}`,
              ],
            ] as const
          ).map(([v, icon, parts, aria]) => (
            <Tabs.Trigger
              key={v}
              value={v}
              aria-label={aria}
              title={aria}
              className="-mb-px flex items-center gap-1.5 border-b-2 border-transparent px-2 py-1.5 text-[12px] text-ink-dim hover:text-ink data-[state=active]:border-ink data-[state=active]:font-semibold data-[state=active]:text-ink"
            >
              <span className="opacity-80">{icon}</span>
              {parts.map(([label, n], i) => (
                <span key={label} className="flex items-center gap-1">
                  {i > 0 ? <span className="text-ink-mute" aria-hidden>·</span> : null}
                  {label}
                  <span className="num rounded-sm bg-raised px-1 text-[10.5px] font-normal text-ink-mute">{n}</span>
                </span>
              ))}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        <Tabs.Content value="questions" className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-2 data-[state=inactive]:hidden">
          <QuestionList questions={questions} activeThreat={activeThreat} busy={busy} onAsk={ask} />
        </Tabs.Content>

        <Tabs.Content value="evidence" className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-2 data-[state=inactive]:hidden">
          <p className="mb-1.5 text-[10.5px] text-ink-mute">
            {activeCount} in use{asideCount > 0 ? `, ${asideCount} set aside` : ""}. Point at one to outline its rooms on the map.
          </p>
          {evidence.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {evidence.map((e) => (
                <EvidenceChip key={e.id} evidence={e} variant="pill" {...chipProps} />
              ))}
            </div>
          ) : (
            <p className="text-[12px] text-ink-mute">Nothing from the call yet: the estimate uses the building prior and the phone fix only.</p>
          )}
          <h4 className="eyebrow mb-1 mt-3 !text-[10px]">Facts ({facts.length})</h4>
          {factGroups.length > 0 ? (
            <div className="space-y-2">
              {factGroups.map(({ c, items }) => (
                <div key={c}>
                  <h5 className={`mb-1 flex items-center gap-1 text-[10.5px] font-semibold uppercase tracking-wider ${FACT_STYLE[c].tone}`}>
                    {FACT_STYLE[c].icon}
                    {FACT_STYLE[c].label}
                    <span className="num font-normal text-ink-mute">({items.length})</span>
                  </h5>
                  <ul className="space-y-1">
                    {items.map((f) => (
                      <FactRow key={f.id} f={f} rooms={rooms} onHoverRooms={onHoverRooms} />
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[12px] text-ink-mute">No facts recorded from the call yet.</p>
          )}
        </Tabs.Content>
      </Tabs.Root>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// One turn
// ---------------------------------------------------------------------------

type ChipProps = Omit<Parameters<typeof EvidenceChip>[0], "evidence" | "variant">;

function TurnHeader({ label, sub, at, n }: { label: string; sub?: string | null; at?: string | null; n?: number }) {
  return (
    <div className="mb-1 flex items-center gap-2">
      {n !== undefined ? <span className="num text-[10px] text-ink-mute">T{n}</span> : null}
      <span className="eyebrow !text-[10px]">
        {label}
        {sub ? <span className="font-normal normal-case tracking-normal"> - {sub}</span> : null}
      </span>
      {at ? <span className="num ml-auto text-[10px] text-ink-mute">{at}</span> : null}
    </div>
  );
}

/** "Caller 1", "Caller: room type" -> label "Caller", sub "room type"; system turns keep their title as sub. */
function turnLabel(t: TranscriptTurn, isSensor: boolean): { label: string; sub: string | null } {
  const base = isSensor ? "Sensor" : titleCase(t.speaker);
  const title = (t.title ?? "").trim();
  if (!title) return { label: base, sub: null };
  if (title.toLowerCase().startsWith(t.speaker)) {
    const [head = base, ...rest] = title.split(":");
    return { label: head.trim() || base, sub: rest.join(":").trim() || null };
  }
  return { label: base, sub: title };
}

function Reading({ text = "Reading…" }: { text?: string }) {
  return (
    <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-ink-mute" role="status">
      <span className="live-dot h-1.5 w-1.5 rounded-full bg-ink-mute" aria-hidden />
      {text}
    </div>
  );
}

function TurnView({
  t,
  own,
  answeredBy,
  askedTurn,
  chipProps,
  evidence,
  busy,
  onAnswer,
  onHoverRooms,
}: {
  t: TranscriptTurn;
  own: { ev: Evidence[]; facts: Fact[]; trace: AgentTraceEntry[] };
  answeredBy: TranscriptTurn | undefined;
  askedTurn: TranscriptTurn | undefined;
  chipProps: ChipProps;
  evidence: Evidence[];
  busy: string | null;
  onAnswer: (q: Question, value: AnswerValue) => void;
  onHoverRooms: (ids: string[] | null) => void;
}) {
  const isSensor = t.phone_fix !== null;
  const isCaller = t.speaker === "caller";
  const at = clock(t.at);
  const { label, sub } = turnLabel(t, isSensor);

  const marks: Mark[] = own.ev
    .filter((e) => e.span)
    .map((e) => {
      const type = evidenceType(e);
      return {
        span: e.span,
        color: type.color,
        title: `${type.label}: ${evidenceValue(e, chipProps.rooms)}${e.status === "active" ? "" : ` (${e.status})`}`,
        struck: e.status !== "active" || (e.clue !== null && clueNegated(e.clue)) || e.negate,
      };
    });
  // turns read before Step 0 carry clues but no evidence ids
  if (own.ev.length === 0)
    for (const c of t.clues) marks.push({ span: c.span, color: "#7fb4ff", title: `${c.type}: ${c.value}`, struck: clueNegated(c) });

  if (isSensor && t.phone_fix) {
    return (
      <li className="text-[13px]">
        <TurnHeader n={t.turn} label={label} sub={sub} at={at} />
        <div className="rounded border border-line bg-raised px-2.5 py-2">
          <div className="flex items-center gap-2 text-ink">
            <Smartphone size={14} className="shrink-0 text-ink-dim" aria-hidden />
            <span className="num text-xs">{fixSummary(t.phone_fix)}</span>
          </div>
          <p className="mt-1 text-xs text-ink-dim">{t.text}</p>
        </div>
      </li>
    );
  }

  const chips =
    own.ev.length > 0 ? (
      <div className="mt-2 space-y-1">
        {own.ev.map((e) => (
          <EvidenceChip key={e.id} evidence={e} variant="row" {...chipProps} />
        ))}
      </div>
    ) : null;

  const factLines =
    own.facts.length > 0 ? (
      <ul className="mt-1.5 space-y-0.5">
        {own.facts.map((f) => (
          <li
            key={f.id}
            className="flex items-start gap-1 text-[11.5px] text-ink-dim"
            onMouseEnter={f.room_id ? () => onHoverRooms([f.room_id!]) : undefined}
            onMouseLeave={f.room_id ? () => onHoverRooms(null) : undefined}
          >
            <span className={`mt-[2px] shrink-0 ${FACT_STYLE[f.category].tone}`}>{FACT_STYLE[f.category].icon}</span>
            <span>
              <span className="font-semibold text-ink-mute">{FACT_STYLE[f.category].label}:</span> {f.text}
            </span>
          </li>
        ))}
      </ul>
    ) : null;

  const trace = own.trace.length > 0 ? <TraceView entries={own.trace} rooms={chipProps.rooms} evidence={evidence} /> : null;
  const parser = parserLabel(t.parser);

  if (isCaller) {
    return (
      <li className="text-[13px]">
        <TurnHeader n={t.turn} label={label} sub={sub} at={at} />
        <div className="rounded-md rounded-tl-none border border-line-strong bg-raised px-3 py-2 text-[14px] leading-relaxed text-ink">
          {t.answer ? (
            <div className="mb-1 flex items-center gap-1.5 text-[11px] text-ink-mute">
              <CornerDownLeft size={11} className="shrink-0 -scale-x-100" aria-hidden />
              Answer{askedTurn ? ` to T${askedTurn.turn}` : ""}:{" "}
              <span className="font-semibold text-ink">{ANSWER_WORD[t.answer.answer]}</span>
            </div>
          ) : null}
          “{highlight(t.text, marks)}”
          {chips}
          {factLines}
          {own.ev.length === 0 && own.facts.length === 0 ? (
            <div className="mt-1.5 text-[11px] text-ink-mute">Nothing about the location or the response was read from this turn.</div>
          ) : null}
          {parser ? <div className={`mt-1.5 text-[10.5px] ${parser.warn ? "text-warn" : "text-ink-mute"}`}>{parser.text}</div> : null}
          {trace}
        </div>
      </li>
    );
  }

  // dispatcher and system turns
  const q = t.question;
  return (
    <li className="text-[13px]">
      <TurnHeader n={t.turn} label={label} sub={q ? "asked a suggested question" : sub} at={at} />
      <div className={`border-l-2 pl-2.5 ${q ? "border-ink-mute" : "border-line"}`}>
        <p className={q ? "flex items-start gap-1.5 text-[13px] text-ink" : "text-xs text-ink-dim"}>
          {q ? <CircleQuestionMark size={13} className="mt-[3px] shrink-0 text-ink-dim" aria-hidden /> : null}
          <span>{marks.length > 0 ? highlight(t.text, marks) : t.text}</span>
        </p>
        {q && q.kind === "yes_no" && q.clue_type && q.value ? (
          answeredBy?.answer ? (
            <p className="mt-1 text-[11px] text-ink-mute">
              Caller answered <span className="font-semibold text-ink-dim">{ANSWER_WORD[answeredBy.answer.answer]}</span> (T{answeredBy.turn})
            </p>
          ) : (
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5" role="group" aria-label="Caller's answer">
              <span className="text-[11px] text-ink-mute">Caller's answer:</span>
              {(
                [
                  ["yes", <Check key="i" size={12} aria-hidden />],
                  ["no", <X key="i" size={12} aria-hidden />],
                  ["unsure", <CircleQuestionMark key="i" size={12} aria-hidden />],
                ] as const
              ).map(([v, icon]) => (
                <button
                  key={v}
                  type="button"
                  disabled={busy !== null}
                  onClick={() => onAnswer(q, v)}
                  className="flex h-7 items-center gap-1 rounded border border-line-strong bg-raised px-2.5 text-[12px] font-medium text-ink hover:border-ink-mute hover:bg-bg disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {icon}
                  {ANSWER_WORD[v]}
                </button>
              ))}
            </div>
          )
        ) : q && q.kind === "open" && !answeredBy ? (
          <p className="mt-1 text-[11px] text-ink-mute">Open question: type the caller's answer below.</p>
        ) : null}
        {chips}
        {factLines}
        {trace}
      </div>
    </li>
  );
}

function TraceView({ entries, rooms, evidence }: { entries: AgentTraceEntry[]; rooms: ReadonlyMap<string, Room>; evidence: Evidence[] }) {
  const dropped = entries.filter((e) => !e.kept).length;
  return (
    <details className="group mt-1.5 text-[11.5px]">
      <summary className="flex cursor-pointer list-none items-center gap-1 text-ink-mute hover:text-ink [&::-webkit-details-marker]:hidden">
        <ChevronRight size={12} className="shrink-0 transition-transform group-open:rotate-90" aria-hidden />
        How this was read
        <span className="num">
          ({entries.length} step{entries.length === 1 ? "" : "s"}
          {dropped > 0 ? `, ${dropped} dropped` : ""})
        </span>
      </summary>
      <ol className="mt-1 space-y-1 border-l border-line pl-2">
        {entries.map((e, i) => (
          <li key={i} className="flex items-start gap-1.5 leading-snug" title={`${e.tool} - result size ${e.result_size}`}>
            {e.kept ? (
              <span className="mt-[1px] flex shrink-0 items-center gap-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-mute">
                <Check size={11} aria-hidden />
                Kept
              </span>
            ) : (
              <span className="mt-[1px] flex shrink-0 items-center gap-0.5 text-[10px] font-semibold uppercase tracking-wide text-warn">
                <Ban size={11} aria-hidden />
                Dropped
              </span>
            )}
            <span className="min-w-0">
              <span className={e.kept ? "text-ink-dim" : "text-ink-dim line-through decoration-ink-mute"}>{describeStep(e, rooms, evidence)}</span>
              {!e.kept ? (
                <span className="block text-[11px] text-warn">Why: {e.reason ?? "no reason recorded"}</span>
              ) : null}
            </span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function FactRow({ f, rooms, onHoverRooms }: { f: Fact; rooms: ReadonlyMap<string, Room>; onHoverRooms: (ids: string[] | null) => void }) {
  const room = f.room_id ? rooms.get(f.room_id) : undefined;
  const when = f.source === "record" ? dateText(f.observed_at) : clock(f.observed_at);
  return (
    <li
      className="rounded border border-line bg-raised/50 px-2 py-1 text-[12px] leading-snug"
      onMouseEnter={f.room_id ? () => onHoverRooms([f.room_id!]) : undefined}
      onMouseLeave={f.room_id ? () => onHoverRooms(null) : undefined}
    >
      <div className="text-ink">{f.text}</div>
      {f.span ? <q className="mt-0.5 block text-[11.5px] italic text-ink-dim">{f.span}</q> : null}
      <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[10.5px] text-ink-mute">
        {f.turn !== null ? <span className="num">T{f.turn}</span> : null}
        {when ? <span className="num">{f.source === "record" ? `record dated ${when}` : when}</span> : null}
        {room ? <span>{room.name} (floor {room.floor_label})</span> : null}
        <span className={`font-semibold uppercase tracking-wide ${f.verified ? "text-ink-dim" : "text-ink-mute"}`}>
          {f.verified ? "verified" : "unverified"}
        </span>
        <span className="ml-auto">
          <SourceMark source={f.source} />
        </span>
      </div>
    </li>
  );
}
