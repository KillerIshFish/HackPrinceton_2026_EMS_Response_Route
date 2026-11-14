import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { CircleQuestionMark, CornerDownLeft, MessageSquare, Smartphone } from "lucide-react";
import type { Clue, NextQuestion, ParserMode, PhoneFix, TranscriptTurn } from "../types";
import { CLUE_STYLE, titleCase } from "../lib";
import { Panel } from "./Panel";

interface Props {
  transcript: TranscriptTurn[];
  clues: Clue[];
  nextQuestion: NextQuestion | null;
  turn: number;
  totalTurns: number;
  /** "rules" or "llm": which parser the next utterance will go through */
  parserMode: ParserMode;
  /** true while a request is in flight */
  busy: boolean;
  /** add something the caller just said */
  onSay: (text: string) => void;
}

function parserLabel(parser: string | null | undefined): { text: string; warn: boolean } | null {
  if (!parser) return null;
  if (parser === "llm") return { text: "Parsed by Claude Haiku", warn: false };
  if (parser === "rules") return { text: "Parsed by keyword rules", warn: false };
  return { text: `Keyword rules - ${parser.replace(/^rules \(|\)$/g, "")}`, warn: true };
}

/** "landmarks:window" (the value of a negative clue) -> "window" */
function clueValue(c: Clue): string {
  return titleCase(c.value.includes(":") ? c.value.slice(c.value.indexOf(":") + 1) : c.value);
}

function clueTitle(c: Clue): string {
  const negated = c.type === "negative" || !c.polarity;
  return `${CLUE_STYLE[c.type].label}: ${negated ? "NOT " : ""}${clueValue(c)} - confidence ${Math.round(c.confidence * 100)}% - polarity ${negated ? "negative" : "positive"}`;
}

/** Split `text` into plain and highlighted segments, one per clue span (first match, no overlaps). */
function highlight(text: string, clues: Clue[]): ReactNode[] {
  const marks: { start: number; end: number; clue: Clue }[] = [];
  for (const clue of clues) {
    if (!clue.span) continue;
    let from = 0;
    // case-sensitive substring; skip occurrences already claimed by another clue
    for (;;) {
      const start = text.indexOf(clue.span, from);
      if (start < 0) break;
      const end = start + clue.span.length;
      if (!marks.some((m) => start < m.end && end > m.start)) {
        marks.push({ start, end, clue });
        break;
      }
      from = start + 1;
    }
  }
  marks.sort((a, b) => a.start - b.start);

  const out: ReactNode[] = [];
  let cursor = 0;
  marks.forEach((m, i) => {
    if (m.start > cursor) out.push(text.slice(cursor, m.start));
    const color = CLUE_STYLE[m.clue.type].color;
    const negated = m.clue.type === "negative" || !m.clue.polarity;
    out.push(
      <mark
        key={`${m.start}-${i}`}
        title={clueTitle(m.clue)}
        className="rounded-[3px] px-0.5 text-ink"
        style={{
          background: `${color}2e`,
          boxShadow: `inset 0 -2px 0 ${color}`,
          textDecoration: negated ? "line-through" : undefined,
          textDecorationColor: color,
        }}
      >
        {text.slice(m.start, m.end)}
      </mark>,
    );
    cursor = m.end;
  });
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
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

export function ClueChip({ clue }: { clue: Clue }) {
  const style = CLUE_STYLE[clue.type];
  const negated = clue.type === "negative" || !clue.polarity;
  return (
    <span
      title={`${clueTitle(clue)} - from "${clue.span}"`}
      className="inline-flex items-center gap-1.5 rounded border border-line bg-raised px-1.5 py-0.5 text-[11px]"
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: style.color }} aria-hidden />
      <span className="text-ink-mute">{style.label}</span>
      <span className="font-medium text-ink">
        {negated ? "not " : ""}
        {clueValue(clue).toLowerCase()}
      </span>
      <span className="num text-ink-mute">{Math.round(clue.confidence * 100)}%</span>
    </span>
  );
}

export function TranscriptPanel({ transcript, clues, nextQuestion, turn, totalTurns, parserMode, busy, onSay }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState("");
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [transcript.length]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    onSay(text);
    setDraft("");
  }

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
      <ol className="scroll-thin min-h-40 flex-1 space-y-2.5 overflow-y-auto p-3" aria-live="polite">
        {transcript.map((t) => {
          const isCaller = t.speaker === "caller";
          const isSensor = t.phone_fix !== null;
          return (
            <li key={t.turn} className="text-[13px]">
              <div className="mb-1 flex items-center gap-2">
                <span className="num text-[10px] text-ink-mute">T{t.turn}</span>
                <span className="eyebrow !text-[10px]">
                  {isSensor ? "Sensor" : titleCase(t.speaker)}
                  {t.title ? <span className="font-normal normal-case tracking-normal"> - {t.title}</span> : null}
                </span>
              </div>
              {isSensor && t.phone_fix ? (
                <div className="rounded border border-line bg-raised px-2.5 py-2">
                  <div className="flex items-center gap-2 text-ink">
                    <Smartphone size={14} className="shrink-0 text-accent" aria-hidden />
                    <span className="num text-xs">{fixSummary(t.phone_fix)}</span>
                  </div>
                  <p className="mt-1 text-xs text-ink-dim">{t.text}</p>
                </div>
              ) : isCaller ? (
                <div className="rounded-md rounded-tl-none border border-line-strong bg-raised px-3 py-2 text-[14px] leading-relaxed text-ink">
                  “{highlight(t.text, t.clues)}”
                  {t.clues.length > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {t.clues.map((c, i) => (
                        <ClueChip key={`${c.type}-${c.value}-${i}`} clue={c} />
                      ))}
                    </div>
                  ) : (
                    <div className="mt-1.5 text-[11px] text-ink-mute">No location clues extracted</div>
                  )}
                  {(() => {
                    const label = parserLabel(t.parser);
                    return label ? (
                      <div className={`mt-1.5 text-[10.5px] ${label.warn ? "text-alert" : "text-ink-mute"}`}>{label.text}</div>
                    ) : null;
                  })()}
                </div>
              ) : (
                <p className="border-l-2 border-line pl-2.5 text-xs text-ink-dim">
                  {t.clues.length > 0 ? highlight(t.text, t.clues) : t.text}
                </p>
              )}
            </li>
          );
        })}
        <div ref={endRef} />
      </ol>

      <form onSubmit={submit} className="shrink-0 border-t border-line p-3">
        <label htmlFor="caller-utterance" className="eyebrow mb-1.5 block">
          Caller said
        </label>
        <div className="flex gap-2">
          <input
            id="caller-utterance"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={600}
            autoComplete="off"
            placeholder="Type what the caller just said"
            className="min-w-0 flex-1 rounded border border-line-strong bg-bg px-2.5 py-1.5 text-[13px] text-ink placeholder:text-ink-mute focus:border-accent focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy || draft.trim().length === 0}
            className="flex shrink-0 items-center gap-1 rounded border border-line-strong px-2.5 py-1.5 text-xs font-medium text-ink hover:bg-raised disabled:cursor-not-allowed disabled:opacity-50"
          >
            Add
            <CornerDownLeft size={13} aria-hidden />
          </button>
        </div>
        <p className="mt-1.5 text-[11px] text-ink-mute">
          {parserMode === "llm"
            ? "Clues are extracted by Claude Haiku (one small request, cached)."
            : "Clues are extracted by keyword rules. Switch the parser to Claude Haiku for free-form speech."}
        </p>
      </form>

      <div className="shrink-0 border-t border-line p-3">
        <div className="eyebrow mb-1.5">Evidence in posterior ({clues.length})</div>
        {clues.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {clues.map((c, i) => (
              <ClueChip key={`${c.type}-${c.value}-${i}`} clue={c} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-ink-mute">No caller clues yet - posterior is prior{" "}× phone fix only.</p>
        )}
      </div>

      <div className="shrink-0 border-t border-line p-3">
        <div className="eyebrow mb-1.5 flex items-center gap-1.5">
          <CircleQuestionMark size={12} aria-hidden />
          Suggested next question
        </div>
        {nextQuestion ? (
          <>
            <p className="text-[13px] font-medium text-ink">“{nextQuestion.question}”</p>
            <p className="mt-1 text-[11px] text-ink-mute">
              Expected information gain{" "}
              <span className="num text-ink-dim">{nextQuestion.expected_info_gain_bits.toFixed(2)} bits</span>
              {nextQuestion.clue_type ? (
                <>
                  {" "}- probes {titleCase(nextQuestion.clue_type).toLowerCase()}
                  {nextQuestion.value ? `: ${titleCase(nextQuestion.value).toLowerCase()}` : ""}
                </>
              ) : null}
            </p>
          </>
        ) : (
          <p className="text-xs text-ink-mute">No further question recommended.</p>
        )}
      </div>
    </Panel>
  );
}
