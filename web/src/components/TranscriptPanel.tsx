import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { CircleQuestionMark, MessageSquare, Smartphone } from "lucide-react";
import type { Clue, NextQuestion, PhoneFix, TranscriptTurn } from "../types";
import { CLUE_STYLE, titleCase } from "../lib";
import { Panel } from "./Panel";

interface Props {
  transcript: TranscriptTurn[];
  clues: Clue[];
  nextQuestion: NextQuestion | null;
  turn: number;
  totalTurns: number;
}

function clueTitle(c: Clue): string {
  const negated = c.type === "negative" || !c.polarity;
  return `${CLUE_STYLE[c.type].label}: ${negated ? "NOT " : ""}${titleCase(c.value)} - confidence ${Math.round(c.confidence * 100)}% - polarity ${negated ? "negative" : "positive"}`;
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
        {titleCase(clue.value).toLowerCase()}
      </span>
      <span className="num text-ink-mute">{Math.round(clue.confidence * 100)}%</span>
    </span>
  );
}

export function TranscriptPanel({ transcript, clues, nextQuestion, turn, totalTurns }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [transcript.length]);

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
