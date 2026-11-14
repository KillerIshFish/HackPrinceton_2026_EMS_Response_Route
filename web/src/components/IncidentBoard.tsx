import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Check, ClipboardCopy, Crosshair, HeartPulse, Radio, UsersRound, FileText } from "lucide-react";
import type { Building, Fact, FactCategory, TranscriptTurn } from "../types";
import { roomIndex } from "../lib";
import { Panel } from "./Panel";
import { clockTime } from "./PremiseFacts";

interface Props {
  facts: Fact[];
  transcript: TranscriptTurn[];
  building: Building;
  /** ISO time the call started, for "+m:ss into the call" */
  callStartedAt: string | null;
}

/** Board sections, in relay order. Facts in other categories go under "Other reports". */
const SECTIONS: { key: FactCategory | "other"; label: string; radio: string; icon: ReactNode }[] = [
  { key: "threat", label: "Threat", radio: "THREAT", icon: <Crosshair size={14} aria-hidden /> },
  { key: "patient", label: "Patients", radio: "PATIENTS", icon: <HeartPulse size={14} aria-hidden /> },
  { key: "people", label: "People", radio: "PEOPLE", icon: <UsersRound size={14} aria-hidden /> },
  { key: "other", label: "Other reports", radio: "OTHER REPORTS", icon: <FileText size={14} aria-hidden /> },
];
const BOARD_CATEGORIES: ReadonlySet<FactCategory> = new Set(["threat", "patient", "people"]);

interface Entry {
  fact: Fact;
  time: string | null;
  elapsed: string | null;
  /** "Caller 2", from the transcript turn's title */
  who: string;
  room: string | null;
}

function elapsedSince(start: string | null, at: string | null): string | null {
  if (!start || !at) return null;
  const s = Math.round((Date.parse(at) - Date.parse(start)) / 1000);
  if (!Number.isFinite(s) || s < 0) return null;
  return `+${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function sourceName(f: Fact, transcript: TranscriptTurn[]): string {
  if (f.source === "record") return "Premise record";
  if (f.source === "dispatcher") return "Dispatcher";
  const turn = f.turn !== null ? transcript[f.turn] : undefined;
  if (turn?.speaker === "caller" && turn.title && /caller/i.test(turn.title)) return turn.title;
  return "Caller";
}

/** The board as plain text, newest first in each section, for reading out on the radio. */
export function radioText(sections: { radio: string; entries: Entry[] }[], buildingName: string): string {
  const now = clockTime(new Date().toISOString());
  const lines = [`FOR RELAY TO UNIFIED COMMAND - ${buildingName} - as of ${now}`, "All items reported by callers, unverified.", ""];
  for (const s of sections) {
    if (s.entries.length === 0) continue;
    lines.push(s.radio);
    for (const e of s.entries) {
      const at = [e.time ?? "time not logged", e.elapsed ? `(${e.elapsed})` : null].filter(Boolean).join(" ");
      const where = e.room ? ` Location given: ${e.room}.` : "";
      const words = e.fact.span ? ` Words: "${e.fact.span}".` : "";
      lines.push(`${at} - ${e.who}: ${e.fact.text.replace(/\.$/, "")}.${where}${words}`);
    }
    lines.push("");
  }
  return lines.join("\n").trimEnd() + "\n";
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // clipboard API blocked (no permission, insecure context): fall back to a hidden textarea
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

/**
 * Active threat: replaces the search plan. A relay log in the style of an ICS-214 activity log,
 * for the dispatcher to pass caller reports to unified command: what was said, when, by which
 * caller and where. It ranks nothing, routes nothing and has no control that contacts a caller.
 */
export function IncidentBoard({ facts, transcript, building, callStartedAt }: Props) {
  const rooms = roomIndex(building);
  const [copied, setCopied] = useState<"ok" | "failed" | null>(null);
  const [showText, setShowText] = useState(false);
  const timer = useRef<number | null>(null);
  useEffect(() => () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
  }, []);

  const entry = (f: Fact): Entry => {
    const room = f.room_id ? rooms.get(f.room_id) : undefined;
    return {
      fact: f,
      time: clockTime(f.observed_at),
      elapsed: elapsedSince(callStartedAt, f.observed_at),
      who: sourceName(f, transcript),
      room: room ? `${room.name}, Floor ${room.floor_label}` : f.room_id,
    };
  };
  const newestFirst = (a: Fact, b: Fact) =>
    (Date.parse(b.observed_at ?? "") || 0) - (Date.parse(a.observed_at ?? "") || 0) || (b.turn ?? -1) - (a.turn ?? -1);

  const sections = SECTIONS.map((s) => ({
    ...s,
    entries: facts
      .filter((f) => (s.key === "other" ? !BOARD_CATEGORIES.has(f.category) && !f.premise_key : f.category === s.key))
      .sort(newestFirst)
      .map(entry),
  })).filter((s) => s.key !== "other" || s.entries.length > 0);
  const total = sections.reduce((n, s) => n + s.entries.length, 0);
  const text = radioText(sections, building.name);

  const onCopy = async () => {
    const ok = await copyText(text);
    setCopied(ok ? "ok" : "failed");
    if (!ok) setShowText(true);
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied(null), 2500);
  };

  return (
    <Panel
      title="Incident board"
      icon={<Radio size={15} aria-hidden />}
      className="h-full"
      bodyClassName="flex flex-col"
      actions={
        <button
          onClick={() => void onCopy()}
          disabled={total === 0}
          className="flex items-center gap-1.5 rounded border border-line-strong px-2 py-1 text-xs font-semibold text-ink hover:bg-raised disabled:cursor-not-allowed disabled:opacity-50"
          title="Copy every report below as plain text with times, newest first"
        >
          {copied === "ok" ? <Check size={13} aria-hidden /> : <ClipboardCopy size={13} aria-hidden />}
          {copied === "ok" ? "Copied" : copied === "failed" ? "Copy failed - text shown" : "Copy for radio"}
        </button>
      }
    >
      <div className="shrink-0 border-b border-line bg-bg px-3 py-2">
        <p className="text-[13px] font-semibold uppercase tracking-wider text-ink">For relay to unified command</p>
        <p className="text-[11px] text-ink-mute">
          Caller reports, newest first in each section. All unverified. Clock time, then time into the call.
        </p>
      </div>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto" aria-live="polite">
        {sections.map((s) => (
          <section key={s.key} aria-label={s.label} className="border-b border-line last:border-b-0">
            <h3 className="sticky top-0 z-[1] flex items-center gap-1.5 border-b border-line bg-panel px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-ink-dim">
              <span className="text-ink-mute">{s.icon}</span>
              {s.label}
              <span className="num ml-auto font-normal text-ink-mute">{s.entries.length}</span>
            </h3>
            {s.entries.length === 0 ? (
              <p className="px-3 py-1.5 text-[12px] italic text-ink-mute">Nothing reported yet</p>
            ) : (
              <ol className="divide-y divide-line">
                {s.entries.map((e) => (
                  <li key={e.fact.id} className="grid grid-cols-[80px_minmax(0,1fr)] gap-x-2 px-3 py-1.5">
                    <div className="text-[11px] leading-tight">
                      <div className="num text-ink">{e.time ?? "-"}</div>
                      {e.elapsed ? <div className="num text-ink-mute">{e.elapsed}</div> : null}
                    </div>
                    <div className="min-w-0">
                      <div className="text-[13px] font-medium leading-snug text-ink">{e.fact.text}</div>
                      {e.fact.span ? (
                        <div className="text-[12px] italic leading-snug text-ink-dim">"{e.fact.span}"</div>
                      ) : null}
                      <div className="mt-0.5 flex flex-wrap gap-x-2 text-[11px] text-ink-mute">
                        <span className="font-semibold text-ink-dim">{e.who}</span>
                        {e.room ? <span>Location given: {e.room}</span> : null}
                        <span>{e.fact.verified ? "verified" : "unverified"}</span>
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
        ))}
      </div>

      <div className="shrink-0 border-t border-line">
        <button
          onClick={() => setShowText((v) => !v)}
          aria-expanded={showText}
          className="w-full px-3 py-1 text-left text-[11px] text-ink-mute hover:bg-raised hover:text-ink"
        >
          {showText ? "Hide radio text" : "Show radio text"}
        </button>
        {showText ? (
          <pre className="num scroll-thin max-h-48 overflow-auto whitespace-pre-wrap border-t border-line bg-bg px-3 py-2 text-[11px] leading-snug text-ink">
            {text}
          </pre>
        ) : null}
      </div>
    </Panel>
  );
}
