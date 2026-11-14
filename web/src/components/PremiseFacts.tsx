import { useState } from "react";
import type { ReactNode } from "react";
import { ChevronDown, ClipboardList, DoorClosed, HeartPulse, MoveVertical, UserRound } from "lucide-react";
import type { Building, Fact, FactSource, PremiseKey, Situation } from "../types";
import { roomIndex } from "../lib";
import { Panel } from "./Panel";

interface Props {
  facts: Fact[];
  situation: Situation;
  building: Building;
  /** false when fire, smoke or an alarm is reported: the elevator row says it is not offered for routing */
  elevatorAllowed: boolean;
}

const ROWS: { key: PremiseKey; label: string; short: string; icon: ReactNode }[] = [
  { key: "means_of_entry", label: "Means of entry", short: "Entry", icon: <DoorClosed size={14} aria-hidden /> },
  { key: "meet_person", label: "Who meets the crew", short: "Meets crew", icon: <UserRound size={14} aria-hidden /> },
  { key: "elevator", label: "Elevator", short: "Elevator", icon: <MoveVertical size={14} aria-hidden /> },
  { key: "defibrillator", label: "Defibrillator", short: "Defibrillator", icon: <HeartPulse size={14} aria-hidden /> },
];

const OPEN_KEY = "console.premiseFacts";

/** Open by default on a tall window; the choice is remembered in this browser (best effort). */
function initialOpen(): boolean {
  try {
    const v = window.localStorage.getItem(OPEN_KEY);
    if (v === "open" || v === "closed") return v === "open";
  } catch {
    /* storage unavailable */
  }
  return window.innerHeight >= 1000;
}

/** Shown in place of any entry code: the console has no sign-in, so codes never reach the screen. */
export const HELD_BY_DISPATCH = "Held by dispatch";

const CODE_WORDS = /\b(?:code|codes|pin|passcode|password|combination|keypad|lockbox|key\s*box|knox)\b/i;
/** Three or more digits in a row, allowing spaces, dashes, # and * between them (keypad codes). */
const DIGIT_RUN = /[#*]?\d(?:[\s\-#*]?\d){2,}[#*]?/g;

/**
 * Entry text with any code taken out. Digit runs become "[held by dispatch]"; when the text
 * names a code without digits ("code is four four two one") the whole text is withheld.
 */
export function redactEntry(text: string): { text: string; withheld: boolean } {
  const replaced = text.replace(DIGIT_RUN, "[held by dispatch]");
  if (replaced !== text) return { text: replaced, withheld: true };
  if (CODE_WORDS.test(text) && /\b(?:is|=|:)\s*\S/.test(text)) return { text: "Entry details held by dispatch", withheld: true };
  return { text, withheld: CODE_WORDS.test(text) };
}

const CALLER: ReadonlySet<FactSource> = new Set(["rules", "llm", "agent"]);

/** Local clock time, "01:17:23 PM", the same form the transcript uses. */
export function clockTime(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function recordDate(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/** Where a fact came from and when, in words. */
function provenance(f: Fact): { who: string; when: string | null; caller: boolean } {
  if (f.source === "record") return { who: "Premise record", when: recordDate(f.observed_at), caller: false };
  if (f.source === "dispatcher") return { who: "Entered by dispatcher", when: clockTime(f.observed_at), caller: false };
  return {
    who: f.verified ? "Reported by caller" : "Reported by caller, unverified",
    when: clockTime(f.observed_at),
    caller: CALLER.has(f.source),
  };
}

interface Item {
  id: string;
  text: string;
  who: string;
  when: string | null;
  caller: boolean;
  withheld: boolean;
  span: string | null;
  room: string | null;
}

/**
 * Premise facts, like the pre-plan record in a CAD premise file: means of entry, who meets the
 * crew, the elevator and the defibrillator. Every line says where it came from and when. A row
 * with nothing on file reads "Not recorded", never "none".
 */
export function PremiseFacts({ facts, situation, building, elevatorAllowed }: Props) {
  const rooms = roomIndex(building);
  const byKey = new Map<PremiseKey, Item[]>();
  for (const f of facts) {
    if (!f.premise_key) continue;
    const p = provenance(f);
    const entry = f.premise_key === "means_of_entry" ? redactEntry(f.text) : { text: f.text, withheld: false };
    const room = f.room_id ? rooms.get(f.room_id) : undefined;
    const list = byKey.get(f.premise_key) ?? [];
    list.push({
      id: f.id,
      text: entry.text,
      who: p.who,
      when: p.when,
      caller: p.caller,
      withheld: entry.withheld,
      // the caller's own words, unless they could carry an entry code
      span: f.span && f.premise_key !== "means_of_entry" && f.span !== f.text ? f.span : null,
      room: room ? `${room.name}, Floor ${room.floor_label}` : null,
    });
    byKey.set(f.premise_key, list);
  }
  // the dispatcher's own record of who will meet the crew, when no statement carries it
  if (situation.meet_person && !(byKey.get("meet_person")?.length)) {
    byKey.set("meet_person", [
      { id: "situation-meet", text: situation.meet_person, who: "Entered by dispatcher", when: null, caller: false, withheld: false, span: null, room: null },
    ]);
  }
  // newest first within a row (records have their filing date)
  for (const list of byKey.values()) list.reverse();

  const [open, setOpenState] = useState(initialOpen);
  const setOpen = (v: boolean) => {
    setOpenState(v);
    try {
      window.localStorage.setItem(OPEN_KEY, v ? "open" : "closed");
    } catch {
      /* ignore */
    }
  };

  return (
    <Panel
      title="Premise facts"
      icon={<ClipboardList size={15} aria-hidden />}
      actions={
        <button
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="flex items-center gap-1 rounded px-1 text-[11px] text-ink-mute hover:bg-raised hover:text-ink"
        >
          {open ? "Hide details" : "Show details"}
          <ChevronDown size={13} className={`transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
        </button>
      }
      className="shrink-0"
      bodyClassName="scroll-thin max-h-[30vh] overflow-y-auto"
    >
      {!open ? (
        // one line per row: what is on file and where it came from; details hold the text and times
        <ul className="grid grid-cols-1 gap-px bg-line 2xl:grid-cols-2">
          {ROWS.map((row) => {
            const top = byKey.get(row.key)?.[0];
            return (
              <li key={row.key} className="flex min-w-0 items-center gap-1.5 bg-panel px-3 py-1 text-[11px]" title={top ? `${top.text} - ${top.who}${top.when ? `, ${top.when}` : ""}` : "Not recorded"}>
                <span className="shrink-0 text-ink-mute">{row.icon}</span>
                <span className="shrink-0 font-medium text-ink-dim">{row.short}</span>
                <span className={`truncate ${top ? (top.caller ? "font-semibold text-warn" : "text-ink") : "italic text-ink-mute"}`}>
                  {top ? (top.caller ? "caller, unverified" : top.who === "Premise record" ? "on record" : "dispatcher") : "Not recorded"}
                </span>
              </li>
            );
          })}
        </ul>
      ) : (
      <dl className="divide-y divide-line">
        {ROWS.map((row) => {
          const items = byKey.get(row.key) ?? [];
          const codeNote =
            row.key === "means_of_entry" ? (
              <span>
                Entry codes: <span className="font-semibold text-ink-dim">{HELD_BY_DISPATCH}</span>
              </span>
            ) : null;
          return (
            <div key={row.key} className="grid grid-cols-[118px_minmax(0,1fr)] gap-x-2 px-3 py-1 2xl:grid-cols-[136px_minmax(0,1fr)]">
              <dt className="flex items-start gap-1.5 pt-px text-[12px] font-medium text-ink-dim">
                <span className="mt-px text-ink-mute">{row.icon}</span>
                {row.label}
              </dt>
              <dd className="m-0 min-w-0 space-y-0.5">
                {items.length === 0 ? (
                  <div className="flex flex-wrap items-baseline gap-x-2 text-[11px] text-ink-mute">
                    <span className="text-[12px] italic">Not recorded</span>
                    {codeNote}
                  </div>
                ) : (
                  items.map((it, i) => (
                    <div key={it.id} className="min-w-0">
                      <div className="text-[12px] leading-snug text-ink" title={it.span ? `Caller's words: "${it.span}"` : undefined}>
                        {it.text}
                        {it.room ? <span className="text-ink-dim"> ({it.room})</span> : null}
                      </div>
                      <div className="flex flex-wrap items-baseline gap-x-1.5 text-[11px] text-ink-mute">
                        <span className={it.caller ? "font-semibold text-warn" : ""}>{it.who}</span>
                        {it.when ? <span className="num">{it.when}</span> : null}
                        {i === items.length - 1 && codeNote ? <span aria-hidden>·</span> : null}
                        {i === items.length - 1 ? codeNote : null}
                      </div>
                    </div>
                  ))
                )}
                {row.key === "elevator" && !elevatorAllowed && items.length > 0 ? (
                  <div className="text-[11px] text-ink-dim">Not offered for routing while fire, smoke or an alarm is reported</div>
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>
      )}
    </Panel>
  );
}
