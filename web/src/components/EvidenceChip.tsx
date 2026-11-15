import type { ReactNode } from "react";
import { Headset, ListChecks, MapPin, MapPinOff, RotateCcw, Sparkles, X } from "lucide-react";
import type { Clue, Evidence, EvidenceSource, Room, RoomSelector } from "../types";
import { CLUE_STYLE, titleCase } from "../lib";

// ---------------------------------------------------------------------------
// Plain-word descriptions (shared with the transcript's highlighting and the agent trace)
// ---------------------------------------------------------------------------

const ROUTE_ANCHOR: Record<string, string> = {
  stairs_up: "coming up the stairs",
  stairs_down: "going down the stairs",
  elevator: "leaving the elevator",
  entrance: "coming in the entrance",
};

/** "stairs_up_right" -> "right after coming up the stairs", "landmarks:window" -> "window" */
export function clueValue(c: Pick<Clue, "value">): string {
  const route = /^(?:routes:)?(stairs_up|stairs_down|elevator|entrance)_(left|right|straight)$/.exec(c.value);
  if (route) return `${route[2]} after ${ROUTE_ANCHOR[route[1] ?? ""] ?? "the stairs"}`;
  const v = c.value.includes(":") ? c.value.slice(c.value.indexOf(":") + 1) : c.value;
  return titleCase(v).toLowerCase();
}

export function clueNegated(c: Clue): boolean {
  return c.type === "negative" || !c.polarity;
}

/** Grounded statements are drawn in this colour (vocabulary clues use CLUE_STYLE). */
export const PLACE_COLOR = "#9fc2ff";

/** Type label and colour for an evidence item. */
export function evidenceType(e: Evidence): { label: string; color: string } {
  if (e.kind === "grounded" || !e.clue) return { label: "Place", color: PLACE_COLOR };
  if (e.clue.type === "negative") return { label: "Ruled out", color: CLUE_STYLE.negative.color };
  return CLUE_STYLE[e.clue.type];
}

type RoomIndex = ReadonlyMap<string, Room>;

function roomLabel(rooms: RoomIndex, id: string): string {
  const r = rooms.get(id);
  return r ? r.name : id;
}

function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} or ${items[items.length - 1]}`;
}

/** A room selector in words: "named 405, floor 4", "suite 860", "with windows facing north". */
export function describeSelector(sel: Partial<RoomSelector> | null | undefined, rooms: RoomIndex): string {
  if (!sel) return "a place in the building";
  const parts: string[] = [];
  if (sel.unit) parts.push(`suite ${sel.unit}`);
  if (sel.tenant_contains) parts.push(`tenant "${sel.tenant_contains}"`);
  if (sel.name_contains) parts.push(`named ${sel.name_contains}`);
  if (sel.room_ids && sel.room_ids.length > 0 && parts.length === 0) {
    // name at most two rooms, then "and N more", so the chip stays one line
    const names = sel.room_ids.slice(0, 2).map((id) => roomLabel(rooms, id));
    parts.push(sel.room_ids.length > 2 ? `${names.join(", ")} and ${sel.room_ids.length - 2} more` : list(names));
  }
  if (sel.room_types && sel.room_types.length > 0) parts.push(list(sel.room_types.map((t) => titleCase(t).toLowerCase())));
  if (sel.floor_labels && sel.floor_labels.length > 0)
    parts.push(`${sel.floor_labels.length > 1 ? "floors" : "floor"} ${list(sel.floor_labels)}`);
  if (sel.window_facing) parts.push(`windows facing ${sel.window_facing}`);
  else if (sel.has_windows === true) parts.push("with windows");
  else if (sel.has_windows === false) parts.push("without windows");
  if (sel.near_elevator === true) parts.push("near the elevator");
  else if (sel.near_elevator === false) parts.push("away from the elevator");
  if (sel.near_stairs === true) parts.push("near stairs");
  else if (sel.near_stairs === false) parts.push("away from stairs");
  if (sel.within_m_of) parts.push(`within ${sel.within_m_of.radius_m} m of ${roomLabel(rooms, sel.within_m_of.room_id)}`);
  return parts.length > 0 ? parts.join(", ") : "a place in the building";
}

/** The statement in plain words, without the type label: "not window", "named 403, floor 4". */
export function evidenceValue(e: Evidence, rooms: RoomIndex): string {
  if (e.kind === "grounded" || !e.clue) return `${e.negate ? "not " : ""}${describeSelector(e.selector, rooms)}`;
  return `${clueNegated(e.clue) && e.clue.type !== "negative" ? "not " : ""}${clueValue(e.clue)}`;
}

/** What the statement resolved to in this building: "Executive Office 403", "Suite 860, 3 rooms". */
export function evidenceResolution(e: Evidence, rooms: RoomIndex): string | null {
  if (e.unmatched) return null;
  const ids = e.matched_room_ids;
  const n = ids.length;
  if (n === 0) return null;
  if (e.kind === "grounded") {
    if (n === 1) return roomLabel(rooms, ids[0]!);
    const sel = e.selector;
    const head = sel?.unit ? `Suite ${sel.unit}` : sel?.tenant_contains ? sel.tenant_contains : null;
    if (head) return `${head}, ${n} rooms`;
    if (n === 2) return ids.map((id) => roomLabel(rooms, id)).join(" and ");
    return `${n} rooms`;
  }
  return n === 1 ? roomLabel(rooms, ids[0]!) : `${n} rooms fit`;
}

export const UNMATCHED_TEXT = "Could not be placed in this building. It does not move the estimate.";

// ---------------------------------------------------------------------------
// Source mark
// ---------------------------------------------------------------------------

export const SOURCE_MARK: Record<EvidenceSource, { short: string; label: string; icon: ReactNode }> = {
  rules: { short: "Rules", label: "Read by keyword rules", icon: <ListChecks size={11} aria-hidden /> },
  llm: { short: "Claude", label: "Worked out by Claude from this building's layout", icon: <Sparkles size={11} aria-hidden /> },
  agent: { short: "Claude", label: "Worked out by Claude from this building's layout", icon: <Sparkles size={11} aria-hidden /> },
  dispatcher: { short: "Disp", label: "From the dispatcher: an answer to a question asked", icon: <Headset size={11} aria-hidden /> },
};

export function SourceMark({ source }: { source: EvidenceSource | "record" }) {
  if (source === "record") {
    return (
      <span className="inline-flex shrink-0 items-center rounded-sm border border-line px-1 text-[10px] leading-4 text-ink-mute" title="From the premise record">
        Record
      </span>
    );
  }
  const m = SOURCE_MARK[source];
  return (
    <span
      className="inline-flex shrink-0 items-center gap-0.5 rounded-sm border border-line px-1 text-[10px] leading-4 text-ink-mute"
      title={m.label}
    >
      {m.icon}
      <span>{m.short}</span>
      <span className="sr-only">{m.label}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// The chip
// ---------------------------------------------------------------------------

const STATUS_WORD: Record<Evidence["status"], string> = {
  active: "In use",
  rejected: "Rejected by dispatcher",
  superseded: "Replaced by the caller's correction",
};

interface Props {
  evidence: Evidence;
  rooms: RoomIndex;
  /** "row": two lines, under a transcript turn. "pill": one line, in the evidence list. */
  variant?: "row" | "pill";
  /** a request is in flight; `busy === "evidence:<id>"` is this chip's own */
  busy: string | null;
  onReject: (id: string) => void;
  onRestore: (id: string) => void;
  /** rooms this statement points at, or null when the pointer leaves */
  onHoverRooms: (roomIds: string[] | null) => void;
}

/**
 * One statement the posterior can use: type, value in plain words, confidence and where it came
 * from. Grounded statements say what they resolved to. Rejected and superseded ones stay visible,
 * struck through, with Restore. Hover or focus outlines the rooms it points at on the map.
 */
export function EvidenceChip({ evidence: e, rooms, variant = "row", busy, onReject, onRestore, onHoverRooms }: Props) {
  const type = evidenceType(e);
  const value = evidenceValue(e, rooms);
  const resolution = evidenceResolution(e, rooms);
  const active = e.status === "active";
  const mine = busy === `evidence:${e.id}`;
  const conf = Math.round(e.confidence * 100);
  const ids = e.unmatched ? [] : e.matched_room_ids;

  const title = [
    `${type.label}: ${value}`,
    `Confidence ${conf}%`,
    SOURCE_MARK[e.source]?.label ?? e.source,
    e.span ? `Caller's words: "${e.span}"` : null,
    e.unmatched ? UNMATCHED_TEXT : resolution ? `Points at: ${resolution}` : null,
    e.rationale ? `Why: ${e.rationale}` : null,
    active ? null : STATUS_WORD[e.status],
  ]
    .filter(Boolean)
    .join("\n");

  const hover = {
    onMouseEnter: () => onHoverRooms(ids.length > 0 ? ids : null),
    onMouseLeave: () => onHoverRooms(null),
    onFocus: () => onHoverRooms(ids.length > 0 ? ids : null),
    onBlur: () => onHoverRooms(null),
  };

  const action = mine ? (
    <span className="shrink-0 px-1 text-[10.5px] text-ink-mute" role="status">
      Saving…
    </span>
  ) : active ? (
    <button
      type="button"
      onClick={() => onReject(e.id)}
      disabled={busy !== null}
      aria-label={`Reject ${type.label.toLowerCase()} ${value}`}
      title="Reject: the estimate stops using this statement"
      className="-mr-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded text-ink-mute hover:bg-alert/15 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
    >
      <X size={13} aria-hidden />
    </button>
  ) : (
    <button
      type="button"
      onClick={() => onRestore(e.id)}
      disabled={busy !== null}
      aria-label={`Restore ${type.label.toLowerCase()} ${value}`}
      title="Restore: the estimate uses this statement again"
      className="flex h-5 shrink-0 items-center gap-1 rounded border border-line-strong px-1.5 text-[10.5px] font-medium text-ink-dim hover:bg-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
    >
      <RotateCcw size={11} aria-hidden />
      Restore
    </button>
  );

  const dot = (
    <span
      className={`h-2 w-2 shrink-0 ${e.kind === "grounded" ? "rotate-45 rounded-[1px]" : "rounded-full"}`}
      style={{ background: active ? type.color : "transparent", boxShadow: active ? undefined : `inset 0 0 0 1.5px ${type.color}` }}
      aria-hidden
    />
  );

  const valueText = (
    <span
      className={`min-w-0 font-medium ${active ? "text-ink" : "text-ink-mute line-through decoration-ink-mute"} ${e.unmatched && active ? "text-ink-dim" : ""}`}
    >
      {value}
    </span>
  );

  if (variant === "pill") {
    return (
      <span
        {...hover}
        title={title}
        data-evidence-id={e.id}
        className={`inline-flex max-w-full items-center gap-1.5 rounded border px-1.5 py-0.5 text-[11.5px] hover:border-ink-mute ${
          active ? "border-line-strong bg-raised" : "border-dashed border-line bg-transparent"
        }`}
      >
        {dot}
        <span className="shrink-0 text-ink-mute">{type.label}</span>
        <span className="truncate">{valueText}</span>
        {e.unmatched ? (
          <span className="flex shrink-0 items-center gap-0.5 text-[10px] uppercase tracking-wide text-warn">
            <MapPinOff size={11} aria-hidden />
            Not placed
          </span>
        ) : null}
        <span className="num shrink-0 text-[10.5px] text-ink-dim">{conf}%</span>
        {active ? null : <span className="shrink-0 text-[10px] uppercase tracking-wide text-ink-mute">{e.status === "rejected" ? "Rejected" : "Replaced"}</span>}
        {action}
      </span>
    );
  }

  return (
    <div
      {...hover}
      title={title}
      data-evidence-id={e.id}
      className={`rounded border px-2 py-1 text-[12px] leading-snug hover:border-ink-mute ${
        active ? "border-line-strong bg-bg/60" : "border-dashed border-line bg-transparent"
      }`}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        {dot}
        <span className="shrink-0 text-[11px] text-ink-mute">{type.label}</span>
        <span className="min-w-0 flex-1 truncate">{valueText}</span>
        <span className="num shrink-0 text-[11px] text-ink-dim" aria-label={`confidence ${conf}%`}>
          {conf}%
        </span>
        <SourceMark source={e.source} />
        {action}
      </div>
      {e.unmatched ? (
        <div className="mt-0.5 flex items-start gap-1 text-[11px] text-warn">
          <MapPinOff size={11} className="mt-[2px] shrink-0" aria-hidden />
          <span>{UNMATCHED_TEXT}</span>
        </div>
      ) : resolution || !active ? (
        <div className="mt-0.5 flex min-w-0 items-center gap-1 pl-3.5 text-[11px] text-ink-mute">
          {resolution ? (
            <>
              <MapPin size={10} className="shrink-0" aria-hidden />
              <span className={`truncate ${e.kind === "grounded" && active ? "text-ink-dim" : ""}`}>{resolution}</span>
            </>
          ) : null}
          {!active ? <span className="ml-auto shrink-0 font-medium">{STATUS_WORD[e.status]}</span> : null}
        </div>
      ) : null}
      {/* why a statement was tied to these rooms: the dispatcher decides whether to trust it */}
      {e.kind === "grounded" && active && !e.unmatched && e.rationale ? (
        <div className="mt-0.5 truncate pl-3.5 text-[10.5px] italic text-ink-mute">{e.rationale}</div>
      ) : null}
    </div>
  );
}
