import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { Check, CircleAlert, Flame, Info, Lightbulb, OctagonAlert, TriangleAlert, X } from "lucide-react";
import type { Banner, BannerLevel, CrewLoad, IncidentType, Situation, SituationPatch } from "../types";
import { INCIDENT_TYPES } from "../types";

/** Plain labels for the incident type selector (what the dispatcher reads, not the enum). */
export const INCIDENT_LABEL: Record<IncidentType, string> = {
  unknown: "Not set yet",
  cardiac_arrest: "Cardiac arrest",
  stroke: "Stroke",
  internal_bleeding: "Internal bleeding",
  external_bleeding: "External bleeding",
  overdose: "Overdose",
  fire_smoke: "Fire or smoke",
  active_threat: "Active threat",
  other_medical: "Other medical",
};

const TEAM_OPTIONS = [1, 2];

interface Props {
  situation: Situation;
  nTeams: number;
  /** a request is in flight: controls are disabled until the new state arrives */
  busy: boolean;
  /** every control posts a patch to /calls/demo/situation */
  onPatch: (patch: SituationPatch) => void;
}

/**
 * The call's situation, always visible under the header: call clock, incident type (with the
 * interpreter's suggestion to confirm or dismiss), crew load, teams and smoke. Laid out like a
 * CAD incident header: one dense row of labelled cells, colour only for status.
 */
export function SituationBar({ situation, nTeams, busy, onPatch }: Props) {
  const s = situation;
  const pendingSuggestion = s.suggested_type !== null && s.suggested_type !== s.incident_type && !s.incident_confirmed;
  const unconfirmed = !s.incident_confirmed && s.incident_type !== "unknown";

  return (
    <div
      role="region"
      aria-label="Situation"
      className="flex flex-wrap items-stretch border-b border-line bg-panel"
    >
      <Cell label="Call time" className="min-w-[124px]">
        <CallClock startedAt={s.call_started_at} />
      </Cell>

      <Cell label="Incident type" className="min-w-[210px]">
        <div className="flex items-center gap-2">
          <select
            aria-label="Incident type"
            value={s.incident_type}
            disabled={busy}
            onChange={(e) => onPatch({ incident_type: e.target.value as IncidentType })}
            className={`h-8 cursor-pointer rounded border bg-bg px-2 text-[14px] font-semibold text-ink focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 ${
              s.incident_type === "active_threat"
                ? "border-alert"
                : s.incident_type === "unknown"
                  ? "border-line-strong text-ink-dim"
                  : "border-line-strong"
            }`}
          >
            {INCIDENT_TYPES.map((t) => (
              <option key={t} value={t}>
                {INCIDENT_LABEL[t]}
              </option>
            ))}
          </select>
          {unconfirmed ? (
            <button
              disabled={busy}
              onClick={() => onPatch({ incident_type: s.incident_type, incident_confirmed: true })}
              title="The type is set but not confirmed. Confirm it once the caller's account supports it."
              className="flex h-8 items-center gap-1 rounded border border-warn/60 px-2 text-xs font-semibold text-warn hover:bg-warn/10 disabled:opacity-50"
            >
              <CircleAlert size={13} aria-hidden />
              Not confirmed
            </button>
          ) : null}
        </div>
      </Cell>

      {pendingSuggestion && s.suggested_type ? (
        <Cell label="Suggested from the call" className="min-w-0 flex-1 basis-[300px] bg-warn/[0.06]">
          <div className="flex min-w-0 items-center gap-2">
            <Lightbulb size={15} className="shrink-0 text-warn" aria-hidden />
            <span className="shrink-0 text-[14px] font-semibold text-ink">{INCIDENT_LABEL[s.suggested_type]}</span>
            {s.suggested_span ? (
              <q
                className="min-w-0 truncate text-[13px] italic text-ink-dim"
                title={`Caller's words: "${s.suggested_span}"`}
              >
                {s.suggested_span}
              </q>
            ) : null}
            <span className="ml-auto flex shrink-0 items-center gap-1.5">
              <button
                disabled={busy}
                onClick={() =>
                  s.suggested_type && onPatch({ incident_type: s.suggested_type, incident_confirmed: true })
                }
                className="flex h-7 items-center gap-1 rounded bg-ink px-2.5 text-xs font-semibold text-bg hover:bg-white disabled:opacity-50"
              >
                <Check size={13} aria-hidden />
                Confirm
              </button>
              <button
                disabled={busy}
                onClick={() => onPatch({ dismiss_suggestion: true })}
                className="flex h-7 items-center gap-1 rounded border border-line-strong px-2 text-xs text-ink-dim hover:bg-raised hover:text-ink disabled:opacity-50"
              >
                <X size={13} aria-hidden />
                Dismiss
              </button>
            </span>
          </div>
        </Cell>
      ) : (
        <div className="hidden flex-1 border-l border-line xl:block" aria-hidden />
      )}

      <Cell label="Crew load">
        <Segmented
          label="Crew load"
          value={s.load}
          disabled={busy}
          options={[
            { value: "light", label: "Light kit" },
            { value: "heavy", label: "Heavy kit" },
          ]}
          onChange={(v) => onPatch({ load: v as CrewLoad })}
        />
      </Cell>

      <Cell label="Teams" title="Search teams sent in; each team is a pair">
        <Segmented
          label="Search teams"
          value={String(nTeams)}
          disabled={busy}
          mono
          options={TEAM_OPTIONS.map((n) => ({ value: String(n), label: String(n) }))}
          onChange={(v) => onPatch({ n_teams: Number(v) })}
        />
      </Cell>

      <Cell label="Smoke">
        <button
          aria-pressed={s.smoke_reported}
          disabled={busy}
          onClick={() => onPatch({ smoke_reported: !s.smoke_reported })}
          className={`flex h-8 items-center gap-1.5 rounded border px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-60 ${
            s.smoke_reported
              ? "border-alert bg-alert/15 font-semibold text-ink"
              : "border-line-strong text-ink-dim hover:bg-raised hover:text-ink"
          }`}
        >
          <Flame size={14} className={s.smoke_reported ? "text-alert" : "text-ink-mute"} aria-hidden />
          {s.smoke_reported ? "Smoke reported" : "No smoke reported"}
        </button>
      </Cell>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Cell({
  label,
  title,
  className = "",
  children,
}: {
  label: string;
  title?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={`flex flex-col justify-center gap-1 border-l border-line px-3 py-1.5 first:border-l-0 ${className}`}
      title={title}
    >
      <span className="eyebrow !text-[10px] leading-none">{label}</span>
      {children}
    </div>
  );
}

/** Single-choice segmented control (Radix toggle group: roving focus, arrow keys). Never empty. */
function Segmented({
  label,
  value,
  options,
  disabled,
  mono = false,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  disabled: boolean;
  mono?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <ToggleGroup.Root
      type="single"
      aria-label={label}
      value={value}
      disabled={disabled}
      onValueChange={(v) => {
        if (v && v !== value) onChange(v);
      }}
      className="flex h-8 overflow-hidden rounded border border-line-strong"
    >
      {options.map((o) => (
        <ToggleGroup.Item
          key={o.value}
          value={o.value}
          className={`border-l border-line-strong px-2.5 text-[13px] first:border-l-0 disabled:cursor-not-allowed data-[state=off]:text-ink-dim data-[state=off]:enabled:hover:bg-raised data-[state=off]:enabled:hover:text-ink data-[state=on]:bg-ink data-[state=on]:font-semibold data-[state=on]:text-bg ${
            mono ? "num min-w-9" : ""
          }`}
        >
          {o.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

/** Seconds since `iso`, refreshed every second. */
function useElapsed(iso: string | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!iso) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [iso]);
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? Math.max(0, (now - t) / 1000) : null;
}

/** "4:07", or "1:02:09" past the hour */
export function clockText(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function CallClock({ startedAt }: { startedAt: string | null }) {
  const elapsed = useElapsed(startedAt);
  const started = startedAt ? new Date(startedAt) : null;
  return (
    <div className="flex items-baseline gap-2">
      <span
        className="num text-[24px] font-semibold leading-none text-ink"
        role="timer"
        aria-label={elapsed === null ? "Call time not recorded" : `Call time ${clockText(elapsed)}`}
      >
        {elapsed === null ? "-:--" : clockText(elapsed)}
      </span>
      {started ? (
        <span className="num text-[11px] text-ink-mute" title={`Call started ${started.toLocaleString()}`}>
          from {started.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Policy banners: directly under the situation bar. Level is coded three ways (colour, icon,
// word) so it survives colour blindness and a glance; each banner names its source.
// ---------------------------------------------------------------------------

const LEVEL_STYLE: Record<BannerLevel, { word: string; icon: ReactNode; box: string; tone: string }> = {
  critical: {
    word: "Critical",
    icon: <OctagonAlert size={15} aria-hidden />,
    box: "border-alert bg-alert/15",
    tone: "text-alert",
  },
  warning: {
    word: "Warning",
    icon: <TriangleAlert size={15} aria-hidden />,
    box: "border-warn bg-warn/[0.08]",
    tone: "text-warn",
  },
  info: {
    word: "Info",
    icon: <Info size={15} aria-hidden />,
    box: "border-info/70 bg-info/[0.06]",
    tone: "text-info",
  },
};

const LEVEL_ORDER: Record<BannerLevel, number> = { critical: 0, warning: 1, info: 2 };

export function PolicyBanners({ banners, prompts = [] }: { banners: Banner[]; prompts?: string[] }) {
  if (banners.length === 0 && prompts.length === 0) return null;
  // most severe first; the server's order is kept within a level
  const sorted = banners
    .map((b, i) => ({ b, i }))
    .sort((x, y) => LEVEL_ORDER[x.b.level] - LEVEL_ORDER[y.b.level] || x.i - y.i)
    .map(({ b }) => b);
  return (
    <div
      role="region"
      aria-label="Notices for this incident"
      className={`grid gap-px border-b border-line bg-line ${sorted.length + prompts.length > 2 ? "xl:grid-cols-2" : ""}`}
    >
      {sorted.map((b, i) => {
        const st = LEVEL_STYLE[b.level];
        return (
          <div
            key={`${b.level}-${i}-${b.text}`}
            role={b.level === "critical" ? "alert" : "status"}
            className={`flex min-w-0 items-center gap-2 border-l-4 bg-bg px-3 py-1 xl:odd:last:col-span-2 ${st.box}`}
          >
            <span className={`flex shrink-0 items-center gap-1.5 ${st.tone}`}>
              {st.icon}
              <span className="w-[68px] text-[11px] font-bold uppercase tracking-wider">{st.word}</span>
            </span>
            <span className="min-w-0 flex-1 text-[13px] font-medium text-ink">{b.text}</span>
            <span className="max-w-[40%] truncate text-[11px] text-ink-mute" title={`Source: ${b.source}`}>
              {b.source}
            </span>
          </div>
        );
      })}
      {prompts.map((p) => (
        <div key={p} role="status" className="flex min-w-0 items-center gap-2 border-l-4 border-line-strong bg-bg px-3 py-1 xl:odd:last:col-span-2">
          <span className="flex shrink-0 items-center gap-1.5 text-ink-dim">
            <Info size={15} aria-hidden />
            <span className="w-[68px] text-[11px] font-bold uppercase tracking-wider">Protocol</span>
          </span>
          <span className="min-w-0 flex-1 text-[13px] text-ink">{p}</span>
        </div>
      ))}
    </div>
  );
}
