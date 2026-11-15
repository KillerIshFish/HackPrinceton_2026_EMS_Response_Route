import { useState } from "react";
import type { ReactNode } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import * as Tooltip from "@radix-ui/react-tooltip";
import { Ban } from "lucide-react";

export type ViewId = "plan" | "3d" | "scan" | "approach";

export interface ViewDef {
  id: ViewId;
  label: string;
  icon?: ReactNode;
  content: ReactNode;
  /** shown greyed out and struck through, with `unavailableReason` in a tooltip on hover; it cannot be opened */
  disabled?: boolean;
  unavailableReason?: string;
}

interface Props {
  views: ViewDef[];
  value: ViewId;
  onValueChange: (id: ViewId) => void;
  /** right-aligned content on the tab row */
  actions?: ReactNode;
}

/**
 * The map views as tabs in the centre column, so the transcript, one map view and the search plan
 * fit one screen. Radix Tabs supplies the ARIA roles and arrow-key navigation. A view is mounted the
 * first time it is opened and then kept (hidden) so the 3D scene and the scan viewer are not rebuilt
 * on every switch. A view that is not available is greyed out, struck through and marked with a
 * "not available" icon, so it reads as off at a glance; pointing at it says why.
 */
export function ViewTabs({ views, value, onValueChange, actions }: Props) {
  const usable = views.filter((v) => !v.disabled);
  // a remembered view that is unavailable here (e.g. Scan with no tour) falls back to the first usable one
  const active: ViewId = usable.some((v) => v.id === value) ? value : (usable[0]?.id ?? value);
  const [visited, setVisited] = useState<Set<ViewId>>(() => new Set([active]));
  // set during render (React's "adjust state while rendering" pattern) so a newly opened view mounts this pass
  if (!visited.has(active)) setVisited(new Set(visited).add(active));

  return (
    <Tabs.Root
      value={active}
      onValueChange={(v) => onValueChange(v as ViewId)}
      className="flex min-h-0 min-w-0 flex-col gap-2 xl:h-full"
    >
      <div className="flex shrink-0 items-end gap-3 border-b border-line">
        <Tabs.List aria-label="Map views" className="flex">
          <Tooltip.Provider delayDuration={150} skipDelayDuration={100}>
            {views.map((v) => {
              if (!v.disabled) {
                return (
                  <Tabs.Trigger
                    key={v.id}
                    value={v.id}
                    className="-mb-px flex items-center gap-1.5 border-b-2 border-transparent px-3 py-1.5 text-[13px] text-ink-dim hover:text-ink data-[state=active]:border-ink data-[state=active]:font-semibold data-[state=active]:text-ink"
                  >
                    {v.icon ? <span className="opacity-80">{v.icon}</span> : null}
                    {v.label}
                  </Tabs.Trigger>
                );
              }
              const reason = v.unavailableReason ?? "Not available for this building";
              return (
                // a disabled button takes no pointer events, so the wrapper carries the tooltip
                <Tooltip.Root key={v.id}>
                  <Tooltip.Trigger asChild>
                    <span data-tab-unavailable={v.id} className="-mb-px flex cursor-not-allowed border-b-2 border-transparent">
                      <Tabs.Trigger
                        value={v.id}
                        disabled
                        aria-label={`${v.label}: not available. ${reason}`}
                        className="pointer-events-none flex items-center gap-1.5 px-3 py-1.5 text-[13px] text-ink-mute opacity-45"
                      >
                        <Ban size={14} aria-hidden />
                        <span className="line-through decoration-ink-mute decoration-[1.5px]">{v.label}</span>
                      </Tabs.Trigger>
                    </span>
                  </Tooltip.Trigger>
                  <Tooltip.Portal>
                    <Tooltip.Content
                      side="bottom"
                      align="start"
                      sideOffset={6}
                      collisionPadding={8}
                      className="z-50 max-w-[280px] rounded border border-line-strong bg-raised px-2.5 py-1.5 text-xs leading-snug text-ink shadow-lg shadow-black/50"
                    >
                      <span className="font-semibold">{v.label} is not available.</span> {reason}
                      <Tooltip.Arrow className="fill-line-strong" />
                    </Tooltip.Content>
                  </Tooltip.Portal>
                </Tooltip.Root>
              );
            })}
          </Tooltip.Provider>
        </Tabs.List>
        {actions ? <div className="ml-auto flex items-center gap-2 pb-1">{actions}</div> : null}
      </div>
      {views.map((v) =>
        visited.has(v.id) && !v.disabled ? (
          <Tabs.Content
            key={v.id}
            value={v.id}
            forceMount
            className="scroll-thin min-h-0 flex-1 overflow-y-auto focus-visible:outline-offset-[-2px] data-[state=inactive]:hidden"
          >
            {v.content}
          </Tabs.Content>
        ) : null,
      )}
    </Tabs.Root>
  );
}
