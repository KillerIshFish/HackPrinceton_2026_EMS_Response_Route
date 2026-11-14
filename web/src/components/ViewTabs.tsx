import { useState } from "react";
import type { ReactNode } from "react";
import * as Tabs from "@radix-ui/react-tabs";

export type ViewId = "plan" | "3d" | "scan" | "approach";

export interface ViewDef {
  id: ViewId;
  label: string;
  icon?: ReactNode;
  content: ReactNode;
  /** shown greyed out with `unavailableReason` as its tooltip */
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
 * on every switch.
 */
export function ViewTabs({ views, value, onValueChange, actions }: Props) {
  const usable = views.filter((v) => !v.disabled);
  const active: ViewId = usable.some((v) => v.id === value) ? value : (usable[0]?.id ?? value);
  const [visited, setVisited] = useState<Set<ViewId>>(() => new Set([active]));
  if (!visited.has(active)) setVisited(new Set(visited).add(active));

  return (
    <Tabs.Root
      value={active}
      onValueChange={(v) => onValueChange(v as ViewId)}
      className="flex min-h-0 min-w-0 flex-col gap-2 xl:h-full"
    >
      <div className="flex shrink-0 items-end gap-3 border-b border-line">
        <Tabs.List aria-label="Map views" className="flex">
          {views.map((v) => (
            <Tabs.Trigger
              key={v.id}
              value={v.id}
              disabled={v.disabled}
              title={v.disabled ? v.unavailableReason : undefined}
              className="-mb-px flex items-center gap-1.5 border-b-2 border-transparent px-3 py-1.5 text-[13px] text-ink-dim hover:text-ink disabled:cursor-not-allowed disabled:text-ink-mute/60 data-[state=active]:border-ink data-[state=active]:font-semibold data-[state=active]:text-ink"
            >
              {v.icon ? <span className="opacity-80">{v.icon}</span> : null}
              {v.label}
            </Tabs.Trigger>
          ))}
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
