import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  icon?: ReactNode;
  /** right-aligned header content (controls, counters) */
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}

/** Flat bordered surface with a compact header - the console's only container. */
export function Panel({ title, icon, actions, className = "", bodyClassName = "", children }: PanelProps) {
  return (
    <section className={`flex min-h-0 flex-col rounded-md border border-line bg-panel ${className}`}>
      <header className="flex min-h-10 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-3 py-1.5">
        <h2 className="flex items-center gap-2 text-[13px] font-semibold tracking-wide text-ink">
          {icon ? <span className="text-ink-mute">{icon}</span> : null}
          {title}
        </h2>
        {actions ? <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
