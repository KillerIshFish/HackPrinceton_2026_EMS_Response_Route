import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../api";
import type { PropertyHit } from "../types";

interface Props {
  disabled: boolean;
  /** true while a model is being generated for the chosen property */
  generating: boolean;
  onChoose: (hit: PropertyHit) => void;
}

/**
 * Address / building-name / property-id search over the property index. Choosing a result
 * generates that building's model now and starts a call on it.
 */
export function BuildingSearch({ disabled, generating, onChoose }: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<PropertyHit[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  // Look up as the dispatcher types (the index answers in about a millisecond). Nothing under 3
  // characters (too broad to be useful), and a 120 ms debounce so a burst of keystrokes sends one
  // request; `cancelled` drops answers to a query that has since changed. Both values are tuned.
  useEffect(() => {
    const q = query.trim();
    if (q.length < 3) {
      setHits([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .lookup(q)
        .then((h) => {
          if (cancelled) return;
          setHits(h);
          setError(null);
          setOpen(true);
        })
        .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query]);

  // close the result list on a click anywhere outside the search box
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  function choose(hit: PropertyHit) {
    setOpen(false);
    setQuery(hit.address);
    onChoose(hit);
  }

  return (
    <div ref={boxRef} className="relative w-[250px] max-w-full">
      <form
        onSubmit={(e) => {
          // Enter takes the first hit in the order /lookup returned them (8 results by default)
          e.preventDefault();
          if (hits[0]) choose(hits[0]);
        }}
        className="flex items-center gap-1.5 rounded border border-line-strong bg-bg px-2 py-1 focus-within:border-accent"
      >
        <Search size={13} className="shrink-0 text-ink-mute" aria-hidden />
        <input
          aria-label="Find a building by address"
          value={query}
          disabled={disabled}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => hits.length > 0 && setOpen(true)}
          placeholder="Address, building name or property ID"
          className="min-w-0 flex-1 bg-transparent text-xs text-ink placeholder:text-ink-mute focus:outline-none"
        />
        {generating ? <span className="shrink-0 text-[11px] text-prob">Generating…</span> : null}
      </form>
      {open && (hits.length > 0 || error) ? (
        <ul
          role="listbox"
          aria-label="Matching properties"
          className="absolute left-0 right-0 top-full z-30 mt-1 max-h-[320px] overflow-auto rounded border border-line-strong bg-raised py-1 shadow-xl"
        >
          {error ? <li className="px-2.5 py-1.5 text-xs text-alert">{error}</li> : null}
          {hits.map((h) => (
            <li key={h.property_id}>
              <button
                role="option"
                aria-selected={false}
                onClick={() => choose(h)}
                className="block w-full px-2.5 py-1.5 text-left hover:bg-panel"
              >
                <span className="block truncate text-xs font-medium text-ink">
                  {h.address}
                  {h.city ? `, ${h.city}` : ""}
                </span>
                <span className="block truncate text-[11px] text-ink-mute">
                  {h.name && h.name !== h.address ? `${h.name} · ` : ""}
                  {h.stories ? `${h.stories} storeys` : "storeys unknown"}
                  {h.units ? ` · ${h.units} units` : ""} · ID {h.property_id}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
