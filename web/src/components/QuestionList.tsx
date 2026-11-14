import { useEffect, useState } from "react";
import { Info, MessageCircleQuestion } from "lucide-react";
import type { Question } from "../types";

interface Props {
  /** state.questions: the top three, in the server's order */
  questions: Question[];
  /** active threat: the console does not prompt caller contact */
  activeThreat: boolean;
  /** a request is in flight */
  busy: string | null;
  /** log the question as a dispatcher turn (POST /calls/demo/ask) */
  onAsk: (q: Question) => void;
}

/**
 * The questions most likely to narrow the search. "Ask" logs the question as a dispatcher turn;
 * a yes/no question then gets Yes / No / Not sure buttons under that turn in the transcript.
 */
export function QuestionList({ questions, activeThreat, busy, onAsk }: Props) {
  const [askedId, setAskedId] = useState<string | null>(null);
  useEffect(() => {
    if (busy !== "ask") setAskedId(null);
  }, [busy]);

  const best = questions.reduce<Question | null>(
    (m, q) => (m === null || q.expected_info_gain_bits > m.expected_info_gain_bits ? q : m),
    null,
  );

  return (
    <section aria-label="Questions to ask">
      {questions.length === 0 ? (
        <p className="flex items-start gap-1.5 text-[12px] text-ink-dim" role="status">
          <Info size={13} className="mt-[2px] shrink-0 text-ink-mute" aria-hidden />
          {activeThreat
            ? "The console does not prompt contact with callers during an active threat."
            : "No further question would narrow the search."}
        </p>
      ) : (
        <ol className="space-y-1">
          {questions.map((q) => {
            const yesNo = q.kind === "yes_no";
            const asking = busy === "ask" && askedId === q.id;
            return (
              <li
                key={q.id}
                className="flex items-center gap-2 rounded border border-line bg-raised/60 py-1 pl-2 pr-1"
                title={`Expected information gain ${q.expected_info_gain_bits.toFixed(2)} bits${
                  q.p_yes !== null ? `; chance of a yes about ${Math.round(q.p_yes * 100)}%` : ""
                }`}
              >
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] leading-snug text-ink">“{q.question}”</p>
                  <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[10.5px] text-ink-mute">
                    <span className="inline-flex items-center gap-1">
                      {yesNo ? null : <MessageCircleQuestion size={11} aria-hidden />}
                      {yesNo ? "Yes / No" : "Open answer: type what they say"}
                    </span>
                    {best && best.id === q.id && questions.length > 1 ? (
                      <span className="font-semibold uppercase tracking-wide text-ink-dim">Narrows most</span>
                    ) : null}
                  </p>
                </div>
                <button
                  type="button"
                  disabled={busy !== null}
                  onClick={() => {
                    setAskedId(q.id);
                    onAsk(q);
                  }}
                  className="flex h-7 shrink-0 items-center rounded bg-ink px-2.5 text-xs font-semibold text-bg hover:bg-white disabled:cursor-not-allowed disabled:bg-raised disabled:text-ink-mute"
                >
                  {asking ? "Asking…" : "Ask"}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
