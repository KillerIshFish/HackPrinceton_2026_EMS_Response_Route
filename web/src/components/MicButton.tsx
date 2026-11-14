import { useEffect, useRef, useState } from "react";
import { Info, Mic, MicOff, Square, TriangleAlert } from "lucide-react";

/**
 * Microphone dictation for the transcript box (BUILD_PLAN step D5).
 *
 * Uses the browser's own speech recognition (Web Speech API, prefixed in Chromium). No server,
 * no key. Interim words go to `onInterim` (shown greyed in the text box); each final result is
 * handed to `onFinal` as one turn for the speaker selected at that moment. The text is passed
 * on exactly as recognised: recognition mistakes are expected downstream, so nothing is
 * corrected or filtered here.
 *
 * Renders as `display: contents`: the button sits in the parent's flex row and the note row
 * (privacy statement and status) takes a full line below it, so the parent row should wrap.
 */

type Speaker = "caller" | "dispatcher";

// --- Web Speech API, declared locally (not every TS DOM lib ships it) -------------------------

interface RecAlternative {
  readonly transcript: string;
  readonly confidence: number;
}
interface RecResult {
  readonly isFinal: boolean;
  readonly length: number;
  readonly [index: number]: RecAlternative;
}
interface RecResultList {
  readonly length: number;
  readonly [index: number]: RecResult;
}
interface RecResultEvent {
  readonly resultIndex: number;
  readonly results: RecResultList;
}
interface RecErrorEvent {
  readonly error: string;
  readonly message?: string;
}
interface Recognizer {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onresult: ((e: RecResultEvent) => void) | null;
  onerror: ((e: RecErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
type RecognizerCtor = new () => Recognizer;

function recognizerCtor(): RecognizerCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { SpeechRecognition?: RecognizerCtor; webkitSpeechRecognition?: RecognizerCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

// --- Messages ---------------------------------------------------------------------------------

type Notice = { tone: "error" | "info"; text: string } | null;

/** Errors that end dictation until the dispatcher presses the button again. */
const FATAL: Record<string, string> = {
  "not-allowed":
    "Microphone blocked: the browser denied access. Allow the microphone for this page (address bar), then press Dictate again. You can keep typing.",
  "service-not-allowed":
    "Speech service not permitted: this browser has speech recognition turned off or blocked here. Type the turn instead.",
  network: "Speech service unreachable (network error). Dictation stopped; type the turn instead.",
  "audio-capture": "No microphone found. Check the headset, then press Dictate again.",
  "language-not-supported": "The speech service does not support this page's language. Type the turn instead.",
};

const NO_SPEECH = "No speech heard yet. Still listening.";

/** More restarts than this inside RESTART_WINDOW_MS means the service keeps dropping: stop. */
const MAX_RESTARTS = 8;
const RESTART_WINDOW_MS = 10_000;

interface Props {
  /** speaker selected in the transcript box; each final result is sent for this speaker */
  speaker: Speaker;
  /** a request is in flight: final results wait in order until it settles */
  busy: boolean;
  onFinal: (text: string, speaker: Speaker) => void;
  /** words recognised so far but not final ("" when there are none) */
  onInterim: (text: string) => void;
}

export function MicButton({ speaker, busy, onFinal, onInterim }: Props) {
  const [supported] = useState(() => recognizerCtor() !== null);
  const [listening, setListening] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [queued, setQueued] = useState<{ text: string; speaker: Speaker }[]>([]);

  const recRef = useRef<Recognizer | null>(null);
  /** the dispatcher wants listening on (drives restart after Chrome's silence timeout) */
  const wantRef = useRef(false);
  const restartsRef = useRef<number[]>([]);
  const speakerRef = useRef(speaker);
  speakerRef.current = speaker;
  const onInterimRef = useRef(onInterim);
  onInterimRef.current = onInterim;
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  function teardown(mode: "stop" | "abort") {
    wantRef.current = false;
    const rec = recRef.current;
    if (!rec) return;
    if (mode === "abort") {
      rec.onresult = rec.onerror = rec.onend = rec.onstart = null;
      recRef.current = null;
      try {
        rec.abort();
      } catch {
        /* already stopped */
      }
    } else {
      // stop() lets the service deliver the words it already heard as a final result; onend clears up
      try {
        rec.stop();
      } catch {
        /* already stopped */
      }
    }
  }

  function start() {
    const Ctor = recognizerCtor();
    if (!Ctor) return;
    teardown("abort");
    const rec = new Ctor();
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.lang = document.documentElement.lang || navigator.language || "en-US";

    rec.onstart = () => {
      if (recRef.current === rec) setListening(true);
    };
    rec.onresult = (e) => {
      if (recRef.current !== rec) return;
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        const text = r?.[0]?.transcript ?? "";
        if (r?.isFinal) {
          const said = text.trim();
          if (said) setQueued((q) => [...q, { text: said, speaker: speakerRef.current }]);
        } else {
          interim += text;
        }
      }
      onInterimRef.current(interim.trim());
      setNotice((n) => (n?.text === NO_SPEECH ? null : n));
    };
    rec.onerror = (e) => {
      if (recRef.current !== rec) return;
      const fatal = FATAL[e.error];
      if (fatal) {
        wantRef.current = false;
        setNotice({ tone: "error", text: fatal });
      } else if (e.error === "no-speech") {
        setNotice({ tone: "info", text: NO_SPEECH });
      } else if (e.error !== "aborted") {
        setNotice({ tone: "info", text: `Speech recognition reported "${e.error}". Still listening.` });
      }
    };
    rec.onend = () => {
      if (recRef.current !== rec) return;
      onInterimRef.current("");
      if (wantRef.current) {
        // Chrome ends a continuous session after a stretch of silence: start it again
        const now = Date.now();
        restartsRef.current = [...restartsRef.current.filter((t) => now - t < RESTART_WINDOW_MS), now];
        if (restartsRef.current.length <= MAX_RESTARTS) {
          try {
            rec.start();
            return;
          } catch {
            /* fall through to stopped */
          }
        }
        setNotice({ tone: "error", text: "Speech recognition keeps stopping. Dictation turned off; type the turn instead." });
      }
      wantRef.current = false;
      recRef.current = null;
      setListening(false);
    };

    recRef.current = rec;
    wantRef.current = true;
    restartsRef.current = [];
    setNotice(null);
    try {
      rec.start();
      setListening(true);
    } catch (err) {
      recRef.current = null;
      wantRef.current = false;
      setListening(false);
      setNotice({ tone: "error", text: `Dictation could not start: ${err instanceof Error ? err.message : String(err)}` });
    }
  }

  function toggle() {
    if (listening || wantRef.current) {
      teardown("stop");
      setListening(false);
    } else {
      start();
    }
  }

  // Stop when the tab is hidden, and for good on unmount.
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === "hidden" && wantRef.current) {
        teardown("stop");
        setListening(false);
        setNotice({ tone: "info", text: "Dictation stopped because the tab was hidden. Press Dictate to resume." });
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      teardown("abort");
    };
  }, []);

  // Hand final results over one at a time, each after the previous request has settled.
  useEffect(() => {
    if (busy || queued.length === 0) return;
    const [next, ...rest] = queued;
    // after the parent's own "request settled" effects have run
    const id = window.setTimeout(() => {
      setQueued(rest);
      if (next) onFinalRef.current(next.text, next.speaker);
    }, 0);
    return () => window.clearTimeout(id);
  }, [busy, queued]);

  if (!supported) return null;

  const who = speaker === "caller" ? "Caller" : "Dispatcher";
  return (
    <div className="contents">
      <button
        type="button"
        onClick={toggle}
        aria-pressed={listening}
        aria-describedby="mic-note"
        title={listening ? "Stop dictation" : "Dictate: speak the turn instead of typing it"}
        className={`flex shrink-0 items-center gap-1 rounded border px-2.5 py-1.5 text-xs font-medium ${
          listening
            ? "border-accent bg-accent/15 text-accent-strong hover:bg-accent/25"
            : "border-line-strong text-ink hover:bg-raised"
        }`}
      >
        {listening ? <Square size={12} aria-hidden /> : <Mic size={13} aria-hidden />}
        {listening ? "Stop" : "Dictate"}
      </button>
      <div id="mic-note" className="order-last basis-full space-y-0.5 text-[10.5px] leading-snug">
        {listening ? (
          <p className="flex items-center gap-1.5 font-medium text-accent-strong" role="status">
            <span className="live-dot h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-hidden />
            Listening: each finished sentence is added as a {who} turn
            {queued.length > 0 ? <span className="num font-normal text-ink-mute">({queued.length} waiting)</span> : null}
          </p>
        ) : null}
        {notice ? (
          <p
            className={`flex items-start gap-1.5 ${notice.tone === "error" ? "text-warn" : "text-ink-dim"}`}
            role={notice.tone === "error" ? "alert" : "status"}
          >
            {notice.tone === "error" ? (
              <TriangleAlert size={12} className="mt-px shrink-0" aria-hidden />
            ) : (
              <MicOff size={12} className="mt-px shrink-0" aria-hidden />
            )}
            {notice.text}
          </p>
        ) : null}
        <p className="flex items-start gap-1.5 text-ink-mute">
          <Info size={12} className="mt-px shrink-0" aria-hidden />
          Dictation: audio is sent to your browser vendor's speech service.
        </p>
      </div>
    </div>
  );
}
