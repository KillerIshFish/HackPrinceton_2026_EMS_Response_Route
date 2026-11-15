"""Caller transcript -> evidence, facts and the next question to ask.

Entry point: ``parser.interpret.interpret_turn``. Mode "rules" runs the deterministic dialog
rules (``parser.dialog``) and the vocabulary extractor over ``configs/clue_vocab.yaml``
(``parser.extract``), with no network. Modes "llm" / "agent" run the layout-reasoning Claude
reader (``parser.agent``), which falls back to the rules when the model cannot be reached.
``parser.next_question`` ranks the dispatcher's next question by expected information gain.
"""
