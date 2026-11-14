# Details

Date : 2026-11-13 23:30:03

Directory c:\\Users\\carco\\HackPrinceton_2026_EMS_Response_Route

Total : 50 files,  6755 codes, 690 comments, 800 blanks, all 8245 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 122 | 0 | 36 | 158 |
| [api/\_\_init\_\_.py](/api/__init__.py) | Python | 0 | 1 | 1 | 2 |
| [api/server.py](/api/server.py) | Python | 220 | 36 | 61 | 317 |
| [configs/clue\_vocab.yaml](/configs/clue_vocab.yaml) | YAML | 101 | 20 | 9 | 130 |
| [configs/demo\_buildings.yaml](/configs/demo_buildings.yaml) | YAML | 107 | 19 | 11 | 137 |
| [configs/priors.yaml](/configs/priors.yaml) | YAML | 73 | 9 | 6 | 88 |
| [core/\_\_init\_\_.py](/core/__init__.py) | Python | 0 | 1 | 1 | 2 |
| [core/building\_graph.py](/core/building_graph.py) | Python | 79 | 15 | 20 | 114 |
| [core/clue\_likelihood.py](/core/clue_likelihood.py) | Python | 120 | 26 | 32 | 178 |
| [core/features.py](/core/features.py) | Python | 56 | 14 | 13 | 83 |
| [core/fusion.py](/core/fusion.py) | Python | 52 | 20 | 16 | 88 |
| [core/phone\_likelihood.py](/core/phone_likelihood.py) | Python | 54 | 13 | 17 | 84 |
| [core/prior.py](/core/prior.py) | Python | 51 | 17 | 19 | 87 |
| [core/schema.py](/core/schema.py) | Python | 68 | 40 | 30 | 138 |
| [core/uncertainty.py](/core/uncertainty.py) | Python | 71 | 14 | 17 | 102 |
| [ingest/\_\_init\_\_.py](/ingest/__init__.py) | Python | 2 | 1 | 3 | 6 |
| [ingest/costar.py](/ingest/costar.py) | Python | 16 | 21 | 12 | 49 |
| [ingest/georegister.py](/ingest/georegister.py) | Python | 35 | 22 | 10 | 67 |
| [ingest/inventory.py](/ingest/inventory.py) | Python | 65 | 13 | 16 | 94 |
| [ingest/matterport.py](/ingest/matterport.py) | Python | 272 | 65 | 56 | 393 |
| [ingest/vexcel.py](/ingest/vexcel.py) | Python | 19 | 19 | 14 | 52 |
| [parser/\_\_init\_\_.py](/parser/__init__.py) | Python | 0 | 6 | 1 | 7 |
| [parser/extract.py](/parser/extract.py) | Python | 197 | 61 | 55 | 313 |
| [parser/next\_question.py](/parser/next_question.py) | Python | 141 | 29 | 24 | 194 |
| [parser/prompts/extract\_clues.md](/parser/prompts/extract_clues.md) | Markdown | 98 | 0 | 25 | 123 |
| [planning/\_\_init\_\_.py](/planning/__init__.py) | Python | 0 | 1 | 1 | 2 |
| [planning/approach.py](/planning/approach.py) | Python | 92 | 16 | 20 | 128 |
| [planning/baselines.py](/planning/baselines.py) | Python | 29 | 20 | 10 | 59 |
| [planning/search\_order.py](/planning/search_order.py) | Python | 323 | 83 | 52 | 458 |
| [pyproject.toml](/pyproject.toml) | TOML | 28 | 0 | 5 | 33 |
| [sim/\_\_init\_\_.py](/sim/__init__.py) | Python | 0 | 1 | 1 | 2 |
| [sim/generate\_calls.py](/sim/generate_calls.py) | Python | 81 | 16 | 10 | 107 |
| [sim/personas.yaml](/sim/personas.yaml) | YAML | 80 | 11 | 9 | 100 |
| [tests/test\_demo\_flow.py](/tests/test_demo_flow.py) | Python | 102 | 8 | 32 | 142 |
| [uv.lock](/uv.lock) | TOML | 1,001 | 0 | 39 | 1,040 |
| [web/index.html](/web/index.html) | HTML | 14 | 0 | 1 | 15 |
| [web/package-lock.json](/web/package-lock.json) | JSON | 1,828 | 0 | 1 | 1,829 |
| [web/package.json](/web/package.json) | JSON | 26 | 0 | 1 | 27 |
| [web/public/favicon.svg](/web/public/favicon.svg) | XML | 8 | 0 | 1 | 9 |
| [web/src/api.ts](/web/src/api.ts) | TypeScript | 50 | 2 | 6 | 58 |
| [web/src/components/FloorPlanHeatmap.tsx](/web/src/components/FloorPlanHeatmap.tsx) | TypeScript JSX | 389 | 7 | 25 | 421 |
| [web/src/components/FloorProbBar.tsx](/web/src/components/FloorProbBar.tsx) | TypeScript JSX | 87 | 3 | 8 | 98 |
| [web/src/components/Panel.tsx](/web/src/components/Panel.tsx) | TypeScript JSX | 23 | 2 | 3 | 28 |
| [web/src/components/TranscriptPanel.tsx](/web/src/components/TranscriptPanel.tsx) | TypeScript JSX | 187 | 2 | 11 | 200 |
| [web/src/index.css](/web/src/index.css) | PostCSS | 92 | 4 | 13 | 109 |
| [web/src/lib.ts](/web/src/lib.ts) | TypeScript | 103 | 23 | 22 | 148 |
| [web/src/main.tsx](/web/src/main.tsx) | TypeScript JSX | 9 | 0 | 2 | 11 |
| [web/src/types.ts](/web/src/types.ts) | TypeScript | 147 | 7 | 19 | 173 |
| [web/tsconfig.json](/web/tsconfig.json) | JSON with Comments | 20 | 0 | 1 | 21 |
| [web/vite.config.ts](/web/vite.config.ts) | TypeScript | 17 | 2 | 2 | 21 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)