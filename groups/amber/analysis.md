# Amber: Analytical Approach

## Definition

Energy attention = the share of a speech's sentences whose primary topic is energy as a fuel-and-power system: supply, production, prices, distribution, security, imports/exports, infrastructure (pipelines, grids, refineries, power plants), conservation/efficiency of energy, civilian nuclear/atomic power, fossil fuels, renewables, clean-energy technology, and energy legislation. Excluded: nuclear weapons and arms control, metaphorical "energy" ("the energy of our people"), generic economy/inflation mentions where fuel is one listed item, environment/climate content that does not name fuels, power, or energy technology, and oil-state geopolitics without an explicit energy stake. Climate content counts only when framed through energy systems, so the construct is applied identically to 1950s "Atoms for Peace" talk and 2010s clean-energy talk.

## Measurement

Each speech was split into sentences (20,813 across 75 documents). All sentences were embedded with `sentence-transformers/all-MiniLM-L6-v2` (normalized, 384-d). A TCAV-style concept activation vector was learned: coding agents labeled every sentence energy/not-energy under the definition above; an L2-regularized logistic regression (balanced classes, positives + 4× random negatives, 75/25 split) was trained on the embeddings, and its unit-normalized weight vector is the "energy axis" (held-out accuracy 0.92, F1 0.87). Every sentence is projected onto this axis; a threshold τ = 0.202 (best held-out F1) converts projections to binary calls; the annual estimate is the share of sentences above τ — an unsmoothed 0–1 sentence share robust to the 5× variation in address length (the 1946 and 1973 documents are written messages, not speeches).

## Agent workflow

The group's coding agent (Claude Code) first ran a five-agent panel from different disciplines (political science, computational linguistics, statistics, ML engineering, energy history) to converge on the construct and design, then: (1) segmented the corpus and built an era-aware keyword dictionary with disambiguation rules (energy-NP rule, weapons exclusion, metaphor filters); (2) fanned out 8 parallel labeling agents, each labeling one era of sentences under a fixed written codebook (the definition above plus explicit trap rules: nuclear weapons = 0, "energy of our people" = 0, climate-without-energy = 0, passing mentions = 0); (3) embedded the sentences and trained the CAV on those labels. Group members intervened to: choose the construct boundary (energy-systems framing of climate), pivot the primary method to the embedding axis, choose the compute host, and cancel/relaunch one labeling batch. All estimates come from the pipeline; no per-year manual adjustments were made.

## Checks

Three independent series were compared: embedding-axis, dictionary, and raw LLM-label shares — Pearson r = 0.94 (dictionary) and 0.96 (LLM labels), Spearman 0.83/0.82. Face-validity checks all pass: 1975 is the maximum (oil-crisis aftermath), 1980 ≫ 1979 (Carter's 1979 energy rhetoric was not in the SOTU), Reagan mean ≪ Carter mean, a 2009–2014 clean-energy plateau, and a near-zero 1960s floor.

## Important choices

(1) Sentence share as the estimand — proportions, not counts, make written and spoken addresses comparable. (2) The energy-systems boundary for climate content — a broader "energy + climate" definition would roughly double post-2009 estimates. (3) Training the axis on agent labels rather than hand-picked anchor words — this makes the measure context-aware but inherits any systematic labeling bias. (4) Threshold τ by best F1 — a lower τ raises all levels, mostly uniformly.

## Limitations

The measure disagrees most with keyword methods before 1960: the embedding axis scores postwar fuels-management and power-development content (dams, coal supply, atomic power) higher than keyword approaches, so late-1940s estimates (0.03–0.09) are the most uncertain. 1979's estimate of 0.000 reflects the actual January 1979 SOTU text, not missing data. Sentence segmentation of the two long written messages is imperfect. No human gold-standard coding was done within the session's time budget; validation rests on inter-method agreement and historical face validity.
