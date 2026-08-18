# Amber: Evaluation of the Coding Agent

*(Drafted by the coding agent from the session record; judgment lines are the group's to edit — marked [group].)*

## Advantages

Speed and scale: within one ~45-minute session the agent segmented 20,813 sentences, produced three independent measures (era-aware dictionary, exhaustive LLM sentence labeling via 8 parallel sub-agents, and an embedding-based concept axis), cross-validated them, and produced the deliverables. Exhaustive labeling of every sentence — normally the cost bottleneck that forces sampling — was trivial. The agent also surfaced substantive knowledge the group would have needed a literature review for: the Comparative Agendas Project boundary rules, the fact that ~40% of pre-1971 "energy" tokens are metaphorical, and that Carter's famous 1979 energy rhetoric was not in the January 1979 SOTU (so the 1979 dip is genuine, not a miss).

## Disadvantages

Transparency shifts from code to prompts: the dictionary is fully inspectable, but the LLM labels depend on a rubric applied by a model whose internal decisions are not auditable, and the embedding axis adds a second opaque layer (a 384-dimensional representation). Reproducibility is weaker than classical code — rerunning label agents can produce slightly different label sets. The methods visibly disagree in the late 1940s–1950s (embedding reads postwar fuels/power-development content higher than keyword methods), and adjudicating who is right still requires a human reading the actual sentences.

## Oversight and trust

Human judgment was required for every consequential definition: the construct boundary (climate counts only when framed through energy systems), the estimand (sentence share), the choice of primary method, and the decision not to adjust any individual year. Verification relied on checks a human specified: inter-method correlations (r = 0.94–0.96), a historical face-validity checklist (1975 maximum, 1980 ≫ 1979, Reagan ≪ Carter, 2009–14 plateau), and spot-reading disagreement sentences. [group] We would not trust unaudited single-method output — particularly the pre-1960 estimates and any borderline construct calls — without reading samples of the classified sentences ourselves.

## Overall assessment

[group draft] We would use a coding agent again for: corpus mechanics (segmentation, pipelines), exhaustive first-pass annotation, generating competing measurement designs, and stress-testing a measure against historical priors. We would keep humans in charge of: the construct definition, the choice among diverging methods, and final validation — the agent made the measurement multiverse cheap to explore, but choosing a defensible branch within it remained the group's job.
