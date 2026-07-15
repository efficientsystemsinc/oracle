# AGENTS.md

<!-- efficient-systems:research-standard:start -->
## Research communication and reproducibility standard

This standard applies to humans and agents. Repository-specific rules remain in force; follow the stricter rule when they differ.

### Evidence and source traceability

- Label substantive claims as `OBSERVED`, `INFERRED`, `HYPOTHESIZED`, `PROPOSED`, or `UNKNOWN`. Never turn an inference into an observation during summarization.
- Support nontrivial external claims with a source you actually opened. Prefer primary sources; give a stable URL or paper identifier and the relevant section, page, table, or figure when possible.
- Support code claims with a commit and file/symbol reference. Support experimental claims with a run ID or immutable artifact path.
- Never invent citations, measurements, commands, test results, or implementation details. If provenance is unavailable, write `UNKNOWN — source not verified` and stop the claim from propagating.
- Keep negative results, failed runs, exclusions, and conflicting evidence visible. Do not select only favorable outcomes.

### Reproducibility block for result-bearing work

Every PR or report that introduces or changes an experimental result must record:

- the question or hypothesis;
- dataset/task set, version or commit, split, exclusions, and sample count;
- source commit plus exact command, configuration, dependencies, relevant environment/hardware, model/provider/version, prompt or protocol version, and all seeds;
- artifact paths or run IDs for raw outputs and logs, with checksums when artifacts can move;
- metric and judge definitions, comparison baseline, and the exact verification command;
- known limitations, failed runs, and expected sources of variance.

Redact secrets, but do not omit the non-secret configuration required to reproduce the work. If a required item is unavailable, mark it `UNKNOWN` rather than guessing.

### Claims, validation, and writing

- PR descriptions must state what changed, why, the supporting evidence, risks, and validation. Keep the summary concise; link artifacts instead of pasting model transcripts or status diaries.
- Do not write “tests pass,” “verified,” or an equivalent unless you ran the stated command against the current change. Record the exact command and outcome. Otherwise write `NOT RUN` with the reason.
- Use calibrated language. Terms such as “proves,” “state of the art,” “robust,” “production-ready,” “comprehensive,” and causal claims require scoped evidence and an explicit comparison.
- Do not hide uncertainty behind polished prose. Distinguish results from interpretation and proposals from completed work.
- Do not include chain-of-thought, raw agent chatter, repeated conclusions, filler, or promotional language.

Use the Feynman technique for every technical explanation: state the problem and result in plain language, define specialized terms, give one concrete example or counterexample, explain the mechanism before the abstraction, then add exact technical detail and limitations. Plain language must not erase precision or uncertainty.

### Academic workflow and independent review

For research questions, papers, benchmarks, evaluations, and result-bearing reports, use the smallest applicable Academic Research Suite workflow when it is installed. Use Socratic scoping when the research question is vague, the experiment workflow for design or reproducibility validation, and the integrity/reviewer workflow for claim-source alignment and adversarial review. Do not claim ARS coverage unless that workflow was actually loaded and run.

After an authoring pass, run a fresh read-only empirical review. The reviewer must inspect raw artifacts instead of trusting the PR narrative, preregister its acceptance criteria before judging results, and test the strongest plausible alternative explanation. This is procedural independence, not statistical or institutional independence.

For empirical work, reviewers must check leakage/contamination, sample construction and exclusions, baseline parity, effect sizes and uncertainty, multiple comparisons, stopping rules, selection/survivorship bias, confounds, failed or negative runs, and whether the conclusion exceeds the evaluated setting.

### Review guidelines

- Correct false or stale claims conspicuously. Name what is retracted, provide the replacement, and update every headline or summary that depended on it.
- Reviewers must request changes when a substantive claim lacks traceable evidence, a result cannot be reproduced from the recorded information, validation is overstated, or the description is dominated by unedited agent output.
- Do not approve or merge result-bearing work while required evidence or reproduction artifacts are missing. `UNKNOWN` is acceptable for work in progress, not for a claimed conclusion.
- Report a finding only when you can name the affected file/line or artifact, the evidence, the consequence, a reproduction or verification path, and your confidence. Do not manufacture issues to fill a quota; formatting preferences are not bugs.
- Do not put secrets, credentials, private keys, tokens, raw participant data, customer data, or unpublished private corpora in prompts, PR bodies, logs, fixtures, artifacts, or commits. Use secret names and unmistakably redacted placeholders.
<!-- efficient-systems:research-standard:end -->
