#!/usr/bin/env bash
#
# run_docfinqa.part_2.sh — build `workspace_docfinqa_part2` from the NEXT 15
# DocFinQA dev filings: 123 questions across documents no run has touched.
#
# Part 1 (run_docfinqa.sh) covers data/dev.formats.top15.json — 15 filings /
# 122 questions — which is every document the token_quality experiment has
# measured so far. This takes the 15 highest-question filings from the 18 in
# top30 that part 1 leaves out, giving a second set of near-identical size.
#
# Two things differ from part 1, and they are the whole reason this script
# exists rather than a flag:
#
#   1. The manifest does not exist yet. top30 records a different shape
#      (`n_q`, `year`, and a `pdf` object of render statistics, not a path),
#      so scripts/make_part2_manifest.py synthesises the `.[].paths.pdf` and
#      `.pdf_pages` fields run_docfinqa.sh reads.
#   2. The PDFs do not exist yet. data/pdf/dev holds only part 1's 15, so each
#      filing is fetched from EDGAR and rendered by scripts/build_formats.py
#      before any of the DGML pipeline runs.
#
# Everything after that is part 1 unchanged: this execs run_docfinqa.sh with a
# different manifest, workspace and docset, so the two sets are built by
# exactly the same pipeline and stay comparable. Fixes to the pipeline are
# inherited rather than duplicated.
#
# Every step is idempotent: the manifest and any PDF already on disk are left
# alone, and run_docfinqa.sh skips files it has already ingested or generated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TQ_ROOT="${TQ_ROOT:-/Users/gregory/EXPE/Docugami/token_quality}"
MANIFEST="${MANIFEST:-$TQ_ROOT/data/dev.formats.part2.json}"
PDF_DIR="${PDF_DIR:-$TQ_ROOT/data/pdf/dev}"
WORKSPACE="${DGML_WORKSPACE:-$REPO_ROOT/workspace_docfinqa_part2}"
DOCSET_NAME="${DOCSET_NAME:-DocFinQA Dev Part 2}"
N_FILINGS="${N_FILINGS:-15}"
# Ladder: haiku transcribe + haiku label -> haiku retry -> sonnet on the short
# pages only.
# Part 1 has NO ladder -- it sets no retry_model, so a page that transcribes
# badly is kept as-is, which is the likely origin of the 12 conversion gaps in
# token_quality/DGML_MISSING_PARTS.md (AON's two segment tables among them).
# Setting retry_model AT ALL is what arms the per-page trigger, even when it
# names the same model as `standard`.
MODEL="${MODEL:-anthropic/claude-haiku-4-5}"
# Labeling on haiku too, so haiku does BOTH passes and sonnet appears only as
# the rescue. Every workspace built so far put labeling on sonnet, so part 2
# diverges here deliberately -- labeling is what produces the semantic tag
# names, which is the thing DGML is being measured on.
LABEL_MODEL="${LABEL_MODEL:-anthropic/claude-haiku-4-5}"
RETRY_MODEL="${RETRY_MODEL:-anthropic/claude-sonnet-5}"
ESCALATE=1
SKIP_FETCH=0
ONLY=""
DRY_RUN=0
PASSTHROUGH=()

usage() {
    cat <<'USAGE'
Usage: ./run_docfinqa.part_2.sh [options] [-- <extra run_docfinqa.sh args>]

  --n N              How many of the unused filings to take, most questions
                     first (default: 15 -> 123 questions)
  --manifest FILE    Manifest to build and use
                     (default: $TQ_ROOT/data/dev.formats.part2.json)
  --workspace DIR    Workspace root  (default: ./workspace_docfinqa_part2)
  --docset-name NAME DocSet name     (default: "DocFinQA Dev Part 2")
  --model M          Transcription model   (default: anthropic/claude-haiku-4-5)
  --label-model M    Labeling model        (default: anthropic/claude-haiku-4-5)
  --retry-model M    Escalation model for short pages; setting it is what arms
                     the per-page rescue (default: anthropic/claude-sonnet-5)
  --no-escalate      Write no retry_model, exactly like part 1. Use this if the
                     two sets must be built by an identical pipeline -- with the
                     ladder on, a part-1-vs-part-2 difference confounds "new
                     documents" with "better conversion".
  --only SUBSTR      Restrict the whole run -- fetch AND generate -- to the
                     filings whose key contains SUBSTR (e.g. --only UNP_2011).
                     Writes a filtered manifest beside the full one, so the
                     first test costs one document rather than fifteen.
                     run_docfinqa.sh's own --limit N only narrows generation,
                     after all 15 PDFs have already been fetched.
  --skip-fetch       Assume the PDFs are already rendered; do not call EDGAR
  --dry-run          Build the manifest, report what is missing, and stop
                     before fetching or spending anything on LLM calls
  -y, --yes          Don't prompt before the generate step (forwarded)
  -h, --help         This message

Anything after `--` is passed straight to run_docfinqa.sh, so its own flags
still apply (--limit, --dpi, --window-size, --skip-generate, -y, ...).
Start with `--dry-run`, then `-- --limit 2` for a cheap first real run.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)           N_FILINGS="$2"; shift 2 ;;
        --manifest)    MANIFEST="$2"; shift 2 ;;
        --workspace)   WORKSPACE="$2"; shift 2 ;;
        --docset-name) DOCSET_NAME="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --label-model) LABEL_MODEL="$2"; shift 2 ;;
        --retry-model) RETRY_MODEL="$2"; shift 2 ;;
        --no-escalate) ESCALATE=0; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        --skip-fetch)  SKIP_FETCH=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -y|--yes)      PASSTHROUGH+=(-y); shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; PASSTHROUGH=("$@"); break ;;
        *)             echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

BOLD=$'\033[1m'; RED=$'\033[31m'; RESET=$'\033[0m'
[[ -t 2 ]] || { BOLD=""; RED=""; RESET=""; }
step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET" >&2; }
info() { printf '    %s\n' "$*" >&2; }
die()  { printf '\n%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

command -v jq >/dev/null 2>&1 || die "\`jq\` not on PATH"
# litellm reads ANTHROPIC_API_KEY; this machine keeps it as ANTHROPIC_KEY.
# run_docfinqa.sh bridges it too, but that runs after our own dgml calls.
if [[ -z "${ANTHROPIC_API_KEY:-}" && -n "${ANTHROPIC_KEY:-}" ]]; then
    export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
fi
[[ -d "$TQ_ROOT" ]] || die "token_quality not found: $TQ_ROOT"
[[ -x "$REPO_ROOT/run_docfinqa.sh" ]] || die "run_docfinqa.sh not found next to this script"

# ---------------------------------------------------------------------------
step "Manifest"
# ---------------------------------------------------------------------------
if [[ -f "$MANIFEST" ]]; then
    info "already built: $MANIFEST"
else
    python3 "$TQ_ROOT/scripts/make_part2_manifest.py" "$N_FILINGS" >&2
fi
jq -e 'length > 0' "$MANIFEST" >/dev/null || die "empty manifest: $MANIFEST"
if [[ -n "$ONLY" ]]; then
    FILTERED="${MANIFEST%.json}.only-${ONLY}.json"
    jq --arg s "$ONLY" '[.[] | select(.key | contains($s))]' "$MANIFEST" > "$FILTERED"
    jq -e 'length > 0' "$FILTERED" >/dev/null \
        || die "--only '$ONLY' matched nothing in $(basename "$MANIFEST")"
    MANIFEST="$FILTERED"
    info "--only '$ONLY' -> $(jq -r '[.[].key] | join(", ")' "$MANIFEST")"
fi
info "$(jq -r 'length' "$MANIFEST") filings, $(jq -r '[.[].n_questions] | add' "$MANIFEST") questions, $(jq -r '[.[].pdf_pages] | add' "$MANIFEST") pages"

# ---------------------------------------------------------------------------
step "Source PDFs"
# ---------------------------------------------------------------------------
# build_formats.py filters with a comma-separated TICKER_YEAR list, while the
# manifest is keyed TICKER_YEAR_ACCESSION -- so the filter is derived, not the
# key itself.
MISSING_KEYS=(); MISSING_FILTERS=()
while IFS=$'\t' read -r key ticker fy; do
    [[ -f "$PDF_DIR/$key.pdf" ]] && continue
    MISSING_KEYS+=("$key"); MISSING_FILTERS+=("${ticker}_${fy}")
done < <(jq -r '.[] | [.key, .ticker, .fiscal_year] | @tsv' "$MANIFEST")

if (( ${#MISSING_KEYS[@]} == 0 )); then
    info "all PDFs present in $PDF_DIR"
elif (( SKIP_FETCH )); then
    die "${#MISSING_KEYS[@]} PDF(s) missing and --skip-fetch was given"
else
    info "${#MISSING_KEYS[@]} PDF(s) missing; fetching from EDGAR and rendering"
    for k in "${MISSING_KEYS[@]}"; do info "  $k"; done
    if (( DRY_RUN )); then
        info "(dry run: not fetching)"
    else
        FILTER=$(IFS=,; echo "${MISSING_FILTERS[*]}")
        ( cd "$TQ_ROOT" && python3 scripts/build_formats.py --split dev --keys "$FILTER" ) \
            || die "build_formats.py failed; the DGML run was not started"
        still=0
        for k in "${MISSING_KEYS[@]}"; do [[ -f "$PDF_DIR/$k.pdf" ]] || { info "  STILL MISSING: $k"; still=$((still+1)); }; done
        (( still == 0 )) || die "$still PDF(s) could not be rendered"
    fi
fi

# ---------------------------------------------------------------------------
step "Workspace + transcription ladder"
# ---------------------------------------------------------------------------
# Done HERE rather than left to run_docfinqa.sh because that script writes a
# config.toml with no retry_model and then leaves any existing one untouched --
# so the ladder has to be on disk before it runs. Creating the workspace the
# same way it does keeps its own "already initialized" path working.
if (( DRY_RUN )); then
    info "dry run: workspace not created"
    (( ESCALATE )) && info "would arm: retry_model = $RETRY_MODEL" || info "ladder OFF (--no-escalate)"
else
    command -v uv >/dev/null 2>&1 || die "\`uv\` not on PATH"
    if [[ -f "$WORKSPACE/workspace.json" ]]; then
        info "workspace already initialized"
    else
        mkdir -p "$WORKSPACE"
        ( cd "$REPO_ROOT" && env -u VIRTUAL_ENV uv run --quiet dgml workspace create "$WORKSPACE" \
            --organization Docugami --name "$(basename "$WORKSPACE")" ) >/dev/null
        info "created $WORKSPACE"
    fi
    # `dgml workspace create` writes its own config.toml ([storage], [workspace]),
    # so "write only if absent" would silently skip the ladder. Merge instead:
    # add [models] and retry_model only where they are not already set, leaving
    # the storage fingerprint and anything hand-edited alone.
    if (( ESCALATE )); then
        python3 - "$WORKSPACE/config.toml" "$MODEL" "$LABEL_MODEL" "$RETRY_MODEL" <<'PYCFG'
import re, sys
path, model, label, retry = sys.argv[1:5]
try:
    txt = open(path).read()
except FileNotFoundError:
    txt = ''
add = []
if not re.search(r'(?m)^\[models\]', txt):
    add.append('\n[models]\n'
               'light    = "anthropic/claude-haiku-4-5"\n'
               f'standard = "{model}"        # per-page transcription\n'
               f'advanced = "{label}"        # labeling, semlinks, value extraction\n'
               'expert   = "anthropic/claude-opus-5"\n')
if not re.search(r'(?m)^\s*retry_model\s*=', txt):
    if re.search(r'(?m)^\[generation\]', txt):
        txt = re.sub(r'(?m)^(\[generation\]\n)',
                     r'\1retry_model = "%s"\n' % retry, txt, count=1)
    else:
        add.append('\n[generation]\n'
                   '# Setting retry_model is what turns on the per-page segmented rescue,\n'
                   '# even when it names the same model as `standard`.\n'
                   f'retry_model = "{retry}"\n')
open(path, 'w').write(txt + ''.join(add))
PYCFG
        info "merged the ladder into $WORKSPACE/config.toml"
        grep -q '^retry_model' "$WORKSPACE/config.toml" \
            || die "config.toml sets no retry_model — the per-page trigger would be off"
        grep -E '^(standard|advanced|retry_model)' "$WORKSPACE/config.toml" | sed 's/^/    /' >&2
    else
        info "--no-escalate: no retry_model written"
    fi
fi

# ---------------------------------------------------------------------------
step "Generate"
# ---------------------------------------------------------------------------
if (( DRY_RUN )); then
    info "dry run: stopping before run_docfinqa.sh"
    info "would run: ./run_docfinqa.sh --manifest $MANIFEST --workspace $WORKSPACE --docset-name '$DOCSET_NAME' ${PASSTHROUGH[*]:-}"
    exit 0
fi
exec "$REPO_ROOT/run_docfinqa.sh" \
    --manifest "$MANIFEST" \
    --workspace "$WORKSPACE" \
    --docset-name "$DOCSET_NAME" \
    "${PASSTHROUGH[@]}"
