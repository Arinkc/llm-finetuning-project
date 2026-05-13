I noticed some data are well written while others are incomplete or have "TODO' written. 
# Data Observations: CodeSearchNet Python Subset

## Sample Used
Random sample of 1000 examples from the `train` split of 
`code-search-net/code_search_net` (Python subset, ~450K examples total).

## Docstring Style Distribution (qualitative)

Inspecting 10 random examples surfaced at least five distinct docstring styles 
coexisting in the dataset:

- **Google-style** (Args:/Returns:/Raises: sections)
- **NumPy-style** (Parameters\n---------- with underline)
- **reStructuredText** (:param:, :return:)
- **JavaDoc-style** (@param, @return — non-idiomatic for Python)
- **Plain prose** (no formal sections, just a description)

This style heterogeneity is the central data quality challenge: without 
filtering, a fine-tuned model would learn to produce a chaotic mix of these.

## Quality Patterns Observed

### Examples I'd want my model to emulate
- `celery_enabled` (developersociety/django-glitter): Clean PEP 257 / Google 
  style. Summary line, blank line, well-formed body. Length appropriate to 
  function complexity.

### Examples that are well-written but wrong style for this project
- `generate_model_name` (ramses-tech/ramses): Clean reStructuredText with 
  `:param:` syntax. Will be filtered out despite being high quality—wrong 
  style for our target output.

### Low-quality examples that demonstrate the filtering need
- `run` (astroduff/commah): 4096-character docstring on an 11K-character 
  function. Comprehensive but far too long for our use case.
- `get_page_content` (varunsrin/one-py): No summary line; dives directly into 
  a numbered enum listing. Bad training signal—teaches the model to skip 
  the summary.
- `walk` (josiah-wolf-oberholtzer/uqbar): Looks well-formatted but parameter 
  descriptions are placeholder text ("foo", "bar"). Demonstrates that simple 
  length/format filters can't catch all garbage.

## Key Issues the Filter Must Address

1. **Style heterogeneity**: 4+ distinct conventions mixed in one dataset.
2. **Length extremes**: Docstrings range from <40 chars to >4000 chars in the 
   sample. Outliers on both ends hurt training.
3. **Garbage markers**: Some docstrings contain TODO, FIXME, or placeholder 
   text (foo/bar) and need to be filtered.
4. **Code-docstring coupling**: `func_code_string` includes the docstring 
   inline, which must be stripped during data formatting (Phase 7 concern).
5. **Non-English content**: A small fraction of docstrings contain non-ASCII 
   characters, suggesting non-English content unsuitable for our task.

## Filter Decision: Docstring Style

Filter to **Google-style docstrings only**, accepting either:
- Docstrings with explicit Google sections (`Args:`, `Returns:`, `Raises:`), or
- Clean prose one-liners (single paragraph, <200 chars, ends with punctuation)

### Reasoning
- Most common modern Python convention (Google, Meta, major OSS projects)
- Heuristically detectable via section markers
- Produces a focused, evaluable before/after comparison
- Standard format for tools like Sphinx napoleon

### Tradeoff Accepted
Filtering to Google-style discards ~70-80% of the dataset, including 
high-quality non-Google examples. Quality and consistency are prioritized 
over volume—the project targets ~10K curated examples from 450K raw, not 
all available data.

## Filter Thresholds (to be confirmed against full-sample distributions)

| Field | Min | Max | Notes |
|-------|-----|-----|-------|
| Code length (chars) | 50 | 2000 | Drops trivial one-liners and outsized functions |
| Docstring length (chars) | 30 | 500 | Drops stubs and walls-of-text |
| Docstring word count | 8 | — | Drops single-phrase stubs |
| Non-ASCII fraction | — | 5% | Drops non-English |

### Style-specific rules
- Reject if matches JavaDoc pattern (`@param`, `@return`)
- Reject if matches reStructuredText pattern (`:param:`, `:return:`)
- Reject if matches NumPy pattern (`Parameters\n----------`)
- Reject if contains Sphinx cross-references (`:py:func:`, etc.)
- Reject if contains TODO / FIXME / XXX / HACK
- Reject if doesn't start with capital letter
- Accept if has Google section markers OR is a clean one-liner


## Filter Calibration Iterations

| Iteration | Pass Rate | Description |
|-----------|-----------|-------------|
| 1: Length + quality only | 60.3% | No style enforcement |
| 2: + Google sections required | 26.5% | Too strict, false negatives on prose |
| 3: + Loosened prose acceptance | 35.5% | Final — accepts Google sections OR short prose |

### Final Filter Behavior
Accepts a docstring if:
- It has Google-style sections (`Args:`, `Returns:`, or `Raises:`) with valid content, OR
- It's clean prose ≤4 lines and <400 chars

Rejects:
- Non-Google styles (JavaDoc `@param`, RST `:param`/`:rtype`/`:raises`, NumPy `Parameters\n---`)
- Sphinx cross-references (`:py:func:`, etc.)
- Length outliers (code <100 or >2000 chars; docs <30 or >500 chars)
- Quality red flags (TODO/FIXME, non-capital start, >5% non-ASCII)
- Stub docstrings (<5-8 words depending on whether sections present)

### Top Rejection Reasons (1000-example sample)
1. reStructuredText style (:param/:return): 155 (15.5%)
2. Too few words in docstring: 139 (13.9%)
3. Docstring too short: 75 (7.5%)
4. Code too long: 72 (7.2%)
5. Doesn't start with capital: 59 (5.9%)

### Projected Full-Dataset Yield
- 35.5% of ~450K ≈ **160,000 examples** would pass the filter
- Training plan: sample **20,000-25,000** examples from filtered pool
- Sampling: uniform random with `random_state=42` for reproducibility