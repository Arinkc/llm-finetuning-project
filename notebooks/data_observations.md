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

## Filter Pass Rate (1000-example sample)

| Filter Stage | Pass Rate |
|--------------|-----------|
| No filter | 100% (1000) |
| Length + quality only | 60.3% (603) |
| + Google-style enforcement | 26.5% (265) |

### Rejection Reasons (top 5)
1. reStructuredText style (:param/:return) — 155 examples (15.5%)
2. Not Google-style and not clean one-liner — 141 (14.1%)
3. Too few words in docstring — 139 (13.9%)
4. Docstring too short — 75 (7.5%)
5. Code too long — 72 (7.2%)

### Projected Full Dataset Yield
- 26.5% of 450K ≈ 119,000 examples would pass filter
- Will sample ~20-30K from filtered pool for training (quality over volume)
- Sampling strategy: uniform random with fixed seed for reproducibility