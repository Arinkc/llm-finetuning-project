# Baseline Model Analysis (Llama 3.1 8B Instruct, no fine-tuning)

## Test Prompt
Document a recursive factorial function. (See `baseline_output.json` for full prompt.)

## Observed Failure Modes

1. **Rewrites the input function** instead of returning only the docstring.
   Users want a docstring they can paste into existing code, not a 
   reimplementation.

2. **Verbose conversational preamble/postamble**: "This docstring follows the 
   guidelines of PEP 257..." — adds 50+ tokens of useless filler.

3. **Output truncation**: Hit the 200-token limit mid-sentence due to 
   verbosity. A more concise output would fit comfortably.

4. **Hallucinated behavior**: Added `ValueError` handling for negative inputs 
   that does not exist in the original function. Risk of misleading users 
   about actual function behavior.

5. **Markdown formatting in output**: Wrapped response in ` ```python ` code 
   fences, requiring manual stripping for any programmatic use.

## Fine-Tuning Targets

The fine-tuned model should produce, for the same input:
- The docstring text only (no surrounding function, no markdown fences)
- No conversational preamble or postamble
- Faithful description of actual code behavior (no hallucinated error handling)
- Concise output that fits comfortably in 100-150 tokens
- Consistent Google-style or NumPy-style format across all examples