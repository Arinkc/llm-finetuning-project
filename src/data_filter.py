"""
Filter for CodeSearchNet Python examples.

Selects high-quality Google-style docstrings or clean prose docstrings,
rejecting reStructuredText, JavaDoc, and NumPy formats.

Calibrated against a 1000-example sample with 35.5% pass rate.
"""
import re
from typing import Tuple


def passes_filter(row: dict) -> Tuple[bool, str]:
    """Determines if a CodeSearchNet example passes the curation filter.
    
    Args:
        row: A dict-like object with 'func_code_string' and 
             'func_documentation_string' keys.
    
    Returns:
        Tuple of (passes: bool, reason: str). Reason is "ok" if passes,
        otherwise a short description of the rejection cause.
    """
    code = row['func_code_string']
    doc = row['func_documentation_string']
    
    if not code or not doc:
        return False, "missing field"
    
    code_len = len(code)
    doc_len = len(doc)
    
    if code_len < 100: return False, "code too short"
    if code_len > 2000: return False, "code too long"
    if doc_len < 30: return False, "doc too short"
    if doc_len > 500: return False, "doc too long"
    
    words = doc.split()
    has_any_section = bool(re.search(r'\n\s*(?:Args|Returns?|Raises|Yields|Examples?):\s*\n', doc))
    min_words = 5 if has_any_section else 8
    if len(words) < min_words: return False, "too few words in doc"
    
    non_ascii = sum(1 for c in doc if ord(c) > 127)
    if non_ascii / max(len(doc), 1) > 0.05: return False, "too much non-ASCII"
    
    if re.search(r'\b(?:TODO|FIXME|XXX|HACK)\b', doc): 
        return False, "has TODO/FIXME"
    
    stripped = doc.strip()
    if not stripped or not stripped[0].isupper(): 
        return False, "doesn't start with capital"
    
    if re.search(r'\n\s*@param\b', doc) or re.search(r'\n\s*@return\b', doc):
        return False, "JavaDoc style (@param/@return)"
    if re.search(r'\n\s*:param\b', doc) or re.search(r'\n\s*:return\b', doc):
        return False, "reStructuredText style (:param/:return)"
    if re.search(r':rtype:', doc):
        return False, "reStructuredText style (:rtype)"
    if re.search(r':raises?:', doc):
        return False, "reStructuredText style (:raises)"
    if re.search(r'\n\s*Parameters\s*\n\s*-+', doc):
        return False, "NumPy style"
    if re.search(r':py:(?:func|class|meth|mod|attr):', doc):
        return False, "Sphinx RST cross-references"
    
    has_args_section = bool(re.search(r'\n\s*Args:\s*\n', doc))
    has_returns_section = bool(re.search(r'\n\s*Returns?:\s*\n', doc))
    has_raises_section = bool(re.search(r'\n\s*Raises:\s*\n', doc))
    has_google_section = has_args_section or has_returns_section or has_raises_section
    
    lines = stripped.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    is_prose_docstring = (
        len(non_empty_lines) <= 4
        and len(stripped) < 400
        and not stripped.endswith(':')
        and not re.search(r'::\s*$', stripped, re.MULTILINE)
    )
    
    if not (has_google_section or is_prose_docstring):
        return False, "not Google-style and not clean prose"
    
    if has_args_section:
        args_match = re.search(
            r'\n\s*Args:\s*\n(.*?)(?=\n\s*(?:Returns?|Raises|Yields|Examples?):|$)', 
            doc, re.DOTALL
        )
        if args_match and not re.search(r'^\s+\w+:', args_match.group(1), re.MULTILINE):
            return False, "malformed Args section"
    
    return True, "ok"