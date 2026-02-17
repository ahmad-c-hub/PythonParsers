"""
map_form_fields.py

Reads output_with_forms.cleaned.md (pages with Form Fields + Content sections),
finds where each field label appears in the page content text using fuzzy
token matching with one-way synonym expansion, and injects the field value
inline right after the matching span.

Output: output_with_forms.mapped.md

Usage:
    python map_form_fields.py [input.md [output.md]]
"""

import re
import sys
import os


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

PAGE_SEP_RE    = re.compile(r'^---\s*$')
PAGE_HEADER_RE = re.compile(r'^## Page \d+\s*$')
FIELDS_HDR_RE  = re.compile(r'^### Form Fields')
CONTENT_HDR_RE = re.compile(r'^### Content\s*$')
FIELD_LINE_RE  = re.compile(r'^\s*-\s+\*\*(.+?)\*\*:\s*(.*)$')


def parse_pages(text):
    lines = text.splitlines(keepends=True)
    pages = []
    i = 0

    while i < len(lines):
        raw  = lines[i]
        line = raw.rstrip('\n')

        if PAGE_SEP_RE.match(line) or PAGE_HEADER_RE.match(line):
            page = {
                'header_lines':   [],
                'before_fields':  [],
                'fields_header':  '',
                'fields':         [],
                'content_header': '',
                'content_lines':  [],
            }

            if PAGE_SEP_RE.match(line):
                page['header_lines'].append(raw)
                i += 1

            if i < len(lines) and PAGE_HEADER_RE.match(lines[i].rstrip('\n')):
                page['header_lines'].append(lines[i])
                i += 1

            section = 'before_fields'

            while i < len(lines):
                raw  = lines[i]
                line = raw.rstrip('\n')

                if PAGE_SEP_RE.match(line) or PAGE_HEADER_RE.match(line):
                    break

                if FIELDS_HDR_RE.match(line):
                    page['fields_header'] = raw
                    section = 'fields'
                    i += 1
                    continue

                if CONTENT_HDR_RE.match(line):
                    page['content_header'] = raw
                    section = 'content'
                    i += 1
                    continue

                if section == 'before_fields':
                    page['before_fields'].append(raw)
                elif section == 'fields':
                    m = FIELD_LINE_RE.match(line)
                    if m:
                        page['fields'].append({
                            'label': m.group(1).strip(),
                            'value': m.group(2).strip(),
                            'raw':   raw,
                        })
                    else:
                        page['fields'].append({'label': None, 'value': None, 'raw': raw})
                elif section == 'content':
                    page['content_lines'].append(raw)

                i += 1

            pages.append(page)
        else:
            i += 1

    return pages


# ---------------------------------------------------------------------------
# Fuzzy token matching with ONE-WAY synonym expansion
# ---------------------------------------------------------------------------

STOPWORDS = {'a', 'an', 'the', 'of', 'or', 'and', 'in', 'to', 'for',
             'at', 'by', 'on', 'is', 'it', 'if', 'any'}

# Map label token → frozenset of acceptable content tokens
# Intentionally NOT symmetric: 'social' alone doesn't trigger 'ssn' label.
LABEL_SYNONYMS = {
    'phone':    frozenset({'phone', 'telephone', 'tel'}),
    'ssn':      frozenset({'ssn', 'social', 'security'}),
    'org':      frozenset({'org', 'organization', 'organisation'}),
    'addr':     frozenset({'addr', 'address'}),
    'num':      frozenset({'num', 'number'}),
    'flr':      frozenset({'flr', 'floor'}),
    'ste':      frozenset({'ste', 'suite'}),
    'apt':      frozenset({'apt', 'apartment'}),
    'zip':      frozenset({'zip', 'zipcode'}),
    'postal':   frozenset({'postal'}),          # only 'postal', NOT 'zip'
}

# Explicit multi-token expansions for abbreviation-only labels.
# These override the single-token tokenization entirely.
LABEL_OVERRIDES = {
    'ssn': [frozenset({'social', 'ssn'}), frozenset({'security'}), frozenset({'number', 'num'})],
}


def tokenize_plain(text):
    """Tokenize content text — just lowercase alpha, stopwords removed."""
    return [w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOPWORDS]


def tokenize_label(text):
    """
    Tokenize a label into a list of frozensets (one per token).
    Each frozenset is the set of content words that satisfy that label token.
    """
    raw = [w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOPWORDS]
    # Check for whole-label override first
    key = ' '.join(raw)
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    # Single-token override
    if len(raw) == 1 and raw[0] in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[raw[0]]
    return [LABEL_SYNONYMS.get(w, frozenset({w})) for w in raw]


def overlap_with_synonyms(label_sets, content_toks):
    """
    Score how well content_toks satisfies label_sets.
    Coverage  = fraction of label positions matched (weighted 0.6)
    Precision = fraction of content tokens that matched (weighted 0.4)
    """
    if not label_sets or not content_toks:
        return 0.0
    cs      = set(content_toks)
    matched = sum(1 for ls in label_sets if ls & cs)
    coverage  = matched / len(label_sets)
    precision = matched / len(content_toks)
    return 0.6 * coverage + 0.4 * precision


def to_plain(line):
    """Strip markdown markers while preserving character positions."""
    plain = re.sub(r'\*\*|__', lambda m: ' ' * len(m.group()), line)
    plain = re.sub(
        r'\[([^\]]+)\]\([^)]+\)',
        lambda m: m.group(1) + ' ' * (len(m.group(0)) - len(m.group(1))),
        plain,
    )
    return plain.replace('`', ' ')


def score_line(label_sets, line, min_score=0.55):
    """
    Slide a word window over the line and return (score, start_char, end_char)
    for the best-matching span, or None.
    """
    plain = to_plain(line)
    words = list(re.finditer(r'\S+', plain))
    if not words:
        return None

    n          = len(label_sets)
    best_score = min_score
    best_span  = None

    for window in range(max(1, n - 1), n + 5):
        for si in range(len(words) - window + 1):
            ei        = si + window - 1
            span_text = plain[words[si].start(): words[ei].end()]
            ctoks     = tokenize_plain(span_text)
            score     = overlap_with_synonyms(label_sets, ctoks)
            if score > best_score:
                best_score = score
                best_span  = (best_score, words[si].start(), words[ei].end())

    return best_span


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def fmt_value(value):
    v = value.strip()
    if not v or v.lower() == 'n/a':
        return '`= N/A`'
    if v == '_(empty)_':
        return '`= (empty)`'
    return f'`= {v}`'


# ---------------------------------------------------------------------------
# Lines we never annotate
# ---------------------------------------------------------------------------

def is_skip_line(line):
    s = line.strip()
    if not s:                                         return True
    if s.startswith('#') or s.startswith('>'):        return True
    if re.match(r'^\|[-|:\s]+\|$', s):               return True
    if s.startswith('-----') or s == '---':          return True
    # Skip lines that are purely a markdown link (e.g. ZIP lookup hyperlink)
    if re.match(r'^\*{0,2}_?\[.+?\]\(.+?\)_?\*{0,2}\s*$', s): return True
    return False


# ---------------------------------------------------------------------------
# Annotation injection — best-match across ALL lines
# ---------------------------------------------------------------------------

def annotate_lines(content_lines, fields):
    """
    For each form field, score every eligible content line,
    pick the best match (highest score, earliest line as tiebreaker),
    and inject the value tag after the matched span.

    Repeated labels (e.g. three 'Unit' entries) are assigned to successive
    top-scoring candidates so each gets its own line.
    """
    lines       = [l.rstrip('\n') for l in content_lines]
    real_fields = [(f['label'], f['value']) for f in fields if f['label']]
    match_count = {}

    for label, value in real_fields:
        label_sets = tokenize_label(label)
        if not label_sets:
            match_count[label] = match_count.get(label, 0) + 1
            continue

        needed = match_count.get(label, 0)
        tag    = ' ' + fmt_value(value)

        candidates = []
        for li, line in enumerate(lines):
            if is_skip_line(line):
                continue
            result = score_line(label_sets, line)
            if result is not None:
                score, start, end = result
                candidates.append((score, li, start, end))

        # Best score first, earliest line as tiebreaker
        candidates.sort(key=lambda x: (-x[0], x[1]))

        occurrence_seen = 0
        for score, li, start, end in candidates:
            if occurrence_seen < needed:
                occurrence_seen += 1
                continue
            lines[li] = lines[li][:end] + tag + lines[li][end:]
            break

        match_count[label] = match_count.get(label, 0) + 1

    return [l + '\n' for l in lines]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    pages = parse_pages(text)
    print(f"Parsed {len(pages)} pages.")

    out = []
    for page in pages:
        out.extend(page['header_lines'])
        out.extend(page['before_fields'])

        if page['fields_header']:
            out.append(page['fields_header'])
        for f in page['fields']:
            out.append(f['raw'])

        if page['content_header']:
            out.append(page['content_header'])

        real_fields = [f for f in page['fields'] if f['label']]
        if real_fields and page['content_lines']:
            out.extend(annotate_lines(page['content_lines'], real_fields))
        else:
            out.extend(page['content_lines'])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(out)

    print(f"Saved: {output_path}")


if __name__ == '__main__':
    inp  = sys.argv[1] if len(sys.argv) > 1 else 'output_with_forms.cleaned.md'
    outp = sys.argv[2] if len(sys.argv) > 2 else 'output_with_forms.mapped.md'
    process(inp, outp)