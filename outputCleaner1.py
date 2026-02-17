import re
import sys
import os

# Matches lines like: - **form1[0].#subform[0].SomeField_FieldName[0]randomhash**: value
FORM_FIELD_LINE_RE = re.compile(
    r'^(\s*-\s+\*\*)([^*]+)(\*\*:\s*.*)$'
)

# Extracts the human-readable label from the field name
# e.g. "form1[0].#subform[0].Line1_MiddleName[0]abc123" -> "Middle Name"
FIELD_NAME_RE = re.compile(
    r'(?:^|[._])([A-Za-z][A-Za-z0-9]*)(?=\[\d+\][a-z0-9]{10,})'
)

def extract_label(raw_name: str) -> str:
    """
    From a raw field name like:
      form1[0].#subform[0].Line1_MiddleName[0]4w7wfftsck5xs5asry219mnhfzazvxlbdmbp
    Extract the meaningful part just before the [N]hash suffix.

    Strategy:
      1. Find the last segment before [N]<hash> (the hash is 10+ lowercase alphanum chars)
      2. Strip leading LineN_ or P1_ prefixes
      3. Split CamelCase into words
    """
    # Step 1: find the last named segment before [index]hash
    # Pattern: grab the last word-like token before [digit]lowercasehash
    match = re.search(r'(?:^|\.)([A-Za-z][A-Za-z0-9_]*)(?=\[\d+\][a-z0-9]{10,})', raw_name)
    if not match:
        # fallback: just return the raw name
        return raw_name

    segment = match.group(1)  # e.g. "Line1_MiddleName" or "P1_Line3_ZipCode"

    # Step 2: Strip numeric/alpha line prefixes like Line1_, Line7a_, Line7b_, P1_, etc.
    # Remove leading parts that look like LineN_, LineNa_, PN_
    segment = re.sub(r'^(?:(?:Line|P)\d+[a-z]?_)+', '', segment)
    # Also strip remaining plain Line_ prefix (e.g. Line_CityTown)
    segment = re.sub(r'^Line_', '', segment)
    # Strip trailing _Part\d+ suffixes (e.g. DaytimePhoneNumber1_Part8)
    segment = re.sub(r'_Part\d+$', '', segment)
    # Strip trailing digit from last word only (e.g. PhoneNumber1 -> PhoneNumber)
    segment = re.sub(r'(\D)\d+$', r'\1', segment)

    # Step 3: Split CamelCase into words
    # First, fix fused lowercase connectors embedded in CamelCase BEFORE splitting
    # e.g. "CompanyorOrg" -> "Companyor Org" won't help, so we target known connectors
    # surrounded by CamelCase boundaries: "Companyor" -> "Company or"
    for connector in ('or', 'of', 'and', 'in', 'to'):
        # connector at end of word before an uppercase letter: "Companyor" + "Org"
        segment = re.sub(
            rf'([A-Z][a-z]+)({connector})([A-Z])',
            rf'\1 {connector} \3',
            segment
        )
    # Insert space before uppercase letters that follow lowercase or digits
    spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', segment)
    # Insert space before sequences of uppercase followed by lowercase (e.g. "USCISForm" -> "USCIS Form")
    spaced = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced)
    # Replace underscores with spaces
    spaced = spaced.replace('_', ' ')
    # Collapse multiple spaces
    spaced = re.sub(r'\s+', ' ', spaced).strip()

    return spaced if spaced else raw_name


def clean_form_field_line(line: str) -> str:
    """
    If the line is a form field bullet, replace the raw field name with a clean label.
    Otherwise return the line unchanged.
    """
    m = FORM_FIELD_LINE_RE.match(line)
    if not m:
        return line

    prefix = m.group(1)       # e.g. "- **"
    raw_name = m.group(2)     # e.g. "form1[0].#subform[0].Line1_MiddleName[0]abc..."
    suffix = m.group(3)       # e.g. "**: N/A"

    # Only process if the raw name looks like an AcroForm path (contains subform or dots+hash)
    if not re.search(r'\[\d+\][a-z0-9]{10,}', raw_name):
        return line

    label = extract_label(raw_name)
    return f"{prefix}{label}{suffix}"


def clean_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned = []
    changed = 0
    for line in lines:
        new_line = clean_form_field_line(line.rstrip('\n'))
        if new_line != line.rstrip('\n'):
            changed += 1
        cleaned.append(new_line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
        if not cleaned[-1].endswith('\n'):
            f.write('\n')

    print(f"Done! {changed} form field lines cleaned.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    # Default paths — adjust as needed
    input_file  = "output_with_forms.md"
    output_file = "output_with_forms.cleaned.md"

    # Allow overrides via CLI args: python clean_form_fields.py input.md output.md
    if len(sys.argv) == 3:
        input_file  = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file  = sys.argv[1]

    clean_file(input_file, output_file)