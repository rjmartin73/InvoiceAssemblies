import re

size_patterns = {
    '3-1/2"': re.compile(r'(?<!\d)3[-| ]1/2(?!\d)'),
    '2-1/2"': re.compile(r'(?<!\d)2[-| ]1/2(?!\d)'),
    '1-3/4': re.compile(r'(?<!/d)1[-| ]3/4(?!\d)'),
    '1-1/2"': re.compile(r'(?<!\d)1[-| ]1/2(?!\d)'),
    '1-1/4"': re.compile(r'(?<!\d)1[-| ]1/4(?!\d)'),
    '3/4"': re.compile(r'(?<!\d)3/4(?!\d)'),
    '1/2"': re.compile(r'(?<!\d)1/2(?!\d)'),
    '4"': re.compile(r'(?<!\d)4'),
    '3"': re.compile(r'(?<!\d)3'),
    '2"': re.compile(r'(?<!\d)2'),
    '1"': re.compile(r'(?<!\d)1'),
}

description = '"1/2"" PVC Conduit, 10\', Sche"'

def get_conduit_size(description):
    for size, pattern in size_patterns.items():
        if pattern.search(description):
            return f'Conduit-{size}'  # ✅ Correctly returns the first match
    return 'Conduit-UNK'  # ✅ Returns UNK only if no match is found

print(get_conduit_size(description=description.replace("- ", " ").replace(" -", " ")))


#  '1/2"': re.compile(r'(?<!\d)1/2(?!\d)'),
#     '1-1/4"': re.compile(r'\b1[- ]?1/4\b'),
#     '3/4"': re.compile(r'\b3/4\b|\b0.75\b'),
#     '4"': re.compile(r'\b4\b'),
#     '3"': re.compile(r'\b3\b'),
#     '2"': re.compile(r'\b2\b'),
#     '1"': re.compile(r'\b1"{,1}\b'),