import pandas as pd
import re

# Precompile regex patterns
date_pattern = re.compile(r'\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}[-/]\d{2}[-/]\d{2})\b')

size_patterns = {
    '1/2"': re.compile(r'\b1/2\b|\b0.5\b'),
    '3/4"': re.compile(r'\b3/4\b|\b0.75\b'),
    '1"': re.compile(r'\b1\b(?!/4)'),
    '1-1/4"': re.compile(r'\b1[- ]?1/4\b'),
    '1-1/2"': re.compile(r'\b1[- ]?1/2\b'),
    '2"': re.compile(r'\b2\b'),
    '2-1/2"': re.compile(r'\b2[- ]?1/2\b'),
    '3"': re.compile(r'\b3\b'),
    '3-1/2"': re.compile(r'\b3[- ]?1/2\b'),
    '4"': re.compile(r'\b4\b')
}

# Keyword sets for quick lookup
wire_terms = {'wire', 'cable', 'thhn', 'xhhw', 'awg', 'stranded', 'thw', 'romex', '12/2', '14/2', '20a', '30a', '4/0',
              '#10', '#12', '#14', '#16'}
conduit_terms = {'pvc', 'ent', 'emt', 'flex', 'pipe', 'cndt', 'conduit', 'grc'}
ent_terms = {'ent', 'smurf'}
emt_terms = {'emt'}
grc_terms = {'grc', 'rmc', 'steel', 'galva', 'rigid'}
pvc_terms = {'pvc'}
flex_terms = {'flex', 'carflex', 'sealtight', 'seal tight', 'liquid', 'liq tite', 'liq-tite'}

# Exclusion terms
conduit_exclude_terms = {'elbow', '90', '45', 'sleeve', 'conn', 'box', 'strap', 'changeover', 'adapt', 'coup', 'bulb',
                         'cap', 'bush', 'coul', 'conen', 'strut', 'comp', 'nipp', 'male', 'female', 'adap', 'mount',
                         'tie', 'nut', 'glue', 'oil', 'lub', 'hang', 'stapl', 'tray', 'city', 'rent', 'degree', 't8',
                         'watt', 'ext', 'fit', 'brush', 'bell', 'cplg', 'lock', '401', 'body', 'clamp', 'center',
                         'equip', 'dent', 'hub', 'insul', 'return', 'ment', 'material', 'caddy', 'plug', 'tape',
                         'vent', 'paint', 'can', 'collar', 'insta', 'saw', 'sand', 'light', 'clip', 'cadd', 'ground',
                         'support', 'piston', 'hous', 'measur', 'seal', 'bender', 'cover', 'ream', 'clam', 'camera',
                         'cutter', 'die', 'rework', 'suppo', 'blow', 'repair', 'vise', 'stand', 'wrap', 'calmp',
                         'groun', 'back', 'stub', 'screw'}
wire_exclude_terms = {'screw', 'screws', 'jumper', 'nut', 'kit', 'washer', 'lug', 'pull', 'fuse', 'photocell', 'stud','head',
                       'adapter', 'bracket', 'racetrack', 'tap', 'square', 'break', 'fault', 'switch', 'tie', 'push', 
                       'recept', 'pull, ''tapcon', 'screw', 'bolt', 'anchor', 'hanger', 'strap', 'clamp', 'bracket',}

# Load data
df = pd.read_csv("./assets/invoice_descriptions.csv")


def classify_item(description):
    if not isinstance(description, str) or date_pattern.search(description):
        return 'Other'

    desc_lower = description.lower()

    # Check for wire classification first
    if wire_terms & set(desc_lower.split()) and not (wire_exclude_terms & set(desc_lower.split())):
        return 'Wire'

    # Check for conduit classification
    if conduit_terms & set(desc_lower.split()) and not (conduit_exclude_terms & set(desc_lower.split())):
        # Determine specific conduit type
        if ent_terms & set(desc_lower.split()):
            conduit_type = 'ENT'
        elif flex_terms & set(desc_lower.split()):
            conduit_type = 'FLEX'
        elif emt_terms & set(desc_lower.split()):
            conduit_type = 'EMT'
        elif grc_terms & set(desc_lower.split()):
            conduit_type = 'GRC'
        elif pvc_terms & set(desc_lower.split()):
            conduit_type = 'PVC'
        else:
            conduit_type = 'General Conduit'

        # Determine conduit size
        for size, pattern in size_patterns.items():
            if pattern.search(desc_lower):
                return f'Conduit-{conduit_type} {size}'

        return f'Conduit-{conduit_type}'

    return 'Other'


df['Label'] = df['Description'].apply(classify_item)
df = df[df['Label'] == 'Wire']
df.to_csv('./assets/classified_items.csv', index=False)
