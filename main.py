import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# import polars as pl

# Conduit types, Wire types

# Load data
date_pattern = re.compile(r'\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}[-/]\d{2}[-/]\d{2})\b')
half_inch_exp = re.compile(r'')
three_quarter_expr = re.compile(r'')
one_inch_expr = re.compile(r'')
one_and_quarter = re.compile(r'"\b[1]\s[1][/][4][\"]{,1}\b"gm')

wire_terms = {
    'wire', 'cable', 'thhn', 'xhhw', 'awg', 'stranded', 'thw', 'romex', '12/2', '14/2', '20a', '30a', '4/0',
    '#10', '# 10', '12/4', '#12', '# 12', '#14', '# 14', '#16', 'cat5', 'cat6', '8/2'}
conduit_terms = {'cndt', 'conduit', 'emt', 'ent', 'flex', 'grc', 'pipe', 'pvc', 'smurf', 'sealtight'}
ent_terms = ['ent ', 'smurf']
emt_terms = ['emt', "thinwall", "thin wall"]
grc_terms = ['grc', 'rmc', 'steel', 'galv', ]
pvc_terms = ['pvc', 'rigid', 'ridgid', 'ridig']
flex_terms = ['flex', 'carflex', 'sealtight', 'seal tight', 'liquid','liq tite', 'liq-tite', "lfmc"]

conduit_exclude_terms = {
    '100A', '110A', '120A', '130A', '140A', '150A', '160A', '170A', '180A', '190A', '200A', '20A', '30A',
    '401', '40A', '45', '50A', '60A', '70A', '80A', '90', '90A', 'accent', 'acorn', 'adap', 'adapt', 'assem',
    'assembly', 'back', 'bell', 'bender', 'blow', 'body', 'box', 'branch', 'brush', 'bulb', 'bush', 'cadd',
    'caddy', 'calmp', 'camera', 'can', 'cap', 'center', 'century', 'chair', 'changeover', 'city', 'cla',
    'clam', 'clamp', 'clip', 'collar', 'comp', 'conen', 'conn', 'coul', 'coup', 'cover', 'cplg', 'crimp',
    'cutter', 'degree', 'dent', 'dfac', 'die', 'elbow', 'entry', 'equip', 'ext', 'fee', 'female', 'fit',
    'flange', 'gallon', 'gang', 'glue', 'groun', 'ground', 'hang', 'heater', 'hous', 'hngr', 'hub', 'insta', 'insul',
    'lift', 'light', 'lock', 'lub', 'lug', 'male', 'marker', 'material', 'measur', 'ment', 'mount', 'nipp',
    'nut', 'offset', 'oil', 'order', 'paint', 'pen', 'pencil', 'piston', 'plug', 'presentation', 'raceway',
    'ream', 'recept', 'recess', 'rent', 'repair', 'return', 'rework', 'rig ', 'rod', 'sand', 'saw', 'sconce',
    'screw', 'service', 'set', 'sharpie', 'sleeve', 'slv', 'stand', 'stapl', 'srap', 'stap', 'stra', 'strap', 'strut', 'stub',
    'supp', 'suppo', 'support', 't8', 'tape', 'tent', 'term', 'tie', 'toglock', 'tower', 'tray', 'vent',
    'vise', 'watt', 'wrap', 'pick', 'ball', 'spray', 'reduc', 'xentry', 'flouresc', 'mitsubishi', 'snap', 'enclosure',
    'oz', 'weather head', 'weatherhed', 'install', 'ush', 'unload', 'fluor', 'space', 'scent', 'cpl', 'temflex', 'cross',
    ' fa', ' ma ', 'ma,', 'male', 'fema', 'bolt', 'hole', 'putty', 'flexpro', 'concrete', 'bonding', 'agent', 'heat' 'gun',
    'seal off', 'outlet', 'price'} 

wire_exclude_terms = {
    '100A', '110A', '120A', '15A', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', 'TY275M', 'adapter',
    'anchor', 'bolt', 'book', 'box', 'bracket', 'break', 'cap', 'clamp', 'clip', 'conn', 'connector', 'cover', 'cutter',
    'duct', 'fault', 'fuse', 'hanger', 'head', 'hngr', 'jumper', 'kit', 'lug', 'marker', 'nut', 'nvent', 'paint', 'photocell',
    'plate', 'pull', 'push', 'racetrack', 'rcpt', 'recept', 'screw', 'screws', 'square', 'strap', 'stud', 'switch',
    'tap', 'tapcon', 'tie', 'tray', 'tug', 'washer', 'wireway', 'install', 'ush', 'unload', 'wiremold', 'mold', 'alert', 'wired',
    'lube', 'reduc', 'dimmer', 'hold', 'riser', 'bit', 'pigtail', 'bend', 'phillip', ' ma ', 'ma,', ' fa', 'term', 'ship', 'crimp',
    'pin ', 'offset', 'conduit', 'emt', 'transit', 'support', 'gutter', 'scissor', 'pig'}

def classify_item(description):
    if isinstance(description, str) and date_pattern.search(description) is None:
        desc_lower = description.lower()
        if any(term.lower() in desc_lower for term in wire_terms) and not any(term.lower() in desc_lower for term in wire_exclude_terms):
            return 'Wire'
        elif any(term in desc_lower for term in conduit_terms) and not any(term in desc_lower for term in conduit_exclude_terms):
            if any(term in desc_lower for term in ent_terms):
                return 'Conduit - ENT'
            elif any(term in desc_lower for term in flex_terms):
                return 'Conduit - FLEX'
            elif any(term in desc_lower for term in emt_terms):
                return 'Conduit - EMT'
            elif any(term in desc_lower for term in grc_terms):
                return 'Conduit - GRC'
            elif any(term in desc_lower for term in pvc_terms):
                return 'Conduit - PVC'
            return 'Conduit'
    return 'Exclude'


def classify_conduit(description):
    if isinstance(description, str) and date_pattern.search(description) is None:
        desc_lower = description.lower()
        if any(term in desc_lower for term in ent_terms):
            return 'CONDUIT - ENT'
        elif any(term in desc_lower for term in flex_terms):
                return 'CONDUIT - FLEX'
        elif any(term in desc_lower for term in emt_terms):
            return 'CONDUIT - EMT'
        elif any(term in desc_lower for term in grc_terms):
            return 'CONDUIT - GRC'
        elif any(term in desc_lower for term in pvc_terms):
            return 'CONDUIT - PVC'
        return 'UNK'


# Load csv
df = pd.read_csv("./assets/InvoiceData.csv", converters={"Description": lambda x: x.replace("\n","")})

# Keep only the description column
df = df[["Description"]]

# Remove any duplicates
df = df.drop_duplicates()

# Add the category as Label
df['ActualLabel'] = (df['Description'].apply(classify_item))

df_conduit = df[df['ActualLabel'].str.contains("Conduit", case=False)]
df_conduit['ConduitType'] = df_conduit['Description'].apply(classify_conduit)
df_conduit_types = df_conduit[['Description', 'ConduitType']]

df_wire = df[df['ActualLabel'].str.contains("wire", case=False)]
df_wire = df_wire[["Description", "ActualLabel"]]
# Get a random sample of 100 records of each label type
df = df.groupby('ActualLabel', group_keys=False,).apply(lambda x: x.sample(n=10)).reset_index(drop=True)
# df_conduit_types = df_conduit_types.groupby('ConduitType', group_keys=False).apply(lambda x: x.sample(n=100)).reset_index(drop=True)

# df_conduit_types = df_conduit_types.sample(n=1000, random_state=42)
# Shuffle
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Filter for spot checking
# df = df[df['Label'].contains('Conduit')]


# Get 1000 random rows from the DataFrame
# random_sample_df = df.groupby('Label', group_keys=False,).apply(lambda x: x.sample(n=3000)).reset_index(drop=True)
# random_sample_df = random_sample_df.drop_duplicates()
# random_sample_shuffled_df = random_sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Save the random sample to a new CSV file
# random_sample_shuffled_df.to_csv('./assets/random_sample_classified_items.csv', index=False)
# df.to_csv('./assets/classified_conduit.csv', index=False)
df_conduit.to_csv('./assets/conduit_types.csv', index=False)
df_conduit_types.to_csv('./assets/conduit_type_train.csv', index=False)
df_wire.to_csv("./assets/wire_main.csv", index=False)

# test_data_df = df.groupby('Label', group_keys=False,).apply(lambda x: x.sample(n=100)).reset_index(drop=True)
# test_data_df.to_csv("./assets/confusion_matrix_test_data.csv")

# Filter the DataFrame to get records containing the term 'bell end'
# bell_end_df = df[df['Description'].str.contains('bell end', case=False, na=False)]

# Get 100 random rows from the filtered DataFrame
# random_bell_end_df = bell_end_df.sample(n=75, random_state=42)

# Save the random sample to a new CSV file
# random_bell_end_df.to_csv('./assets/random_bell_end_items.csv', index=False)
