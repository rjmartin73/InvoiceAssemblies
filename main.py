import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Conduit types, Wire types

# Load data
date_pattern = re.compile(r'\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}[-/]\d{2}[-/]\d{2})\b')
half_inch_exp = re.compile(r'')
three_quarter_expr = re.compile(r'')
one_inch_expr = re.compile(r'')
one_and_quarter = re.compile(r'"\b[1]\s[1][/][4][\"]{,1}\b"gm')

wire_terms = {'wire', 'cable', 'thhn', 'xhhw', 'awg', 'stranded', 'thw', 'romex', '12/2','14/2', '20a', '30a', '4/0',
              '#10','# 10', '12/4', '#12','# 12', '#14', '# 14', '#16', 'cat5', 'cat6'}
conduit_terms = {'cndt', 'conduit', 'emt', 'ent', 'flex', 'grc', 'pipe', 'pvc', 'smurf'}
ent_terms = ['ent', 'smurf']
emt_terms = ['emt']
grc_terms = ['grc', 'rmc', 'steel', 'galva', 'rigid']
pvc_terms = ['pvc']
flex_terms = ['flex', 'carflex', 'sealtight', 'seal tight', 'liquid','liq tite', 'liq-tite']

conduit_exclude_terms = {'100A', '110A', '120A', '130A', '140A', '150A', '160A', '170A', '180A', '190A', '200A', '20A', '30A',
                         '401', '40A', '45', '50A', '60A', '70A', '80A', '90', '90A', 'accent', 'acorn', 'adap', 'adapt', 'assem',
                         'assembly', 'back', 'bell', 'bender', 'blow', 'body', 'box', 'branch', 'brush', 'bulb', 'bush', 'cadd',
                         'caddy', 'calmp', 'camera', 'can', 'cap', 'center', 'century', 'chair', 'changeover', 'city', 'cla',
                         'clam', 'clamp', 'clip', 'collar', 'comp', 'conen', 'conn', 'coul', 'coup', 'cover', 'cplg', 'crimp',
                         'cutter', 'degree', 'dent', 'dfac', 'die', 'elbow', 'entry', 'equip', 'ext', 'fee', 'female', 'fit',
                         'flange', 'gallon', 'gang', 'glue', 'groun', 'ground', 'hang', 'heater', 'hous', 'hngr', 'hub', 'insta', 'insul',
                         'lift', 'light', 'lock', 'lub', 'lug', 'male', 'marker', 'material', 'measur', 'ment', 'mount', 'nipp',
                         'nut', 'offset', 'oil', 'order', 'paint', 'pen', 'pencil', 'piston', 'plug', 'presentation', 'raceway',
                         'ream', 'recept', 'recess', 'rent', 'repair', 'return', 'rework', 'rig', 'rod', 'sand', 'saw', 'sconce',
                         'screw', 'seal', 'service', 'set', 'sharpie', 'sleeve', 'stand', 'stapl', 'stra', 'strap', 'strut', 'stub',
                         'supp', 'suppo', 'support', 't8', 'tape', 'tent', 'term', 'tie', 'toglock', 'tower', 'tray', 'vent',
                         'vise', 'watt', 'wrap', 'pick', 'ball', 'spray', 'reduc', 'xentry', 'flouresc', 'mitsubishi', 'snap', 'enclosure',
                         'oz', 'weather head', 'weatherhed', 'install', 'ush', 'unload' } 

wire_exclude_terms = {'100A', '110A', '120A', '15A', '20A', '30A', '40A', '50A', '60A', '70A', '80A', '90A', 'TY275M', 'adapter',
                      'anchor', 'bolt', 'book', 'box', 'bracket', 'break', 'cap', 'clamp', 'clip', 'connector', 'cover', 'cutter',
                      'duct', 'fault', 'fuse', 'hanger', 'head', 'hngr', 'jumper', 'kit', 'lug', 'marker', 'nut', 'nvent', 'paint', 'photocell',
                      'plate', 'pull', 'push', 'racetrack', 'rcpt', 'recept', 'screw', 'screws', 'square', 'strap', 'stud', 'switch',
                      'tap', 'tapcon', 'tie', 'tray', 'tug', 'washer', 'wireway', 'install', 'ush', 'unload', 'wiremold', 'mold', 'alert', 'wired',
                      'lube'}

diameter_class = [{'half':["1/2"]}, {'three_quarter':["3/4"]}, {'one':["1"]}]

emt_labels = {'1/2" CONDUIT - EMT ','3/4" CONDUIT - EMT ','1" CONDUIT - EMT ','1-1/4" CONDUIT - EMT ',
                        '1-1/2" CONDUIT - EMT ','2" CONDUIT - EMT ','2-1/2" CONDUIT - EMT ','3" CONDUIT - EMT ',
                        '3-1/2" CONDUIT - EMT ','4" CONDUIT - EMT '}
ent_patterns = {'3/4" CONDUIT - ENT',
                '1" CONDUIT - ENT',
                '1-1/4" CONDUIT - ENT'}
pvc_patterns ={'1/2" CONDUIT - PVC ','3/4" CONDUIT - PVC ','1" CONDUIT - PVC ','1-1/4" CONDUIT - PVC ',
              '1-1/2" CONDUIT - PVC ','2" CONDUIT - PVC ','2-1/2" CONDUIT - PVC ','3" CONDUIT - PVC ',
              '3-1/2" CONDUIT - PVC ','4" CONDUIT - PVC '}
grc_patterns = {'1/2" CONDUIT - GRC','3/4" CONDUIT - GRC','1" CONDUIT - GRC','1-1/4" CONDUIT - GRC',
              '1-1/2" CONDUIT - GRC','2" CONDUIT - GRC','2-1/2" CONDUIT - GRC','3" CONDUIT - GRC',
              '3-1/2" CONDUIT - GRC','4" CONDUIT - GRC'}
flex_patterns = {'FLEX/SEALTIGHT/CARFLEX','1/2" FLEX/SEALTIGHT/CARFLEX','3/4" FLEX/SEALTIGHT/CARFLEX',
               '1" FLEX/SEALTIGHT/CARFLEX','1-1/4" FLEX/SEALTIGHT/CARFLEX','1-1/2" FLEX/SEALTIGHT/CARFLEX',
               '2" FLEX/SEALTIGHT/CARFLEX','2-1/2" FLEX/SEALTIGHT/CARFLEX','3" FLEX/SEALTIGHT/CARFLEX',
               '3-1/2" FLEX/SEALTIGHT/CARFLEX','4" FLEX/SEALTIGHT/CARFLEX'}
wire_patterns = {'#14-#6 MC/ROMEX CABLE':re.compile(r'\b#14[-#6]\b|\bMC\b|\bROMEX\b'),
               'LOW VOLTAGE CABLE':re.compile(r'\bLOW\b|\bVOLTAGE\b|\bCABLE\b'),
               'DEVICE/JUNCTION BOX MAKEUP':re.compile(r'\bDEVICE\b|\bJUNCTION\b|\bBOX\b|\bMAKEUP\b'),
               '#14 WIRE THHN/XHHW/OTHER':re.compile(r'\b#14\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#12 WIRE THHN/XHHW/OTHER':re.compile(r'\b#12\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#10 WIRE THHN/XHHW/OTHER':re.compile(r'\b#10\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#8 WIRE THHN/XHHW/OTHER':re.compile(r'\b#8\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#6 WIRE THHN/XHHW/OTHER':re.compile(r'\b#6\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#4-#1/0 MC/ROMEX CABLE':re.compile(r'\b#4[-#1/0]\b|\bMC\b|\bROMEX\b'),
               '#2/0 AND LARGER MC/ROMEX CABLE':re.compile(r'\b#2/0\b|\bLARGER\b|\bMC\b|\bROMEX\b'),
               '#4 WIRE THHN/XHHW/OTHER':re.compile(r'\b#4\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#3 WIRE THHN/XHHW/OTHER':re.compile(r'\b#3\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#2 WIRE THHN/XHHW/OTHER':re.compile(r'\b#2\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#1 WIRE THHN/XHHW/OTHER':re.compile(r'\b#1\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#1/0 WIRE THHN/XHHW/OTHER':re.compile(r'\b#1/0\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#2/0 WIRE THHN/XHHW/OTHER':re.compile(r'\b#2/0\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#3/0 WIRE THHN/XHHW/OTHER':re.compile(r'\b#3/0\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#4/0 WIRE THHN/XHHW/OTHER':re.compile(r'\b#4/0\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#250 WIRE THHN/XHHW/OTHER':re.compile(r'\b#250\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#300 WIRE THHN/XHHW/OTHER':re.compile(r'\b#300\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#350 WIRE THHN/XHHW/OTHER':re.compile(r'\b#350\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#400 WIRE THHN/XHHW/OTHER':re.compile(r'\b#400\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#500 WIRE THHN/XHHW/OTHER':re.compile(r'\b#500\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#600 WIRE THHN/XHHW/OTHER':re.compile(r'\b#600\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),
               '#750 WIRE THHN/XHHW/OTHER':re.compile(r'\b#750\b|\bWIRE\b|\bTHHN\b|\bXHHW\b|\bOTHER\b'),}
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


df = pd.read_csv("./assets/invoice_descriptions.csv")

# print(sorted(wire_exclude_terms))

def classify_item(description):
    if isinstance(description, str) and date_pattern.search(description) is None:
        desc_lower = description.lower()
        if any(term.lower() in desc_lower for term in wire_terms) and not any(term.lower() in desc_lower for term in wire_exclude_terms):
            return 'Wire'
        elif any(term in desc_lower for term in conduit_terms) and not any(term in desc_lower for term in conduit_exclude_terms):
            # if any(term in desc_lower for term in ent_terms):
            #     return 'Conduit-ENT'
            # elif any(term in desc_lower for term in flex_terms):
            #     return 'Conduit-FLEX'
            # elif any(term in desc_lower for term in emt_terms):
            #     return 'Conduit-EMT'
            # elif any(term in desc_lower for term in grc_terms):
            #     return 'Conduit-GRC'
            # elif any(term in desc_lower for term in pvc_terms):
            #     return 'Conduit-PVC'
            return 'Conduit'
    return 'Exclude'


df['Label'] = df['Description'].apply(classify_item)

# df = df[df['Label'] == 'Wire']
# ...existing code...



# Get 1000 random rows from the DataFrame
random_sample_df = df.groupby('Label', group_keys=False,).apply(lambda x: x.sample(n=3000)).reset_index(drop=True)
random_sample_df = random_sample_df.drop_duplicates()
random_sample_shuffled_df = random_sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Save the random sample to a new CSV file
random_sample_shuffled_df.to_csv('./assets/random_sample_classified_items.csv', index=False)
df.to_csv('./assets/classified_items.csv', index=False)

# Filter the DataFrame to get records containing the term 'bell end'
# bell_end_df = df[df['Description'].str.contains('bell end', case=False, na=False)]

# Get 100 random rows from the filtered DataFrame
# random_bell_end_df = bell_end_df.sample(n=75, random_state=42)

# Save the random sample to a new CSV file
# random_bell_end_df.to_csv('./assets/random_bell_end_items.csv', index=False)
