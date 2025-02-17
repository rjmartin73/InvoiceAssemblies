import re
import pandas as pd

size_patterns = {
    '3-1/2"': re.compile(r'(?<!\d)3[-| ]1/2(?!\d)'),
    '2-1/2"': re.compile(r'(?<!\d)2[-| ]1/2(?!\d)'),
    '1-3/4"': re.compile(r'(?<!/d)1[-| ]3/4(?!\d)'),
    '1-1/2"': re.compile(r'(?<!\d)1[-| ]1/2(?!\d)'),
    '1-1/4"': re.compile(r'(?<!\d)1[-| ]1/4(?!\d)'),
    '3/4"': re.compile(r'(?<!\d)3/4(?!\d)'),
    '1/2"': re.compile(r'(?<!\d)1/2(?!\d)'),
    '6"': re.compile(r'(?<!\d)6[”|"|in| |.]|(?<!\d)6$'),
    '5"': re.compile(r'(?<!\d)5[”|"|in| |.]|(?<!\d)5$'),
    '4"': re.compile(r'(?<!\d)4[”|"|in| |.]|(?<!\d)4$'),
    '3"': re.compile(r'(?<!\d)3[”|"|in| |.]|(?<!\d)3$'),
    '2"': re.compile(r'(?<!\d)2[”|"|in| |.]|(?<!\d)2$'),
    '1"': re.compile(r'(?<!\d)1[”|"|in| |.]|(?<!\d)1$'),
}
wire_sizes = {
    "4/0": re.compile(r'#4'),
    "3/0": re.compile(r''),
    "2/0": re.compile(r''),
    "1/0": re.compile(r''),
    "#250": re.compile(r''),
    "#300": re.compile(r''),
    "#350": re.compile(r''),
    "#400": re.compile(r''),
    "#500": re.compile(r''),
    "#600": re.compile(r''),
    "#750": re.compile(r''),
    "#14": re.compile(r'(\b#*14)\s?(awg)*[-/]?\b'),
    "#12": re.compile(r'(\b#*14)\s?(awg)*[-/]?\b'),
    "#10": re.compile(r'(\b#*14)\s?(awg)*[-/]?\b'),
    "#8": re.compile(r'(\b#*14)\s?(awg)*[-/]?\b'),
    "#6": re.compile(r'(\b#*14)\s?(awg)*[-/]?\b'),
    "LOW VOLTAGE": re.compile(r''),    
}

df = pd.read_csv("./assets/conduit_types.csv")
# print (df.describe())


# description = '"2- 1/2"" PVC Conduit, 10\', Sche"'

def get_conduit_size(description):
    description=description.replace("- ", " ").replace(" -", " ") 
    for size, pattern in size_patterns.items():
        if pattern.search(description):
            return f'{size}'  # ✅ Correctly returns the first match
    return 'UNK'  # ✅ Returns UNK only if no match is found


def get_wire_guage(description):
    description = description.lower()
    for guage, pattern in wire_sizes.items():
        if pattern.search(description):
            return f'{guage}' # ✅ Correctly returns the first match
    return 'UNK'  # ✅ Returns UNK only if no match is found 

df["ConduitDiameter"] = df["Description"].apply(get_conduit_size)
# def size_conduit():
#     for x, y in df[df["ActualLabel"], df["Description"]]:
#         print(f"{y} {x} {get_conduit_size(y)}")

# df["ConduitDiameter"] = df["ConduitDiameter"].astype(str)
# df["AssemblyDescription"] = df["AssemblyDescription"].str.replace('""', '"', regex=False)

df_size_training = df[["Description", "ConduitDiameter"]]
df_size_training = df_size_training.sample(n=2000, random_state=42)


df_size_training.to_csv("./assets/conduit_diameter_train.csv", index=False)

# print(get_conduit_size(description=description.replace("- ", " ").replace(" -", " ")))


#  '1/2"': re.compile(r'(?<!\d)1/2(?!\d)'),
#     '1-1/4"': re.compile(r'\b1[- ]?1/4\b'),
#     '3/4"': re.compile(r'\b3/4\b|\b0.75\b'),
#     '4"': re.compile(r'\b4\b'),
#     '3"': re.compile(r'\b3\b'),
#     '2"': re.compile(r'\b2\b'),
#     '1"': re.compile(r'\b1"{,1}\b'),