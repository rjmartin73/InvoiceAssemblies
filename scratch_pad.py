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
    "#4/0": re.compile(r'(?<!\w)#?4/0(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#3/0": re.compile(r'(?<!\w)#?3/0(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#2/0": re.compile(r'(?<!\w)#?2/0(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#1/0": re.compile(r'(?<!\w)#?1/0(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#750": re.compile(r'(?<!\w)#?750(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#600": re.compile(r'(?<!\w)#?600(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#500": re.compile(r'(?<!\w)#?500(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#400": re.compile(r'(?<!\w)#?400(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#350": re.compile(r'(?<!\w)#?350(?:KCMIL|mcm|(?=[a-zA-Z]))?', re.IGNORECASE),
    "#300": re.compile(r'(?<!\w)#?300(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#250": re.compile(r'(?<!\w)#?250(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#14": re.compile(r'(?<!\w)#?14(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#12": re.compile(r'(?<!\w)?#?12(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#10": re.compile(r'(?<!\w)#?10(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#8": re.compile(r'(?<!\w)#?8(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#6": re.compile(r'(?<!\w)#?6(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#4": re.compile(r'(?<!\w)#?4(?!/0)(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#3": re.compile(r'(?<!\w)#?3(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#2": re.compile(r'(?<!\w)#?2(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "#1": re.compile(r'(?<!\w)#?1(?:KCMIL|mcm)?(?=[a-zA-Z\s_/.-]|$)', re.IGNORECASE),
    "LOW VOLTAGE": re.compile(r'\b(?:low\s*voltage|cat[56][E]?|romex|fplr|cable|["|#]?16|18)\b', re.IGNORECASE)
}



df = pd.read_csv("./assets/conduit_types.csv")
df_wire = pd.read_csv("./assets/wire_main.csv")
# print (df_wire.describe())


# description = '"2- 1/2"" PVC Conduit, 10\', Sche"'

def get_conduit_size(description):
    description = description.replace("- ", " ").replace(" -", " ") 
    for size, pattern in size_patterns.items():
        if pattern.search(description):
            return f'{size}'  # ✅ Correctly returns the first match
    return 'UNK'  # ✅ Returns UNK only if no match is found


def get_wire_gauge(description):
    description = description.lower()
    for gauge, pattern in wire_sizes.items():
        if pattern.search(description):
            return f'{gauge}' # ✅ Correctly returns the first match
    return 'UNK'  # ✅ Returns UNK only if no match is found 


df["ConduitDiameter"] = df["Description"].apply(get_conduit_size)
df_wire["WireGauge"] = df_wire["Description"].apply(get_wire_gauge)
# def size_conduit():
#     for x, y in df[df["ActualLabel"], df["Description"]]:
#         print(f"{y} {x} {get_conduit_size(y)}")

# df["ConduitDiameter"] = df["ConduitDiameter"].astype(str)
# df["AssemblyDescription"] = df["AssemblyDescription"].str.replace('""', '"', regex=False)

df_size_training = df[["Description", "ConduitDiameter"]]
df_size_training = df_size_training.sample(n=2000, random_state=42)
df_wire_training = df_wire[["Description", "WireGauge"]]
df_wire_training = df_wire_training.sample(n=1000, random_state=42)


df_size_training.to_csv("./assets/conduit_diameter_train.csv", index=False)
df_wire_training.to_csv("./assets/wire_training.csv", index=False)

# print(get_conduit_size(description=description.replace("- ", " ").replace(" -", " ")))


#  '1/2"': re.compile(r'(?<!\d)1/2(?!\d)'),
#     '1-1/4"': re.compile(r'\b1[- ]?1/4\b'),
#     '3/4"': re.compile(r'\b3/4\b|\b0.75\b'),
#     '4"': re.compile(r'\b4\b'),
#     '3"': re.compile(r'\b3\b'),
#     '2"': re.compile(r'\b2\b'),
#     '1"': re.compile(r'\b1"{,1}\b'),