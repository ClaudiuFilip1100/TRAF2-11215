import pandas as pd
import json

f = open("../../data/45a-update-2_annotations.json")
js = json.load(f)

del js["id"]
del js["relations"]
del js["annotators_per_example"]
del js["dataset"]
del js["name"]

tag_types = []
for tag in js["schema"]["tags"]:
    tag_types.append(tag["name"])
del js["schema"]


def replace_dots(x):
    return x.replace(":", " ")


def replace_comma(x):
    return x.replace(",", "")


# create DataFrame
df = pd.DataFrame(columns=["words", "sentence #", "tag"])

for i in range(len(js["examples"])):
    # create df to append
    df_app = pd.DataFrame(columns=["words", "sentence #", "tag"])

    # remove commas and :
    content = replace_dots(replace_comma(js["examples"][i]["content"]))

    # create a list with all the annotations
    annotations = js["examples"][i]["annotations"]

    # create a list with all tags
    tags = ["O" for i in range(len(content.split()))]

    # for each annotation get the value and add its tag
    for annotation in annotations:
        value = annotation["value"]
        for idx, word in enumerate(content.split()):
            if word in value:
                tags[idx] = annotation["tag"]

    # create the DataFrame to append
    df_app["words"] = content.split()
    df_app["sentence #"] = i
    df_app["tag"] = tags
    df = pd.concat([df, df_app], ignore_index=True)

df.to_csv("../../data/dataset_from_json_v2.csv", index=False)
