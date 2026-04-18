from pathlib import Path
import json
import shutil
import re
import numpy as np
import re
import pandas as pd
import copy
from typing import Literal
from cal_metric_pc import cal_metric
from collections import Counter
import argparse

# from rich.traceback import install
# install()


def get_status(dataset_name, response_tasks):

    if dataset_name == "pancreatic_cancer":
        if "status:alive" in response_tasks.lower().replace(" ", "").replace(
            "\n", ""
        ).replace("*", "") or "answer:alive" in response_tasks.lower().replace(
            " ", ""
        ).replace(
            "\n", ""
        ).replace(
            "*", ""
        ):
            return "alive"
        elif "status:dead" in response_tasks.lower().replace(" ", "").replace(
            "\n", ""
        ).replace("*", "") or "answer:dead" in response_tasks.lower().replace(
            " ", ""
        ).replace(
            "\n", ""
        ).replace(
            "*", ""
        ):
            return "dead"
        else:
            return "nan"
    else:
        return response_tasks

def update_num_agents(content):
    num = 0
    for key, value in content.items():
        if "num_agents" in key:
            if value >= 1:
                num = value
            if "Turn 1" not in key and value == 0:
                content[key] = num
    return content


def get_role_answer(dataset_name, number, patient_id, content, json_output, num_agent):
    def inner():
        for role in content.keys():
            if "initial_assessment-" in role and "think" not in role:
                status = get_status(dataset_name, content[f"{role}"])
                role = role.replace("initial_assessment-", "")

                if num_agent == 1:
                    role = "role1"

                if role not in json_output.keys():
                    json_output[role] = []
                row = {}
                row["number"] = number
                row["patient_id"] = patient_id
                row["status"] = status
                json_output[role].append(row)

    try:
        inner()
    except Exception as e:
        print(f" error - {e.__class__.__name__}({e})")
    print()
    # inner()
    return json_output


def generate_json_results(
    file,
    version, dataset_name: Literal["pancreatic_cancer"],
    model_name: Literal["deepseek-reasoner", "gpt-4.1-mini"],
    num_agent=1
):
    file = Path(file)
    output_dir = file.parent
    final_decision = "decision-Moderator"

    # model_name = "deepseek-reasoner"
    files = sorted(file.glob("*.json"), key=lambda x: int(x.stem.split("-")[-1]))
    print(f"Length: {len(files)}")
    
    if files == 0:
        raise FileNotFoundError

    json_output = {final_decision: [], "major_voting": []}

    for number, path in enumerate(files):
        patient_id = path.name.split(".")[0].replace("_1", "")
        print(patient_id, end="")

        if patient_id == "RADCURE-0986":
            ...
        
        try:
            json_file = open(path, "r", encoding="utf-8")
            text = json_file.read()
            data = json.loads(text)
        except UnicodeDecodeError:
            json_file.close()
            with open(path, "r", encoding="gbk") as json_file:
                text = json_file.read()
            data = json.loads(text)
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

        content = data[patient_id]
        json_output = get_role_answer(
            dataset_name, number, patient_id, content, json_output, num_agent
        )

        status = get_status(dataset_name, content[f"{final_decision}"])
        row = {}
        row["number"] = number
        row["patient_id"] = patient_id
        row["status"] = status
        json_output[final_decision].append(row)

        if num_agent > 1:
            votes = []
            for role in json_output.keys():
                if role not in [final_decision, "major_voting"]:
                    votes.append(json_output[role][-1]["status"])
                    vote_counts = Counter(votes)

            final_result = vote_counts.most_common(1)[0][0]
            row = {}
            row["number"] = number
            row["patient_id"] = patient_id
            row["status"] = final_result
            json_output["major_voting"].append(row)

    print(len(json_output[final_decision]))

    with open(
        rf"{output_dir}\{dataset_name}_{model_name}_{version}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_output[final_decision],
            # {"number": number, "patient_id": patient_id, "status": status},
            f,
            ensure_ascii=False,
            indent=4,
        )

    if num_agent == 1:
        with open(
            rf"{output_dir}\{dataset_name}_{model_name}_{num_agent}_CoT.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                json_output["role1"],
                # {"number": number, "patient_id": patient_id, "status": status},
                f,
                ensure_ascii=False,
                indent=4,
            )
    else:
        with open(
            rf"{output_dir}\{dataset_name}_{model_name}_{num_agent}_major_voting.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                json_output["major_voting"],
                # {"number": number, "patient_id": patient_id, "status": status},
                f,
                ensure_ascii=False,
                indent=4,
            )

if __name__ == "__main__":

    import os

    file_mini_1_kamac = (
        rf"{os.path.dirname(__file__)}/results/pancreatic_cancer_gpt-4.1-mini_1_kamac_results"
    )
    file_mini_1_CoT = (
        rf"{os.path.dirname(__file__)}/results/pancreatic_cancer_gpt-4.1-mini_1_CoT_results"
    )
    version = "1_kamac"
    model_name = "gpt-4.1-mini"
    cal_metric(file_mini_1_kamac, dataset_name="pancreatic_cancer", model_name=model_name)
    cal_metric(file_mini_1_CoT, dataset_name="pancreatic_cancer", model_name=model_name)
