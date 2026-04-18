import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable
from pptree import Node
from pptree import *
from agent.client import Agent, CachedBase, time_it, create_logger, print_log
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import copy
from typing import Literal
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm
import json
import re
import optuna
import argparse
from types import SimpleNamespace

print = tqdm.write

agent_emoji = [
    "\U0001f468\u200d\u2695\ufe0f",
    "\U0001f468\U0001f3fb\u200d\u2695\ufe0f",
    "\U0001f469\U0001f3fc\u200d\u2695\ufe0f",
    "\U0001f469\U0001f3fb\u200d\u2695\ufe0f",
    "\U0001f9d1\u200d\u2695\ufe0f",
    "\U0001f9d1\U0001f3ff\u200d\u2695\ufe0f",
    "\U0001f468\U0001f3ff\u200d\u2695\ufe0f",
    "\U0001f468\U0001f3fd\u200d\u2695\ufe0f",
    "\U0001f9d1\U0001f3fd\u200d\u2695\ufe0f",
    "\U0001f468\U0001f3fd\u200d\u2695\ufe0f",
]
random.shuffle(agent_emoji)
process_count = 0


def create_question(sample, dataset):
    if dataset == "med_qa":
        question = sample["question"] + " Options: "
        options = []
        for k, v in sample["options"].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question
    return sample["question"]


def load_dataset(dataset_name: Literal["radcure", "pancreatic_cancer", "med_qa"], add_examplers):

    if dataset_name == "radcure":
        with open(
            f"{os.environ['work_dir']}/data/radcure_test.json",
            "r",
            encoding="utf-8",
        ) as json_file:
            patient_list = json.load(json_file)

        return patient_list, None, None
    
    elif dataset_name == "pancreatic_cancer":

        with open("data/pancreatic_cancer_data_clean.json", "r", encoding="utf-8") as json_file:
            patient_list = json.load(json_file)
        
        return patient_list, None, None

    elif dataset_name == "med_qa":
        import datasets

        def vis_func(data):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.imshow(data["image"])
            ax.set_title(data["question"], fontsize=10, wrap=True)

        cached_name = f"cached_{dataset_name}"
        dataset_name, vis_func = {
            "med_qa": ("bigbio/med_qa", lambda x: print(x)),
        }[dataset_name]
        dataset_name = dataset_name.split("/")[1].replace("-", "_")

        if os.path.exists(f"{os.environ["work_dir"]}/data/{cached_name}.arrow"):
            dataset = datasets.load_from_disk(
                f"{os.environ["work_dir"]}/data/{cached_name}.arrow"
            )
        else:
            dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)
            dataset.save_to_disk(f"{os.environ["work_dir"]}/data/{cached_name}.arrow")

        examplers = []
        test_data = []
        if dataset_name == "med_qa":
            for line in dataset["train"]:
                line["options"] = {
                    item["key"]: item["value"] for item in line["options"]
                }
                if add_examplers:
                    examplers.append(line)

            for idx, line in enumerate(dataset["test"]):
                line["patient_id"] = f"{dataset_name}-{idx:04d}"
                line["options"] = {
                    item["key"]: item["value"] for item in line["options"]
                }
                test_data.append(line)

        print(str(dataset))

    return test_data, examplers, vis_func


def ReadSeriesImage(data_directory) -> sitk.Image:
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    if not series_IDs:
        raise FileNotFoundError(
            'ERROR: given directory "'
            + data_directory
            + '" does not contain a DICOM series.'
        )

    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        data_directory, series_IDs[0]
    )

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    return sitk.GetArrayFromImage(image3D)


def apply_mask(image, mask, color, alpha=0.8):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    image = image.astype(np.float32)
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return np.clip(image, 0, 255).astype(np.uint8)


def downsample_images(image_2d, scale_factor=[0.5, 0.5]):
    # image_2d: z,y,x
    # bbox (list or tuple): [z_min, y_min, x_min, z_max, y_max, x_max]
    image_ds = resize(
        image_2d,
        (
            int(image_2d.shape[0] * scale_factor[0]),
            int(image_2d.shape[1] * scale_factor[1]),
        ),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(image_2d.dtype)

    return image_ds


def parse_hierarchy(info, emojis):
    moderator = Node("moderator (\U0001f468\u200d\u2696\ufe0f)")
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split("-")[0].split(".")[1].strip()
        except:
            expert = expert.split("-")[0].strip()

        if hierarchy is None:
            hierarchy = "Independent"

        if "independent" not in hierarchy.lower():
            try:
                parent = hierarchy.split(">")[0].strip()
                child = hierarchy.split(">")[1].strip()
            except Exception as e:
                child = "Independent"
                # print("parse_hierarchy: ", info, "\n", hierarchy, e)

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{}".format(child), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{}".format(expert), moderator)
            agents.append(agent)

        count += 1

    return agents


def generate_assessment(
    args,
    dataset_name,
    question,
    answer_template,
    agent_dict,
    action,
    round_opinions,
    interaction_log,
    fewshot_examplers="",
    logger=None,
    img_path=None,
    num_agents=0,
    use_cached=False,
):
    """ """
    # initial_report = ""
    if args.cot:
        prompt = """
                    The following are binary questions (with answers) about medical knowledge.
                    \n\nYou are not allowed to switch to any other medical specialty.
                    When questions fall outside of core cardiology (e.g., neurology, dermatology), do not reject them outright.
                    Instead, consider how a {role} might reasonably interpret or respond to such questions from their perspective
                    — possibly by relating them to your expertise.
                    Avoid general-purpose explanations. Always respond as a focused medical professional.
                    Even when the case touches on other specialties (e.g., oncology, ENT), you must interpret it from a {role}'s perspective
                    and relate it back to cardiovascular implications if relevant.
                    \n\nFirst, think step by step as a {role} analyzing this patient's survival probability.
                    Then, at the end of your reasoning, state your final answer as exactly one word on a new line:
                    You are provided with example below.
                    \n\n{fewshot_examplers}\n\n**Question:** {question}\nResponse Format: \n{answer_template}\n\n"""
    else:
        prompt = """Given the examplers, as a {role}, please return your answer to the medical query among the option provided. You are not allowed to switch to any other medical specialty.
                    \n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\n{answer_template}"""

    for idx, (k, v) in enumerate(agent_dict.items()):
        # if use_cached:
        #     v.messages.append({"role": "user", "content": prompt})
        #     v.messages.append({"role": "assistant", "content": round_opinions[k.lower()]})
        # else:
        opinion = v.temp_responses(
            prompt.format(
                role=v.role,
                question=question,
                answer_template=answer_template,
                fewshot_examplers=fewshot_examplers,
            ),
            action=action,
            img_path=img_path,
            use_cached=use_cached,
        )
        # initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[k.lower()] = opinion
        interaction_log[f"Agent {idx+num_agents+1}"]["VP"] = get_status(
            dataset_name, opinion
        )
        print_log(f"({k.lower()}): {opinion}\n", logger=logger)

    return round_opinions, interaction_log


def recruit_stage(
    dataset_name,
    patient_id,
    question,
    num_agents=3,
    logger=None,
    use_cached=False,
    cache=None,
    examplers=None,
):
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts 
                     with diverse identity and ask them to discuss and solve the given medical query."""

    tmp_agent = Agent(
        instruction=recruit_prompt,
        role="recruiter",
        logger=logger,
        meta=patient_id,
        model_info="gpt-4.1-mini",
        cache=cache,
    )

    if dataset_name == "radcure":
        instruction = (
            f"Question: {question}\n"
            f"Considering the medical question and the options for the answer, "
            f"what kinds of experts will you recruit to better make an accurate decision?"
            f"\n\nYou also need to clearly specify the communication structure between experts "
            f"or indicate if they are independent.\n\n"
            f"You must recruit exactly the following {num_agents} experts, with no substitutions, no additional experts, and no omissions:\n"
            f"(e.g., Radiation Oncologist == Medical Oncologist == Pathologist == Surgical Oncologist (Recurrence/Secondary Cancers) == Targeted Therapy Expert), "
            f"Please strictly follow the format shown below, without adding any extra explanation or reasoning.\n"
            f"\nFormat example if recruiting {num_agents} experts:"
            f"\n1. Radiation Oncologist - Your expertise is strictly limited to radiation therapy planning and dosing for head and neck squamous cell carcinoma, especially HPV-positive cases. - Hierarchy: Radiation Oncologist == Medical Oncologist"
            f"\n2. Medical Oncologist - Your expertise is strictly limited to systemic therapy decisions, including chemotherapy and immunotherapy in head and neck cancers. - Hierarchy: Medical Oncologist == Radiation Oncologist"
            f"\n3. Surgical Oncologist (Recurrence/Secondary Cancers) - Your expertise is strictly limited to evaluating surgical options for recurrent or secondary malignancies in head and neck cancers. - Hierarchy: Surgical Oncologist == Pathologist"
            f"\n4. Pathologist - Your expertise is strictly limited to pathological diagnosis of head and neck squamous cell carcinoma, HPV status evaluation, and margin assessment post-surgery. - Hierarchy: Pathologist == Surgical Oncologist"
            f"\n5. Targeted Therapy Expert - Your expertise is strictly limited to clinical application of EGFR inhibitors and novel agents targeting HPV-positive tumors. - Hierarchy: Targeted Therapy Expert -> Medical Oncologist"
            f"\n\nYour answer must conform exactly to the format above."  # , with {num_agents} experts.
        )
    elif dataset_name == "pancreatic_cancer":
        # Modify this prompt to change the recruitment process
        instruction = (
            f"Question: {question}"
            f"You can recruit {num_agents} experts in different medical specialties."
            f"Considering the pancreatic cancer prognosis question and the need to estimate vital status (Alive or Dead), what experts should be recruited to improve the accuracy of the prediction?"
            f"Also specify the communication structure between experts (e.g., Pancreatic Surgical Oncologist == Medical Oncologist == GI Pathologist > Abdominal Radiologist), or indicate if they are independent."
            f"Prefer experts relevant to pancreatic ductal adenocarcinoma prognosis, such as:"
            f"- Pancreatic Surgical Oncologist"
            f"- Medical Oncologist"
            f"- Gastrointestinal Pathologist"
            f"- Abdominal Radiologist"
            f"- Cancer Prognosis Specialist / Clinical Outcomes Specialist"
            f"Format exactly like:"
            f"1. Pancreatic Surgical Oncologist - Focuses on resectability, pathologic stage, and surgical-risk-related prognosis in pancreatic cancer. - Hierarchy: Independent"
            f"2. Medical Oncologist - Focuses on systemic therapy implications and clinical prognosis in pancreatic cancer - Hierarchy: Pancreatic Surgical Oncologist > Medical Oncologist"
            f"3. Gastrointestinal Pathologist - Focuses on morphology, grade, nodal disease, and pathologic staging interpretation - Hierarchy: Independent"
            f"4. Abdominal Radiologist - Focuses on imaging-based tumor extent and metastatic-risk interpretation - Hierarchy: Independent"
            f"5. Clinical Outcomes Specialist - Focuses on prognosis estimation from structured clinical variables and follow-up patterns - Hierarchy: Independent"
            f"Please answer in exactly the above format and do not include your reason."
        )
    elif dataset_name == "med_qa":
        instruction = (
            f"Question: {question}\n"
            f"You can recruit {num_agents} experts in different medical expertise. "
            f"Considering the medical question and the options for the answer, "
            f"what kind of experts will you recruit to better make an accurate answer?\n"
            f"Also, you need to specify the communication structure between experts "
            f"(e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), "
            f"or indicate if they are independent.\n\nFor example, if you want to recruit five experts, "
            f"you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, "
            f"children, and adolescents. - Hierarchy: Independent\n"
            f"2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. "
            f"- Hierarchy: Pediatrician > Cardiologist\n"
            f"3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. "
            f"- Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, "
            f"especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n"
            f"5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\n"
            f"Please answer in above format, and do not include your reason."
        )

    recruited = tmp_agent.temp_responses(
        instruction, action="recruit", use_cached=use_cached
    )

    fewshot_examplers = ""
    new_examplers = []
    if examplers != None:
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = f"[Example {ie+1}]\n" + exampler["question"]
            options = [f"({k}) {v}" for k, v in exampler["options"].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.temp_responses(
                f"Below is an example of medical knowledge question and answer. "
                f"After reviewing the below medical question and answering, "
                f"can you provide 1-2 sentences of reason that support the answer as "
                f"you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}",
                action=f"{ie}-examplers",
                use_cached=use_cached,
            )

            tmp_exampler["question"] = exampler_question
            tmp_exampler["reason"] = exampler_reason
            tmp_exampler["answer"] = exampler_answer
            new_examplers.append(tmp_exampler)
            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    agents_info = [
        agent_info.split(" - Hierarchy: ")
        for agent_info in recruited.split("\n")
        if agent_info
    ]
    agents_data = [
        (info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info
    ]

    if len(agents_data) != num_agents:
        comment = "\n".join(f"{info[0]} - Hierarchy: {info[1]}" for info in agents_info[:num_agents])
        cache.do_save_messages("recruit", comment, "recruiter", None)
    return tmp_agent, agents_data[:num_agents], fewshot_examplers


class MDTPanel:
    def __init__(self, patient_id, num_agents, num_turns, num_rounds, logger=None):

        self.logger = logger
        self.num_agents = num_agents
        self.patient_id = patient_id
        # self.agent_emoji = agent_emoji

        self.interaction_log = {
            f"Round {round_num}": {
                f"Turn {turn_num}": {
                    f"Agent {source_agent_num}": {
                        **{
                            f"Agent {target_agent_num}": None
                            for target_agent_num in range(1, num_agents + 1)
                        },
                        "VP": None,
                    }
                    for source_agent_num in range(1, num_agents + 1)
                }
                for turn_num in range(1, num_turns + 1)
            }
            for round_num in range(1, num_rounds + 2)
        }

    def plot_interaction(self):

        myTable = PrettyTable(
            [""]
            + [f"Agent {i+1} " for i in range(self.num_agents)]
            + ["VP"]  # ({agent_emoji[i]})
        )
        num_cooperations = 0
        for i in range(1, self.num_agents + 1):
            row = [f"Agent {i} "]  # ({agent_emoji[i-1]})

            vp = ""
            for k in range(1, len(self.interaction_log) + 1):
                for l in range(1, len(self.interaction_log["Round 1"]) + 1):
                    tmp = self.interaction_log[f"Round {k}"][f"Turn {l}"][f"Agent {i}"][
                        "VP"
                    ]
                    if tmp is not None and tmp.lower() != "no":
                        if k == 1 and l == 1:
                            vp += self.interaction_log[f"Round {k}"][f"Turn {l}"][
                                f"Agent {i}"
                            ]["VP"]
                        else:
                            vp += (
                                "->"
                                + self.interaction_log[f"Round {k}"][f"Turn {l}"][
                                    f"Agent {i}"
                                ]["VP"]
                            )

            for j in range(1, self.num_agents + 1):
                if i == j:
                    row.append("")
                else:
                    i2j = any(
                        self.interaction_log[f"Round {k}"][f"Turn {l}"][
                            f"Agent {i}"
                        ].get(f"Agent {j}")
                        is not None
                        for k in range(1, len(self.interaction_log) + 1)
                        for l in range(1, len(self.interaction_log[f"Round {k}"]) + 1)
                    )

                    j2i = any(
                        self.interaction_log[f"Round {k}"][f"Turn {l}"][f"Agent {j}"][
                            f"Agent {i}"
                        ]
                        is not None
                        for k in range(1, len(self.interaction_log) + 1)
                        for l in range(1, len(self.interaction_log["Round 1"]) + 1)
                    )

                    # and self.interaction_log[f"Round {k}"][f"Turn {l}"][
                    # f"Agent {j}"
                    # ][f"Agent {i}"].lower() != "no"
                    if not i2j and not j2i:
                        row.append(" ")
                    elif i2j and not j2i:
                        num_cooperations += 1
                        row.append(f"({i}->{j})")  # \u270b
                    elif j2i and not i2j:
                        num_cooperations += 1
                        row.append(f"({i}<-{j})")
                    elif i2j and j2i:
                        num_cooperations += 1
                        row.append(f"({i}<->{j})")
            row.append(vp)
            myTable.add_row(row)
            if i != self.num_agents:
                myTable.add_row(["" for _ in range(self.num_agents + 2)])
        print(f"\n{self.patient_id} Interaction Log")
        print_log(f"\n{self.patient_id} Interaction Log", logger=self.logger)
        print_log(myTable, logger=self.logger)
        print(str(myTable))

        return num_cooperations


class DiscussionPanel:
    def __init__(
        self,
        patient_id,
        question,
        answer_template,
        agent_list,
        agent_dict,
        medical_agents,
        round_opinions,
        panel,
        recruiter=None,
        fewshot_examplers=None,
        img_path=None,
        bbox_coords=None,
        logger=None,
        df_pred_masked=None,
        args=None,
        cache=None,
    ):
        self.args = args
        self.dataset_name = args.dataset_name
        self.logger = logger
        self.num_rounds = args.num_rounds
        self.panel = panel
        self.interaction_log = panel.interaction_log

        self.question = question
        self.answer_template = answer_template
        self.patient_id = patient_id
        self.round_opinions = round_opinions
        self.medical_agents = medical_agents
        self.agent_list = agent_list
        self.agent_dict = agent_dict

        self.recruiter = recruiter
        self.assistant = Agent(
            instruction="You are a professional medical assistant helping to analyze a multi-expert discussion.",
            role="assistant",
            meta=self.patient_id,
            model_info="gpt-4.1-mini",
            cache=cache,
        )

        self.df_pred_masked = df_pred_masked
        self.img_path = img_path
        self.bbox_coords = bbox_coords
        self.num_turns = args.num_turns
        self.cache = cache
        self.round_gap = {
            turn_num: {role: None for role in self.agent_dict.keys()}
            for turn_num in range(self.num_turns)
        }
        self.auto_recruit = args.auto_recruit

    def start_discuss(self, cache=None):

        final_answer = ""
        interaction_log = copy.deepcopy(self.panel.interaction_log)
        comment = {}
        for idx, (k, option) in enumerate(self.round_opinions[1].items()):
            initial_answer = get_status(self.dataset_name, option)
            interaction_log[f"Round 1"]["Turn 1"][f"Agent {idx+1}"][
                "VP"
            ] = initial_answer
            comment[k] = initial_answer
        self.cache.save_round_comment("Round 1", comment, self.agent_dict.keys(), None)
        self.turn_comments = {t: {} for t in range(1, self.num_turns + 1)}
        for round_num in range(1, self.num_rounds + 1):
            print("v" * 50 + f" Round {round_num+1} " + "v" * 50)
            print_log(
                "v" * 50 + f" Round {round_num+1} " + "v" * 50, logger=self.logger
            )
            round_name = f"Round {round_num+1}"
            # for turn_num in range(num_turns):
            turn_num = 0
            last_turn_name = f"Turn {turn_num + 1}"
            self.turn_comments[turn_num] = self.round_opinions[round_num]
            while True:
                turn_name = f"Turn {turn_num + 1}"
                print("=" * 50 + f" {turn_name} " + "=" * 50)
                print_log("=" * 50 + f" {turn_name} " + "=" * 50, logger=self.logger)

                outputs = []
                for idx, v in enumerate(self.medical_agents):
                    outputs.append(f"{idx+1}. {v.role}  ({v.model_info}) |")
                    # print(f"{idx+1}. {v.role}  ({v.model_info}) |", end=" ")
                print(" ".join(outputs))
                print_log(" ".join(outputs), logger=self.logger)
                print("")
                num_yes = 0
                num_yes_gap = 0
                assessment = [
                    f"{k}: {get_status(self.dataset_name, v)}"
                    for k, v in self.turn_comments[turn_num].items()
                ]
                options = "".join(
                    f"({k.lower()}): {v}\n"
                    for k, v in self.turn_comments[turn_num].items()
                )
                if len(set(assessment)) != 1:
                    assessment = options
                for idx, _agent in enumerate(self.medical_agents):

                    # turn + round is a two-stage decision-making,
                    # turn stage is very weak.
                    # but we can simply deal with it with one stage.
                    all_comments = "".join(
                        f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n"
                        for _k, _v in self.interaction_log[round_name][
                            turn_name
                        ].items()
                    )

                    # _agent.messages[0]["content"] = f"\nDiscussion options:\n {assessment}"

                    num_yes_tmp = self.discuss(
                        "discuss",
                        idx,
                        _agent,
                        turn_num,
                        turn_name,
                        round_name,
                        assessment,
                        all_comments,  # no use
                    )

                    if self.auto_recruit:
                        num_yes_gap_tmp = self.ask_gap(
                            _agent, turn_num, turn_name, round_name
                        )
                    else:
                        num_yes_gap_tmp = 0

                    num_yes += num_yes_tmp
                    num_yes_gap += num_yes_gap_tmp

                num_yes, num_yes_gap = self.consensus(
                    num_yes,
                    num_yes_gap,
                    options,
                    round_num,
                    turn_num,
                    turn_name,
                    round_name,
                    cache=cache,
                )

                # last_turn_name = turn_name
                turn_num += 1
                if num_yes == 0 and num_yes_gap == 0:
                    self.round_opinions[round_num + 1] = self.turn_comments[
                        turn_num - 1
                    ]
                    break
                elif num_yes_gap != 0:
                    continue

                # if status == "break":
                #     self.round_opinions[round_num + 1] = self.turn_comments[turn_num]
                #     break
                # elif status == "continue":
                #     continue
                if turn_num >= self.num_turns:
                    self.round_opinions[round_num + 1] = self.turn_comments[
                        turn_num - 1
                    ]
                    break

            print("^" * 50 + f" Round {round_num+1} " + "^" * 50)
            print_log(
                "^" * 50 + f" Round {round_num+1} " + "^" * 50, logger=self.logger
            )
            if num_yes == 0:
                break
        """
        "To provide a comprehensive and accurate response based on the given information, let's break down the key elements of the case:\n\n1. **Patient Demographics and Risk Factors**: \n   - Age 72.9 years (elderly patient)\n   - Male sex\n   - Poor performance status (ECOG 2) indicating significant functional limitations\n   - Significant smoking history (35 pack-years, ex-smoker)\n\n2. **Cancer Characteristics**:\n   - Site: Lip and Oral Cavity\n   - Subsite: Retromolar Trigone\n   - T Stage: T3 (locally advanced tumor)\n   - N Stage: N0 (no regional lymph node involvement)\n   - M Stage: M0 (no distant metastasis)\n   - Pathology: Squamous Cell Carcinoma\n   - HPV status: Not specified\n\n3. **Treatment and Follow-up**:\n   - Primary treatment modality: Radiotherapy alone\n   - No chemotherapy given\n   - Total radiation dose: 60 Gy in fractions of 25 Gy each\n\n4. **Outcome Indicators**:\n   - Second primary cancer history (2nd Ca: Y)\n   - Contrast-enhanced imaging details are not provided, but there is no mention of distant metastasis or other complications.\n\nConsidering the ECOG performance status at 2, which indicates significant functional impairment, along with advanced age and a smoking history, it's important to note that such factors can impact long-term survival. However, given the lack of information about specific treatment outcomes (such as local control, tumor recurrence, second primary cancer progression) and the absence of distant metastasis noted, we need to rely on the provided answer options.\n\nGiven the context and the options provided:\n\n**Final Answer: Status: Alive**\n\nThis conclusion is drawn from the fact that the patient has not been reported to be deceased despite significant risk factors, advanced disease stage (III), and the treatment approach which typically aims at controlling the cancer locally.\n\n"
        """
        # round_num = 0
        comment = {}
        final_answer = []
        for k, v in self.round_opinions[round_num + 1].items():
            answer = get_status(self.dataset_name, v)
            final_answer.append(f"({k.lower()}): {v}\n")
            comment[k] = answer

        final_answer = "".join(final_answer)
        self.cache.save_round_comment(f"Total", comment, self.agent_dict.keys(), None)
        print("-" * 50 + f" Discussion End " + "-" * 50)
        print_log("-" * 50 + f" Discussion End " + "-" * 50, logger=self.logger)
        self.panel.interaction_log = self.interaction_log
        num_cooperations = self.panel.plot_interaction()
        self.cache.do_save_messages(
            "num_cooperations", num_cooperations, "panel", False
        )

        return final_answer

    def ask_gap(self, agent, turn_num, turn_name, round_name):
        print("gap: ")
        num_yes_gap = 0
        # all_gap_response = []
        # if num_yes_gap != 0:
        #     for idx, v in enumerate(medical_agents):
        #         all_gaps = "".join(
        #             f"({k.lower()}): {v}\n" for k, v in round_gap.items()
        #         )
        #         summary_gap = v.chat(
        #             f"""A Knowledge gap is: {all_gaps} from  multiple expert discussion. Please indicate the **unique, useful, and specific expert recommendations**, and **exclude any of the following**:
        #             - Repeated recommendations (e.g., if multiple experts suggest the same type of expert, only keep it once).
        #             - Any recommendation that **does not clearly specify the type of expert needed**.
        #             - Recommendations that are **too vague or uninformative** (e.g., answers like "Yes" without further explanation).
        #             - If a recommendation is **only focused on psychological or emotional support**, exclude it **unless it provides unique insights directly relevant to medical management or clinical decision-making**.
        #             """
        #         )
        #         all_gap_response.append({v.role: summary_gap})

        if "gpt" in agent.model_info:
            gap = agent.temp_responses(
                f"You are part of the team: {self.agent_list}."
                "Earlier in this conversation, a set of discussion opinions from one or more medical experts in your team was provided. "
                "Please carefully review that information now.\n\n"
                "Based on your professional boundaries, determine whether there is a knowledge limitation or missing perspective that requires support from another specialist."
                "Please answer yes or no.\n"
                "If yes, specify the type of expert needed and provide a short reason.\n"
                "Be specific and consider the multidisciplinary needs involved in managing complex patient information (e.g., diagnostic imaging, supportive care, pathology review, and other medical expertise)."
                f"It is acceptable to recognize areas of expertise already covered by current team members (e.g., {self.agent_list})."
                "Do not recommend a specialist if their expertise is already represented in the team",
                # f"The following are the current team experts involved in the discussion:\n\n {self.agent_list}.",
                action=f"{round_name}-{turn_name}-ask_gap",
            )
        else:
            gap = agent.temp_responses(
                f"""You are part of the team: {self.agent_list}.
                    \nEarlier in this conversation, expert opinions from this team were shared. Your task now is to determine, based strictly on your professional scope, whether **any additional type of specialist** is needed to address a knowledge gap or decision-making limitation.
                    \nRespond using the exact structure below:
                    \n---
                    \n Answer: yes or no
                    \n Reason: a short reason
                    \n---
                    \nGuidelines:
                    \n- Do not suggest experts whose specialties already exist in the team.
                    \n- Do not explain, debate, or simulate internal reasoning.
                    \n- Do not justify the current team's sufficiency — only evaluate if new expertise is needed.
                    \n- Do not include any commentary outside the structured output.""",
                action=f"{round_name}-{turn_name}-ask_gap",
            )

        # if len(self.medical_agents) != 1:
        #     gap = agent.temp_responses(
        #         "For the question: '{self.question}', review last information, where given the opinions from other medical experts in your team, please assess the current discussion indicates"
        #         "a potential knowledge gap that requires consultation with an additional expert outside the current team. Please answer (yes/no)\n\n"
        #         "If Yes, briefly specify the type of expert needed with a short reason."
        #     )
        # else:
        #     gap = agent.temp_responses(
        #                 (
        #                     f"For the question: '{self.question}', please assess the current discussion indicates which key areas of analysis are missing and whether there are any significant missing areas would benefit from additional expert consultation."  # (e.g., pathology, radiological imaging, clinical evaluation, laboratory testing, genetic analysis, etc.).
        #                     "Please answer with 'yes' or 'no' for each aspect.\n\n"
        #                     "If either aspect is missing and you believe additional expertise is required, specify the type of expert needed and give a brief reason.\n\n"
        #                     "Discussion opinions:\n{}"
        #                 ).format(assessment if n == 1 else all_comments)
        #             )
        # agent.messages.pop()
        # agent.messages.pop()
        self.round_gap[turn_num][agent.role] = gap
        if "yes" in gap.lower().strip():
            num_yes_gap += 1

        return num_yes_gap

    def respond_gap(
        self,
        num_yes,
        assessment,
        round_num,
        turn_num,
        turn_name,
        round_name,
        cache=None,
    ):
        all_gaps = "".join(
            f"({k.lower()}): {v}\n" for k, v in self.round_gap[turn_num].items()
        )
        all_gaps_response = self.assistant.temp_responses(
            f"A Knowledge gap is: {all_gaps} for the question:\n {self.question}\n\n"
            f"Considering the medical question, the options for the discussion and the existing expert team {self.agent_list}, "
            f"the following text contains multiple expert recommendations for additional consultations. Your task is to extract only the **unique, useful, and specific expert recommendations**, and **exclude any of the following**:\n"
            "- Repeated recommendations (e.g., if multiple experts suggest the same type of expert, only keep it once)."
            "- Any recommendation that **does not clearly specify the type of expert needed**."
            "- If a recommendation is **only focused on psychological or emotional support**, exclude it **unless it provides unique insights directly relevant to clinical decision-making**.",  # medical management or
            action=f"{round_name}-{turn_name}-gap_identified",
        )
        recruited = self.recruiter.temp_responses(
            f"Rules:\n"
            "- Do **not** suggest removing, substituting, or duplicating existing experts. Only add **new** experts if necessary.\n\n"
            f"Knowledge gap identified: {all_gaps_response}\n\n"
            f"Medical question:\n {self.question}\n\n"
            f"Existing expert team (DO NOT DUPLICATE): {self.agent_list}\n\n"
            f"Your task: \nConsidering the medical question, discussion options and the current expert team {self.agent_list}, "
            f"identify any that require recruiting **new types of experts** to ensure an accurate decision (exclude {self.agent_list})."
            f"\n\nYou also need to clearly specify the communication structure between experts (e.g. Targeted Therapy Expert -> Medical Oncologist, Medical Oncologist ==  Radiation Oncologist)"
            f"or indicate if the new expert(s) will work independently.\n\n"
            f"Do **not** suggest removing, substituting, or duplicating existing experts. Only add **new** experts if necessary."
            f"\nFormat example if recruiting experts:"
            f"\n1. Medical Oncologist - Your expertise is strictly limited to systemic therapy decisions, including chemotherapy and immunotherapy in head and neck cancers. - Hierarchy: Independent"
            f"\n2. Other Medical Experts."
            f"\n\nYour answer must conform exactly to the format above. If the existing expert team comprehensively have covered the necessary expertise for accurate decision, answer: <skip recruitment>",
            action=f"{round_name}-{turn_name}-gap_recruit",
        )

        if "skip" in recruited:

            for _, agent in self.agent_dict.items():
                agent.messages[-1]["content"] = recruited

            self.update_with_discussion(turn_num, [])
            return num_yes, 0

        agents_info = [
            agent_info.split(" - Hierarchy: ")
            for agent_info in recruited.split("\n")
            if agent_info
        ]
        agents_data = [
            (info[0], info[1]) if len(info) > 1 else (info[0], None)
            for info in agents_info
        ]

        agent_list_new, agent_dict, new_medical_agents = parse_mpt(
            self.dataset_name,
            self.patient_id,
            agents_data,
            self.img_path,
            self.bbox_coords,
            self.logger,
            len(self.medical_agents),
            cache=self.cache,
        )

        for tmp_n in range(1, self.num_rounds + 2):
            for tmp_turn_num in range(self.num_turns):
                for source_agent_num in range(
                    1, len(self.medical_agents) + len(new_medical_agents) + 1
                ):
                    if source_agent_num < len(self.medical_agents) + 1:
                        self.interaction_log[f"Round {tmp_n}"][
                            f"Turn {tmp_turn_num+1}"
                        ][f"Agent {source_agent_num}"].update(
                            **{
                                f"Agent {target_agent_num}": None
                                for target_agent_num in range(
                                    len(self.medical_agents) + 1,
                                    len(self.medical_agents)
                                    + len(new_medical_agents)
                                    + 1,
                                )
                            }
                        )
                    else:
                        self.interaction_log[f"Round {tmp_n}"][
                            f"Turn {tmp_turn_num+1}"
                        ][f"Agent {source_agent_num}"] = {
                            **{
                                f"Agent {target_agent_num}": None
                                for target_agent_num in range(
                                    1,
                                    len(self.medical_agents)
                                    + len(new_medical_agents)
                                    + 1,
                                )
                            },
                            "VP": None,
                        }

        # try:
        (
            self.turn_comments[turn_num],
            self.interaction_log[f"Round {round_num+1}"][f"Turn {turn_num+1}"],
        ) = generate_assessment(
            self.args,
            self.dataset_name,
            self.question,
            self.answer_template,
            agent_dict,
            action=f"{round_name}-{turn_name}-gap_recruited_assessment",
            round_opinions=self.turn_comments[turn_num],
            interaction_log=self.interaction_log[f"Round {round_num+1}"][
                f"Turn {turn_num+1}"
            ],
            fewshot_examplers=assessment,
            num_agents=len(self.medical_agents),
            img_path=self.img_path,
        )
        # except Exception as e:
        #     print(agents_data, e)
        #     if num_yes == 0:
        #         return
        #         # break
        #     else:
        #         return
        #         # continue

        assessment = [
            get_status(self.dataset_name, v)
            for _, v in self.turn_comments[turn_num].items()
        ]
        options = "".join(
            f"({k.lower()}): {v}\n" for k, v in self.turn_comments[turn_num].items()
        )
        if len(set(assessment)) != 1:
            assessment = options
        start_idx = len(self.medical_agents)
        self.medical_agents.extend(new_medical_agents)
        self.agent_dict.update(agent_dict)
        self.agent_list += agent_list_new
        self.panel.num_agents = len(self.medical_agents)
        for idx, _agent in enumerate(new_medical_agents, start=start_idx):
            all_comments = "".join(
                f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n"
                for _k, _v in self.interaction_log[round_name][turn_name].items()
            )
            num_yes_tmp = self.discuss(
                f"{round_name}-{turn_name}-gap_recruited_discuss",
                idx,
                _agent,
                turn_num,
                turn_name,
                round_name,
                assessment,
                all_comments,
            )
            num_yes += num_yes_tmp

        if num_yes != 0:
            tmp_final_answer = {}
            comment = {}
            for i, agent in enumerate(self.medical_agents):
                response = agent.temp_responses(
                    f"Now that you've interacted with other medical experts, "
                    f"remind your expertise and the comments from other experts "
                    f"and make your final answer to the given question:\n{self.question}\n{self.answer_template}",
                    action=f"{round_name}-{turn_name}-gap_recruited_assessment",
                )
                tmp_final_answer[agent.role] = response
                try:
                    answer = get_status(self.dataset_name, response)
                    self.interaction_log[f"Round {round_num+1}"][f"Turn {turn_num+1}"][
                        f"Agent {i+1}"
                    ]["VP"] = answer
                    comment[agent.role] = answer
                except Exception as e:
                    print(self.interaction_log[f"Round {round_num+1}"]["Turn 1"])
                    raise e
            self.turn_comments[turn_num + 1] = tmp_final_answer
            self.cache.save_round_comment(
                f"Round {round_num+1} - Turn {turn_num+1}",
                comment,
                self.agent_dict.keys(),
                None,
            )
        else:
            self.turn_comments[turn_num + 1] = self.turn_comments[turn_num]
            # if num_yes == 0:
        #     round_opinions[n+1] = round_opinions[n]
        #     break

        if turn_num == self.num_turns - 1:
            self.update_with_discussion(turn_num, new_medical_agents)

        return num_yes, 0

    def update_with_discussion(self, turn_num, new_medical_agents):
        self.num_turns += 1
        self.round_gap[turn_num + 1] = {
            role: None for role in self.agent_dict.keys()
        }  # from 0
        self.turn_comments[self.num_turns] = {}  # from 1
        for tmp_n in range(1, self.num_rounds + 2):
            self.interaction_log[f"Round {tmp_n}"][f"Turn {self.num_turns}"] = {}
            for source_agent_num in range(1, len(self.medical_agents) + 1):
                self.interaction_log[f"Round {tmp_n}"][f"Turn {self.num_turns}"][
                    f"Agent {source_agent_num}"
                ] = {
                    **{
                        f"Agent {target_agent_num}": None
                        for target_agent_num in range(
                            1,
                            len(self.medical_agents) + len(new_medical_agents) + 1,
                        )
                    },
                    "VP": None,
                }

    def discuss(
        self,
        action,
        idx,
        agent,
        turn_num,
        turn_name,
        round_name,
        assessment,
        all_comments,
    ):

        num_yes = 0
        # num_yes_gap = 0
        # assessment = "".join(f"({k.lower()}): {v}\n" for k, v in self.round_opinions[n].items())

        # for idx, v in enumerate(self.medical_agents):

        print("#" * 50 + f" Agent {idx+1} - {agent.role} " + "#" * 50)
        participate = agent.temp_responses(
            "Earlier in this conversation, a set of discussion opinions from other medical experts in your team was provided. "
            "Please do not forget those earlier opinions. Now, additional new opinions have been provided. "
            "Considering both the earlier and the latest opinions together, please indicate whether you want to talk to any additional expert (yes/no). "
            "Opinions:\n{}".format(assessment if turn_num == 0 else all_comments),
            action=f"{round_name}-{turn_name}-{action}-participate",
        )

        if "yes" in participate.lower().strip():
            chosen_expert = agent.temp_responses(
                f"Enter the number of the expert you want to talk to:\n{self.agent_list}\nFor example,"
                f"if you want to talk with Agent 1. Pediatrician, return just 1."
                f"If you want to talk with more than one expert, please return 1,2 and don't return the reasons.",
                action=f"{round_name}-{turn_name}-{action}-talk",
            )

            chosen_experts = [
                int(ce)
                for ce in chosen_expert.replace(".", ",").split(",")
                if ce.strip().isdigit()
            ]

            for ce in chosen_experts:
                specific_question = agent.temp_responses(
                    f"Please remind your medical expertise and "
                    f"then leave your opinion to an expert you chose (Agent {ce}. {self.medical_agents[ce-1].role}). "
                    f"You should deliver your opinion once you are confident enough "
                    f"and in a way to convince other expert with a short reason.",
                    action=f"{round_name}-{turn_name}-{action}-comment",
                )

                print_log(
                    f" Agent {idx+1} ({self.medical_agents[idx].role}) -> "
                    f"Agent {ce} ({self.medical_agents[ce-1].role}) : {specific_question}",
                    self.logger,
                )  # ({agent_emoji[idx]}
                print(
                    f" Agent {idx+1} ({self.medical_agents[idx].role}) -> "
                    f"Agent {ce} ({self.medical_agents[ce-1].role}) : {specific_question}",
                )
                self.interaction_log[round_name][turn_name][f"Agent {idx+1}"][
                    f"Agent {ce}"
                ] = specific_question

            num_yes += 1
        else:
            # self.interaction_log[round_name][
            #         turn_name
            #     ] = {
            #     source_agent: {target_agent: "No" for target_agent in target_agents}
            #     for source_agent, target_agents in self.interaction_log[round_name][
            #         turn_name
            #     ].items()
            # }

            print(f" Agent {idx+1} ({agent.role}): No")
            print_log(f" Agent {idx+1} ({agent.role}): No", logger=self.logger)

        return num_yes

    def consensus(
        self,
        num_yes,
        num_yes_gap,
        assessment,
        round_num,
        turn_num,
        turn_name,
        round_name,
        cache=None,
    ):
        if num_yes == 0 and num_yes_gap == 0:
            # self.turn_comments[turn_num + 1] = self.turn_comments[turn_num]
            return num_yes, num_yes_gap  # "break"
        elif num_yes_gap != 0:
            num_yes, num_yes_gap = self.respond_gap(
                num_yes,
                assessment,
                round_num,
                turn_num,
                turn_name,
                round_name,
                cache=cache,
            )
            return num_yes, num_yes_gap
        else:
            tmp_final_answer = {}
            comment = {}
            for i, agent in enumerate(self.medical_agents):
                response = agent.temp_responses(
                    f"Now that you've interacted with other medical experts, "
                    f"remind your expertise and the comments from other experts "
                    f"and make your final answer to the given question:\n{self.question}\n{self.answer_template}",
                    action=f"{round_name}-{turn_name}-assessment",
                )
                tmp_final_answer[agent.role] = response
                try:
                    answer = get_status(self.dataset_name, response)
                    self.interaction_log[f"Round {round_num+1}"][f"Turn {turn_num+1}"][
                        f"Agent {i+1}"
                    ]["VP"] = answer
                    comment[agent.role] = answer
                except Exception as e:
                    # RADCURE-0920
                    print(self.interaction_log)
                    print(len(self.medical_agents))
                    raise e
            self.turn_comments[turn_num + 1] = tmp_final_answer
            self.cache.save_round_comment(
                f"Round {round_num+1} - Turn {turn_num+1}",
                comment,
                self.agent_dict.keys(),
                None,
            )

        return num_yes, num_yes_gap


def parse_mpt(
    dataset_name,
    patient_id,
    agents_data,
    img_path=None,
    bbox_coords=None,
    logger=None,
    num_agents=0,
    cache=None,
):

    for idx, agent in enumerate(agents_data):
        idx += num_agents
        try:
            print_log(
                f"Agent {idx+1} ({agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}",
                logger=logger,
            )
        except:
            print_log(
                f"Agent {idx+1}: {agent[0]}", logger=logger
            )  # ({agent_emoji[idx]})

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        i += num_agents
        try:
            agent_role = agent[0].split("-")[0].split(".")[1].strip().lower()
            description = agent[0].split("-")[1].strip().lower()
            agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
        except:
            print(str(agents_data))
            break

    if agent_list == "":
        return "nan"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split("-")[0].split(".")[1].strip().lower()
            description = agent[0].split("-")[1].strip().lower()
        except:
            continue

        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(
            instruction=inst_prompt,
            role=agent_role,
            meta=patient_id,
            logger=logger,
            img_path=img_path,
            cache=cache,
        )
        if dataset_name == "radcure":
            if "gpt" in _agent.model_info:
                vision_prompt = (
                    "\nYou will be provided with a head and neck CT scan that includes a masked region of interest (ROI). "
                    "Alongside the scan, one or more 3D bounding box coordinates will be supplied, each defining specific volumetric regions within the scan. These may correspond to organs, pathological regions, or cellular structures. "
                    "Each bounding box is defined by its minimum and maximum values along the z, y, and x axes, normalized relative to the original image size.\n\n"
                    f"The given bounding box coordinates are: {str(bbox_coords)}.\n\n"
                    "**Task Instructions:**\n"
                    "1. **Initial Assessment:**\n Carefully analyze the CT scan image (without using the bounding box data). Describe any visible anatomical structures, patterns, abnormalities, and and note the characteristics of the masked region of interest (ROI). "
                    "Do not use the bounding box data at this stage.\n"
                    "2. **Mapping Bounding Boxes:**\n Consider the bounding box coordinates and map them to the corresponding areas within the scan.\n"
                    "3. **Clinical Reasoning:**\n Summarize the patient's clinical context and findings in a clear, structured bullet-point format and reason through the patient's condition step by step.\n"
                    "3. **Integrated Conclusion:**\n Combine your findings from the image analysis, bounding box mapping, and masked ROI to concisely synthesize your final clinical impression.\n\n"
                    "Be thorough and precise in both your image-based observations and your clinical reasoning."
                )

            else:
                vision_prompt = (
                    "\nYou will be provided with a head and neck CT scan that includes one or more masked region of interest (ROI). "
                    "Alongside the scan, one or more 3D bounding box coordinates will be supplied, each defining specific volumetric regions within the scan."
                    "These coordinates identify either organs, disease regions, or cellular structures. "
                    "Each bounding box is defined by its minimum and maximum values along the z, y, and x axes, and is normalized relative to the original image size.\n\n"
                    f"The given bounding box coordinates are: {str(bbox_coords)}.\n\nDescribe any visible anatomical structures, patterns, or abnormalities. "
                )

            _agent.vision_prompt = vision_prompt
            _agent.vision_prompt_part = (
                "**Task Instructions:**\n"
                "1. **Initial Assessment:**\n Carefully analyze the CT scan image (without using the bounding box data). Describe any visible anatomical structures, patterns, abnormalities, and and note the characteristics of the masked region of interest (ROI). "
                "Do not use the bounding box data at this stage.\n"
                "2. **Mapping Bounding Boxes:**\n Consider the bounding box coordinates and map them to the corresponding areas within the scan.\n"
                "3. **Clinical Reasoning:**\n Summarize the patient's clinical context and findings in a clear, structured bullet-point format and reason through the patient's condition step by step.\n"
                "3. **Integrated Conclusion:**\n Combine your findings from the image analysis, bounding box mapping, and masked ROI to concisely synthesize your final clinical impression.\n\n"
                # "Your analysis should be thorough, but your wording brief and to the point.\n"
                "Be thorough and precise in both your image-based observations and your clinical reasoning."
            )
        
        elif dataset_name == "pancreatic_cancer":

            _agent.vision_prompt = f"""
                You will be provided with a pancreatic CT scan that includes a masked region of interest (ROI).
                Alongside the scan, one or more 3D bounding box coordinates will be supplied, each defining specific volumetric regions within the scan. These may correspond to organs, pathological regions, or cellular structures.
                Each bounding box is defined by its minimum and maximum values along the z, y, and x axes, normalized relative to the original image size.
                The given bounding box coordinates are: {str(bbox_coords)}.
                Task Instructions:
                1. Initial Assessment: Carefully analyze the CT scan image (without using the bounding box data). Describe any visible anatomical structures, patterns, abnormalities, and and note the characteristics of the masked region of interest (ROI).
                Do not use the bounding box data at this stage.
                2. Mapping Bounding Boxes: Consider the bounding box coordinates and map them to the corresponding areas within the scan.
                3. Clinical Reasoning: Summarize the patient's clinical context and findings in a clear, structured bullet-point format and reason through the patient's condition step by step.
                4. Integrated Conclusion: Combine your findings from the image analysis, bounding box mapping, and masked ROI to concisely synthesize your final clinical impression.
                Be thorough and precise in both your image-based observations and your clinical reasoning.
            """

            _agent.vision_prompt_part = f"""
                Task Instructions:
                1. Initial Assessment: Carefully analyze the CT scan image (without using the bounding box data). Describe any visible anatomical structures, patterns, abnormalities, and and note the characteristics of the masked region of interest (ROI).
                Do not use the bounding box data at this stage.
                2. Mapping Bounding Boxes: Consider the bounding box coordinates and map them to the corresponding areas within the scan.
                3. Clinical Reasoning: Summarize the patient's clinical context and findings in a clear, structured bullet-point format and reason through the patient's condition step by step.
                4. Integrated Conclusion: Combine your findings from the image analysis, bounding box mapping, and masked ROI to concisely synthesize your final clinical impression.
                Be thorough and precise in both your image-based observations and your clinical reasoning.
            """
        
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    print_tree(hierarchy_agents[0], horizontal=False)

    return agent_list, agent_dict, medical_agents


# extract alive
def process_intermediate_query(
    question,
    patient_id,
    examplers=None,
    logger=None,
    bbox_coords=None,
    img_path=None,
    df_pred_masked=None,
    args=None,
    answer="",
):

    cache = CachedBase()
    cache.init(patient_id)
    cache.do_save_messages("question", question + f"\nAnswer: {answer}", "", False)
    # use_cached = args.use_cached #and patient_id in args.assessments.keys()
    logger.info("[INFO] Step 1. Expert Recruitment")
    print("[INFO] Step 1. Expert Recruitment")
    round_opinions = {
        n: {} for n in range(1, args.num_rounds + 2)
    }  # {t: {} for t in range(1, args.num_tuns+1)}

    recruiter, agents_data, fewshot_examplers = recruit_stage(
        args.dataset_name,
        patient_id,
        question,
        args.num_agents,
        logger,
        use_cached=True,
        cache=cache,
        examplers=examplers,
    )

    # if use_cached:
    #     panel, messages, agents_data, round_opinions[1] = load_initial_assessment(
    #         args, patient_id, logger
    #     )
    #     recruiter.messages.append({"role": "assistant", "content": messages})
    # else:
    panel = MDTPanel(
        patient_id, len(agents_data), args.num_turns, args.num_rounds, logger=logger
    )

    logger.info("[INFO] Step 2. Collaborative Decision Making")
    logger.info("[INFO] Step 2.1. Hierarchy Selection")
    print("[INFO] Step 2. Collaborative Decision Making")
    print("[INFO] Step 2.1. Hierarchy Selection")

    agent_list, agent_dict, medical_agents = parse_mpt(
        args.dataset_name,
        patient_id,
        agents_data,
        img_path,
        bbox_coords,
        logger,
        cache=cache,
    )

    # round_answers = {n: None for n in range(1, num_rounds + 1)}

    #########################
    round_opinions[1], panel.interaction_log["Round 1"]["Turn 1"] = generate_assessment(
        args,
        args.dataset_name,
        question,
        args.answer_template,
        agent_dict,
        action="initial_assessment",
        round_opinions=round_opinions[1],
        interaction_log=copy.deepcopy(panel.interaction_log["Round 1"]["Turn 1"]),
        logger=logger,
        img_path=img_path,
        use_cached=True,
        fewshot_examplers=fewshot_examplers,
    )

    # if not use_cached:
    #     save_initial_assessment(args, patient_id, round_opinions[1], agents_data)
    print("")
    print_log("[INFO] Step 2.2. Participatory Debate", logger=logger)

    #########################
    discuss = DiscussionPanel(
        patient_id=patient_id,
        question=question,
        answer_template=args.answer_template,
        agent_list=agent_list,
        agent_dict=agent_dict,
        medical_agents=medical_agents,
        round_opinions=round_opinions,
        panel=panel,
        recruiter=recruiter,
        logger=logger,
        img_path=img_path,
        bbox_coords=bbox_coords,
        df_pred_masked=df_pred_masked,
        args=args,
        cache=cache,
    )
    final_answer = discuss.start_discuss()

    logger.info("\n[INFO] Step 3. Final Decision")
    print("\n[INFO] Step 3. Final Decision")

    moderator = Agent(
        "You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.",
        "Moderator",
        model_info="gpt-4.1-mini",
        meta=patient_id,
        cache=cache,
    )

    _decision = moderator.temp_responses(
        f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote."
        f"Only output your final answer should be like below format:\n{args.final_answer_template}\n{final_answer} \n\nQuestion: {question}",
        img_path=None,
        action="decision",
    )
    final_decision = {"majority": _decision}
    print_log(
        f"{patient_id} - moderator's final decision (by majority vote): {_decision}",
        logger=logger,
    )
    print(f"{patient_id} - moderator's final decision (by majority vote): {_decision}")
    print("")

    # _decision = moderator.temp_responses(
    #     f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question."
    #     f"Only output your final answer should be like below format:\n{args.final_answer_template}\n{final_answer} \n\nQuestion: {question}",
    #     img_path=None,
    #     action="decision",
    # )
    # final_decision = {"consensus": _decision}

    del cache.saved_messages

    return final_decision["majority"]  # RADCURE-0904 13


def get_status(dataset_name, response_tasks):

    if dataset_name == "radcure" or dataset_name == "pancreatic_cancer":
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
    elif dataset_name == "med_qa":
        match = re.search(
            r"answer:\s*\(?([a-z])\)?",
            response_tasks.lower()
            .replace("final answer:\n\n", "")
            .replace(" ", "")
            .replace("\n", "")
            .replace("*", ""),
        )
        if match:
            return match.group(1)
    else:
        return response_tasks


def interact(args, case, results, number, patient_id, df_pred=None, logger=None):
    # args.use_cached = os.path.exists(f"{os.environ["work_dir"]}/radcure_basic/{patient_id}.json")

    examplers = None

    if args.dataset_name in ["med_qa"]:
        case = create_question(case, args.dataset_name)
        examplers = args.examplers
        answer = ""
    else:
        case = case["question"][0]
        answer = ""

    print("*" * 100)
    print_log("*" * 100, logger=logger)
    print(f"{patient_id} - {case}")
    print_log(f"{patient_id} - {case}", logger=logger)
    print("")

    if args.vlm:
        if args.dataset_name == "radcure":
            path = Path(r"data\RADCURE\{patient_id}")
            masked_image, bbox_coords = get_masked_image(path)
        elif args.dataset_name == "pancreatic_cancer":
            path = Path(f"data/PC/{patient_id}")
            masked_image, bbox_coords = get_masked_image(path)
        else:
            raise NotImplementedError
    else:
        masked_image = None
        bbox_coords = None

    response_tasks = process_intermediate_query(
        case,
        patient_id,
        examplers=examplers,
        logger=logger,
        bbox_coords=bbox_coords,
        img_path=masked_image,
        df_pred_masked=None,
        args=args,
        answer=answer,
        # bbox_coords=None,
        # img_path=None,
    )

    logger.info("number: ", number, "patient_id: ", patient_id, "-", response_tasks)

    status = get_status(args.dataset_name, response_tasks)

    results.append({"number": number, "patient_id": patient_id, "status": status})
    results = sorted(results, key=lambda x: x["number"])
    print(str(len(results)))
    with open(
        f"{args.work_dir}/{os.environ["saved_as_path"]}_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            # {"number": number, "patient_id": patient_id, "status": status},
            f,
            ensure_ascii=False,
            indent=4,
        )

    logger.info("#" * 100)
    # time.sleep(1)

    return status


def determine_difficulty(question, difficulty, patient_id):
    if difficulty != "adaptive":
        return difficulty
    cache = CachedBase()
    cache.init(patient_id)
    cache.do_save_messages("question", question + f"\nAnswer: ", "", False)

    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""

    medical_agent = Agent(
        instruction="You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.",
        role="medical expert",
        meta=patient_id,
        cache=cache,
    )

    response = medical_agent.temp_responses(
        difficulty_prompt, action="determine_difficulty"
    )

    if "basic" in response.lower() or "1)" in response.lower():
        return "basic"
    elif "intermediate" in response.lower() or "2)" in response.lower():
        return "intermediate"
    elif "advanced" in response.lower() or "3)" in response.lower():
        return "advanced"


def objective(
    trial,
    args,
    results,
    patient_list,
    patient_number,
    df_pred=None,
    pbar=None,
    logger=None,
):
    number = trial.suggest_int("patient_id", 0, len(patient_list)-1)
    patient = patient_list[number]

    if "patient_id" not in patient.keys():
        patient_id = f"{args.dataset_name}-{number}"
    else:
        patient_id = patient["patient_id"]

    status = interact(
        args, patient, results, number, patient_id, df_pred=df_pred, logger=logger
    )
    pbar.update(1)

    return status


def get_masked_image(subsubdir):
    # scale_factor = [1.0, 0.25, 0.125]
    scale_factor = [1.0, 1.0, 1.0]
    target = list(subsubdir.rglob("*ct.nii.gz"))[0]
    # print(target)
    mask_files = list(subsubdir.rglob("*tumor.nii.gz"))
    dicom_slices = sitk.GetArrayFromImage(sitk.ReadImage(str(target)))
    # dicom_slices = dicom_slices[:, ::-1, ::-1]
    mask_list = [
        sitk.GetArrayFromImage(sitk.ReadImage(mask_file)).transpose(0, 1, 2)
        for mask_file in mask_files
    ]

    x_min_between = 0
    x_max_between = dicom_slices.shape[2]
    x_idx = (x_min_between + x_max_between) // 2
    for mask in mask_list:
        mask = mask > 0  # 设置阈值以创建 mask
        coordinates = np.argwhere(mask)
        if coordinates.size > 0:
            z_min, y_min, x_min = coordinates.min(axis=0)
            z_max, y_max, x_max = coordinates.max(axis=0)
            x_min_between = max(x_min_between, x_min)
            x_max_between = min(x_max_between, x_max)

    if x_min_between >= x_max_between:
        print("No mask shared in the same slices")
        x_min_between = 0
        x_max_between = mask.shape[2]

    metas = []
    # x_min_ = []
    for idx, mask in enumerate(mask_list):
        # mask = sitk.GetArrayFromImage(
        #     sitk.ReadImage(mask_files[idx])
        # ).transpose(0, 1, 2)
        # # mask = mask[:, ::-1, ::-1]

        mask = mask > 0 
        areas = [mask[:, :, idx].sum() for idx in range(x_min_between, x_max_between)]
        x_idx = np.argmax(areas) + x_min_between
        mask2d = mask[:, :, x_idx]
        coordinates = np.argwhere(mask2d)
        if coordinates.size > 0:
            z_min, y_min = coordinates.min(axis=0)
            z_max, y_max = coordinates.max(axis=0)
            bbox = (x_idx, y_min, z_min, x_idx, y_max, z_max)
            bbox = [int(v) for v in bbox]
            print(
                f"{os.path.basename(mask_files[idx])}, Bounding Box coordinates: {[
                    int(round(x_min * scale_factor[0])),
                    int(round(y_min * scale_factor[1])),
                    int(round(z_min * scale_factor[2])),
                    int(round(x_max * scale_factor[0])),
                    int(round((y_max + 1) * scale_factor[1])),
                    int(round((z_max + 1) * scale_factor[2])),
                    # int(round((y_min + bbox_width) * scale_factor)),
                    # int(round((z_min + bbox_height) * scale_factor)),
                ]}, {
                areas[x_idx - x_min_between]}"
            )
        else:  # Note that RADCURE-1486 RADCURE-1574 RADCURE-1759
            coordinates = np.argwhere(mask)
            if coordinates.size > 0:
                z_min, y_min, x_min = coordinates.min(axis=0)
                z_max, y_max, x_max = coordinates.max(axis=0)
                bbox = (x_min, y_min, z_min, x_max, y_max, z_max)
                bbox = [int(v) for v in bbox]
                print(
                    f"{os.path.basename(mask_files[idx])}, Bounding Box coordinates: {bbox}, {areas[x_idx - x_min_between]}"
                )
            else:
                bbox = "nan"

            with open("mask_error.txt", "a+") as f:
                f.write(str(subsubdir) + "\n")
        metas.append(
            {
                "mask": mask,
                "bbox": bbox,
                "name": os.path.basename(mask_files[idx]),
            }
        )
    bbox_list = []
    masked_image = dicom_slices[:, :, x_idx]
    if any([v != 1.0 for v in scale_factor]):
        masked_image = downsample_images(masked_image, scale_factor[1:])
    print(f"size: {masked_image.shape}")
    for meta in metas:
        # fig = plt.figure()
        bbox = meta["bbox"]
        mask = meta["mask"]
        if bbox != "nan":
            (x_min, y_min, z_min, x_max, y_max, z_max) = bbox
        else:
            x_min = dicom_slices.shape[2] // 2
        mask_2d = mask[:, :, x_min]

        mask_2d = downsample_images(mask_2d.astype(np.float32), scale_factor[1:])
        mask_2d = mask_2d > 0

        # bbox_width = y_max - y_min
        # bbox_height = z_max - z_min
        if bbox != "nan":
            bbox = [
                int(round(x_min * scale_factor[0])),
                int(round(y_min * scale_factor[1])),
                int(round(z_min * scale_factor[2])),
                int(round(x_max * scale_factor[0])),
                int(round((y_max + 1) * scale_factor[1])),
                int(round((z_max + 1) * scale_factor[2])),
                # int(round((y_min + bbox_width) * scale_factor)),
                # int(round((z_min + bbox_height) * scale_factor)),
            ]
        else:
            bbox = []

        random_color = [random.random() for _ in range(3)]
        masked_image = apply_mask(masked_image, mask_2d, random_color)
        bbox_list.append(bbox)

    return masked_image, bbox_list


def parse_args():
    parser = argparse.ArgumentParser(description="Run the inference pipeline for clinical datasets.")

    parser.add_argument(
        "--dataset_name",
        default="pancreatic_cancer",
        choices=["radcure", "pancreatic_cancer", "med_qa"],
        help="Dataset to evaluate. Options: 'radcure', 'pancreatic_cancer', 'med_qa'."
    )

    parser.add_argument(
        "--model_name",
        default="gpt-4.1-mini",
        help="Model identifier to be used for inference."
    )

    parser.add_argument(
        "--prefix",
        default="gpt-4.1-mini_kamac",
        help="Experiment name prefix used for logging and result saving."
    )

    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Enable vision-language mode (for image-text datasets)."
    )

    parser.add_argument(
        "--cot",
        action="store_true",
        help="Enable Chain-of-Thought (CoT) reasoning."
    )

    parser.add_argument(
        "--auto_recruit",
        action="store_true",
        help="Automatically recruit agents for specific tasks (default: True)."
    )

    parser.add_argument(
        "--add_examplers",
        action="store_true",
        help="Use few-shot exemplars in the prompt (few-shot learning)."
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=1,
        help="Number of agents involved in the decision process."
    )

    parser.add_argument(
        "--resampling_mode",
        choices=["specific_ids", "all_specific_ids"],
        default="specific_ids",
        help="Select how patient cases are loaded: specific IDs or all."
    )

    parser.add_argument(
        "--specific_ids",
        nargs="+",
        default=[],
        help="List of specific patient IDs to evaluate."
    )

    parser.add_argument(
        "--cache_mode",
        default="all",
        type=str,
        help="List of specific patient IDs to evaluate."
    )

    parser.add_argument(
        "--cache_path",
        default="results/your_cache_path",
        type=str,
        help="List of specific patient IDs to evaluate."
    )

    parser.add_argument(
        "--saved_as_path",
        default=None,
        type=str,
        help="List of specific patient IDs to evaluate.",
    )
    
    parser.add_argument(
        "--n_jobs",
        default=1,
        type=int,
        help="Number of jobs to run in parallel.",
    )

    return parser.parse_args()


def inference():
    args_cli = parse_args()

    work_dir = os.path.dirname(__file__)
    vlm = args_cli.vlm
    auto_recruit = args_cli.auto_recruit
    add_examplers = args_cli.add_examplers
    cot = args_cli.cot
    num_agents = args_cli.num_agents
    resampling_mode = args_cli.resampling_mode
    specific_ids = args_cli.specific_ids

    os.environ["cached_mode"] = (cache_mode := "all")  # type: ignore Literal["all", "skip", "none"]
    os.environ["work_dir"] = work_dir

    dataset_name = args_cli.dataset_name
    # dataset_name = "radcure"

    # if dataset_name == "radcure":
    #     os.environ["cached_action_keys"] = (
    #         ",".join(["question",
    #                 "recruit-recruiter",
    #                 "initial_assessment-radiation oncologist",
    #                 "initial_assessment-medical oncologist",
    #                 "initial_assessment-surgical oncologist (recurrence/secondary cancers)",
    #                 "initial_assessment-pathologist",
    #                 "initial_assessment-targeted therapy expert"]
    #     ))
    # os.environ["cached_action_keys"] = "recruit-recruiter"
    os.environ["cached_action_keys"] = "recruit-recruiter,initial_assessment-,examplers-"

    # os.environ["cached_action_keys"] = "all"

    # model_name_list = [
    # "deepseek-r1:7b",
    # "deepseek-r1:14b",
    # "deepseek-r1:32b",
    # "deepseek-r1:70b",
    # "llama3.3:70b",
    # "qwen2.5:7b",
    # "qwen2.5:14b",
    # "qwen2.5:32b",
    # "qwen2.5:72b",
    # "Qwen/Qwen2.5-32B-Instruct"
    # ]
    model_name = args_cli.model_name
    # model_name = ["qwen2.5:7b"]
    # model_name = ["claude-3-7-sonnet-20250219"]
    # model_name = ["deepseek-reasoner"][0]
    # model_name = [
    #     "gpt-4.1-mini,deepseek-reasoner,gpt-4o,claude-3-7-sonnet-20250219"
    # ]

    prefix = args_cli.prefix
    # prefix = "gpt-4.1-mini_1_kamac"
    # prefix = "deepseek-reasoner_1_CoT"

    os.environ["MODEL_INFO"] = model_name.replace("_", ":")
    os.environ["SOURCE"] = "openai"
    os.environ["cache_path"] = args_cli.cache_path
    os.environ["saved_as_path"] = args_cli.saved_as_path if args_cli.saved_as_path is not None else args_cli.cache_path
    os.makedirs(f"{work_dir}/{os.environ['cache_path']}", exist_ok=True)
    os.makedirs(f"{work_dir}/{os.environ['saved_as_path']}", exist_ok=True)

    specific_ids = []
    patient_list, examplers, _ = load_dataset(dataset_name, add_examplers=add_examplers)

    iters = 0
    df_pred = None
    try:
        with open(f"{work_dir}/{os.environ['saved_as_path']}_results.json", "r", encoding="utf-8") as results_handle:
            results = json.load(results_handle)
            missing_patient_id = set(
                [int(v["patient_id"].split("-")[1]) for v in patient_list]
            ) - set(
                [int(v["patient_id"].split("-")[1]) for v in results if "patient_id" in v.keys()]
            )

            # LOAD MISSING ITEMS
            missing_patient_id = list(missing_patient_id)
            missing_number_list = []
            if missing_patient_id:
                for idx, patient in enumerate(patient_list):
                    patient_id = int(patient["patient_id"].split("-")[1])
                    if patient_id in missing_patient_id:
                        missing_number_list.append(idx)

            nan_list = list(set(
                [v["number"] for v in results if "patient_id" in v.keys() and v["status"] == "nan"]
            ))

        if resampling_mode == "specific_ids":
            specific_ids = args_cli.specific_ids
        elif resampling_mode == "all_specific_ids":
            missing_number_list = [int(v["patient_id"].split("-")[1]) for v in patient_list]

        if specific_ids:
            if dataset_name == "med_qa":
                specific_ids = [int(number.split("-")[1]) if isinstance(number, str) else int(number) for number in specific_ids]

            results = [x for x in results if x["number"] in specific_ids]
            missing_number_list = [x for x in missing_number_list if x in specific_ids]
            nan_list = []

    except FileNotFoundError:
        user_result = input(f"Not Found {work_dir}/{os.environ['cache_path']}_results.json, y/n ")
        if user_result == "n":
            return
        results = []
        nan_list = []
        missing_number_list = []
        if not specific_ids:
            missing_number_list = list(range(len(patient_list)))

    if dataset_name == "med_qa" and specific_ids:
        specific_ids = [int(number.split("-")[1]) if isinstance(number, str) else int(number) for number in specific_ids]
    total_list = nan_list + missing_number_list + specific_ids
    pbar = tqdm(total=len(total_list))

    if not total_list:
        print("The experiment is completed")
    else:
        missing_entries = [(idx, patient_list[idx]["patient_id"]) for idx in missing_number_list]
        print(f"missing: {missing_entries}, {len(missing_entries)}")
        print(f"nan: {nan_list}")

        sampler = optuna.samplers.GridSampler({"patient_id": total_list})
        db_path = f"{work_dir}/{dataset_name}_{prefix}{model_name}_{iters}.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
        )

        study = optuna.create_study(
            study_name="ollama",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )

        if dataset_name == "radcure" or dataset_name == "pancreatic_cancer":
            final_answer_template = ("Answer: Alive or Dead",)
            answer_template = ("Answer:<Alive|Dead>",)
        elif dataset_name == "med_qa":
            answer_template = ("Answer: ",)
            final_answer_template = ("Answer: C) 2th pharyngeal arch\n",)

        args = SimpleNamespace(
            experimental_desc=f"{dataset_name}_{prefix}_test",
            debug=False,
            source="ollama",
            model_info=model_name,
            work_dir=work_dir,
            num_agents=num_agents,
            num_rounds=1, # Modify this to add more rounds
            num_turns=3, # Modify this to add more turns
            prefix=prefix,
            vlm=vlm,
            cot=cot,
            auto_recruit=auto_recruit,
            dataset_name=dataset_name,
            examplers=examplers,
            answer_template=answer_template,
            final_answer_template=final_answer_template,
        )
        args.log_file = f"{work_dir}/{os.environ['cache_path']}/{args.experimental_desc}.log"
        logger = create_logger(
            args.experimental_desc,
            work_dir=os.path.dirname(args.log_file),
            cfg=None,
        )
        logger.info(f"-" * 100)

        try:
            study.optimize(
                lambda trial: objective(
                    trial,
                    args,
                    results,
                    patient_list,
                    total_list,
                    df_pred,
                    pbar,
                    logger,
                ),
                n_trials=sampler._n_min_trials,
                n_jobs=args_cli.n_jobs,
            )
        except KeyboardInterrupt:
            command = input("Please input the command (q to exit): ")
            if command == "q":
                raise KeyboardInterrupt


if __name__ == "__main__":
    inference()
