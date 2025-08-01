import json
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    average_precision_score,
    precision_score,
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import pandas as pd
import copy
from typing import Literal


def remove_duplicates_by_key(dicts, key):
    seen = set()
    unique_dicts = []

    for d in dicts:
        value = d.get(key)
        if value not in seen and d["status"] != "nan":
            seen.add(value)
            unique_dicts.append(d)

    return unique_dicts


def multiclass_specificity(cm, labels=None):
    num_classes = cm.shape[0]
    specificity_per_class = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_per_class.append(specificity)

    return np.mean(specificity_per_class)


def calculate_mc_metrics_and_plot_roc(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    conf_matrix = confusion_matrix(y_true, y_pred)
    spec = multiclass_specificity(conf_matrix)

    report = classification_report(y_true, y_pred, digits=4)


    print("Accuracy:", accuracy)
    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("Spec (macro):", spec)
    print("F1 Score (macro):", f1)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)

    return accuracy, None, precision, None

def calculate_bc_metrics_and_plot_roc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mismatched_indices = np.where(y_true != y_pred)[0]
    y_scores = y_pred


    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # print(cm)
    print(TN, FP, FN, TP, end=" |")


    accuracy = (TP + TN) / (TP + TN + FP + FN)


    recall = TP / (TP + FN) if (TP + FN) != 0 else 0


    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0


    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    auc = 0.5 * (TN / (TN + FP) + recall)


    roc_auc = roc_auc_score(y_true, y_scores)


    ap = average_precision_score(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (TPR): {recall:.4f}, tpr: {tpr}")
    print(f"FPR: {FPR:.4f}, fpr: {fpr}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}, {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")



    # plt.figure()
    # plt.plot(
    #     fpr,
    #     tpr,
    #     color="blue",
    #     label=f"ROC curve (area = {roc_auc:.2f})",
    # )
    # plt.plot([0, 1], [0, 1], color="red", linestyle="--") 
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="lower right")
    # plt.grid()
    # plt.show()

    return accuracy, roc_auc, ap, mismatched_indices


def lists_are_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for idx, (a, b) in enumerate(zip(l1, l2)):
        if a != b:
            print(f"{idx}, {a}, {b}")
            return False
    return True


def load_radcure(exp_name):
    # model_name = "gpt-4.1-mini"
    # model_name = "deepseek-reasoner"
    # file_dir = work_dir  # os.path.dirname(__file__)
    # print(model_name)
    with open(os.environ["data_file"], "r", encoding="utf-8") as json_file:
        patient_list = json.load(json_file)
        choices = patient_list[0]["options"]
        # patient_list = [v for v in patient_list if "test" in v["question"][0]]
        patient_list_id = [v["patient_id"] for v in patient_list]
        gt = [choices[v["answer"]].lower() for v in patient_list]
        gt = np.where(np.array(gt) == "alive", 1, 0)

    with open(
        f"{exp_name}.json",
        "r",
        encoding="utf-8",
    ) as json_file:
        results = json.load(json_file)
        results = remove_duplicates_by_key(results, "patient_id")

    with open(
        f"{exp_name}.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    return results, patient_list, patient_list_id, gt


def letter_to_number(c):
    return ord(c.lower()) - ord("a")


def cal_metric(file,
                dataset_name: Literal["radcure", "med_qa", "path_vqa"],
                model_name: Literal["deepseek-reasoner", "gpt-4.1-mini"]):

    os.environ["data_file"] = r"data/radcure_test.json"
    error_show = False
    specific_ids = []

    if dataset_name == "radcure":
        results, gt_metas, patient_list_id, gt = load_radcure(file)

    elif dataset_name in ["med_qa", "path_vqa"]:
        results, gt_metas, patient_list_id, gt = load_dataset(
            file, dataset_name, model_name
        )

    num_samples = len(patient_list_id)

    if specific_ids:
        gt = [x for i, x in enumerate(gt) if i in specific_ids[:num_samples]]

    results = [v for v in results if v["patient_id"] in patient_list_id]

    if dataset_name == "radcure":
        pred = [v["status"].lower() for v in results if results[0]["patient_id"]]
        pred = np.where(np.array(pred) == "alive", 1, 0)
    elif dataset_name == "med_qa":
        gt = [letter_to_number(v.lower()) for v in gt[:num_samples]]
        # pred = [letter_to_number(v["status"].lower()) for v in results[:num_samples]]
        pred = []
        for v in results[:num_samples]:
            if v['status'].lower() not in ["a", "b", "c", "d", 'e']:
                print(v)
                raise ValueError
            pred.append(letter_to_number(v["status"].lower()))
    elif dataset_name == "path_vqa":
        pred = [v["status"].lower() for v in results if results[0]["patient_id"]]
        pred = np.where(np.array(pred) == "yes", 1, 0).tolist()

    if "nan" in pred:
        raise ValueError(pred.index("nan"))

    a = [v["patient_id"] for v in gt_metas]
    b = [v["patient_id"] for v in results]

    if not lists_are_equal(a, b):
        missing_patient_id = set(
            [int(v["patient_id"].split("-")[1]) for v in gt_metas]
        ) - set([int(v["patient_id"].split("-")[1]) for v in results])
        print(f"missing patient_id: {missing_patient_id}, {len(missing_patient_id)}")

        raise ValueError("The ordering is wrong")

    # b_number_list = list(range(len(patient_list)))
    # a_number_list = [int(v["number"]) for v in results]
    if not lists_are_equal(a, b):
        raise ValueError("The number is wrong")

    if dataset_name == "med_qa":
        accuracy, roc_auc, ap, mismatched_indices = calculate_mc_metrics_and_plot_roc(
            gt, pred
        )
    elif dataset_name in ["path_vqa","radcure"]:
        accuracy, roc_auc, ap, mismatched_indices = calculate_bc_metrics_and_plot_roc(
            gt, pred
        )
    # print(f"Acc: {accuracy}, ROC-AUC: {roc_auc}, AP: {ap}")

    if dataset_name == "radcure":
        choices = {"A": "dead", "B": "alive"}
        answer = "['Alive', 'Dead']"

        # mismatched_indices = mismatched_indices[:5]
        error_patient_list = [gt_metas[idx] for idx in mismatched_indices]
        error_results = [results[idx] for idx in mismatched_indices]

        for gt_tmp, pred_tmp in zip(error_patient_list, error_results):
            name = gt_tmp["patient_id"]

            if error_show:
                print(
                    name,
                    f"gt: {choices[gt_tmp["answer"]]}",
                    f"pred: {pred_tmp["status"]}",
                    gt_tmp["question"],
                )
                template = "The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n **Question: {question} **\n **Answer: {answer}. We masked patient alive or dead status for you."
                case = template.format(answer=answer, question=gt_tmp["question"])
                print(
                    "You are a prediction model customized for cancer prediction. Now you need to predict patient status with multiple choice about medical knowledge.",
                )
                print(case)

        print(len(error_patient_list))

    return accuracy

def load_dataset(
    file,
    dataset_name: Literal["med_qa", "path_vqa", "radcure"],
    model_name,
):
    # import textwrap
    # model_name = "gpt-4.1-mini"
    with open(
        f"{file}.json",
        "r",
        encoding="utf-8",
    ) as json_file:
        results = json.load(json_file)
        results = remove_duplicates_by_key(results, "patient_id")

    with open(
        f"{file}.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    import datasets

    def vis_func(data):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(data["image"])
        # wrapped_title = "\n".join(textwrap.wrap(data["question"], width=60))
        # plt.title(wrapped_title)
        ax.set_title(data["question"], fontsize=10, wrap=True)

    cached_name = f"cached_{dataset_name}"
    dataset_name, vis_func = {
        "path_vqa": ("path-vqa", vis_func),
        "med_qa": ("med_qa", lambda x: print(x)),
    }[dataset_name]

    if os.path.exists(f"data/{cached_name}.arrow"):
        dataset = datasets.load_from_disk(f"data/{cached_name}.arrow")
    else:
        dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)
        dataset.save_to_disk(f"{cached_name}.arrow")

    gt = []
    test_data_id = []
    test_data = []

    if dataset_name == "med_qa":
        for idx, line in enumerate(dataset["test"]):
            gt.append(line['answer_idx'])

            line["patient_id"] = f"{dataset_name}-{idx:04d}"
            line["options"] = {item["key"]: item["value"] for item in line["options"]}
            test_data.append(line)

            test_data_id.append(line["patient_id"])

        print(str(dataset))

    elif dataset_name == "path-vqa":
        gt = np.ones_like(results).tolist()
        test_data = results
        test_data_id = [item["patient_id"] for item in results]

    return results, test_data, test_data_id, gt


if __name__ == "__main__":
    cal_metric()
