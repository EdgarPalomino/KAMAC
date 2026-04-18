import csv
import json
import os
import shutil

data = []
cases = []

with open("pancreatic_cancer_data.csv", "r") as file:

    data = list(csv.DictReader(file))

    for i, patient in enumerate(data):
        vital_status = patient.pop("Vital Status")
        cases.append(patient["Case Submitter ID"])
        patient["patient_id"] = f"PC-{i+1}"
        patient["question"] = [", ".join([f"{key}: {value}" for key, value in patient.items()])]
        patient["options"] = {"A": "Dead", "B": "Alive"}
        patient["answer"] = "A" if vital_status == "Dead" else "B"

with open("pancreatic_cancer_data_clean.json", "w") as file:
    json.dump(data, file, indent=4)

for patient in data:
    ct_image = f"CT Images/{patient["Case Submitter ID"]}.nii.gz"
    tumor_mask = f"Tumor Masks/{patient["Case Submitter ID"]}_tumor.nii.gz"
    os.mkdir(f"PC/{patient["patient_id"]}")
    shutil.copy(ct_image, f"PC/{patient["patient_id"]}/{patient["patient_id"]}_ct.nii.gz")
    shutil.copy(tumor_mask, f"PC/{patient["patient_id"]}/{patient["patient_id"]}_tumor.nii.gz")
