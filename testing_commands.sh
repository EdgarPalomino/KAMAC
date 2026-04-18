# 1 agent without CoT reasoning

python main_kamac.py \
    --dataset_name pancreatic_cancer \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac \
    --vlm \
    --auto_recruit \
    --resampling_mode all_specific_ids \
    --cache_path final_results/pancreatic_cancer_gpt-4.1-mini_1_kamac \
    --saved_as_path results/pancreatic_cancer_gpt-4.1-mini_1_kamac

# 1 agent with CoT Reasoning

python main_kamac.py \
    --dataset_name pancreatic_cancer \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac \
    --vlm \
    --cot \
    --auto_recruit \
    --resampling_mode all_specific_ids \
    --cache_path final_results/pancreatic_cancer_gpt-4.1-mini_1_CoT \
    --saved_as_path results/pancreatic_cancer_gpt-4.1-mini_1_CoT

python select_pc.py

# 3 agents without CoT reasoning

python main_kamac.py \
    --dataset_name pancreatic_cancer \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac \
    --vlm \
    --auto_recruit \
    --num_agents 3 \
    --resampling_mode all_specific_ids \
    --cache_path final_results/pancreatic_cancer_gpt-4.1-mini_3_kamac \
    --saved_as_path results/pancreatic_cancer_gpt-4.1-mini_3_kamac

# 3 agents with CoT reasoning

python main_kamac.py \
    --dataset_name pancreatic_cancer \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac \
    --vlm \
    --auto_recruit \
    --num_agents 3 \
    --resampling_mode all_specific_ids \
    --cache_path final_results/pancreatic_cancer_gpt-4.1-mini_3_CoT \
    --saved_as_path results/pancreatic_cancer_gpt-4.1-mini_3_CoT
