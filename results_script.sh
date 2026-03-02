export OPENAI_API_KEY="your_api_key_here"

python main_kamac.py  \
    --dataset_name radcure \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac \
    --cot \
    --auto_recruit \
    --resampling_mode all_specific_ids \
    --cache_path final_results/radcure_gpt-4.1-mini_1_kamac \
    --saved_as_path results/radcure_gpt-4.1-mini_1_kamac

python select_radcure.py

python main_kamac.py  \
    --dataset_name radcure \
    --model_name gpt-4.1-mini \
    --prefix gpt-4.1-mini_kamac_test \
    --cot \
    --auto_recruit \
    --resampling_mode all_specific_ids \
    --cache_mode "none" \
    --num_agents 1 \
    --cache_path results/radcure_gpt-4.1-mini_kamac_test \
    --n_jobs 8
