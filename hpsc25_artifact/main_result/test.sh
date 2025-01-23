# 分别在offline、online、whole_process模式下进行实验
# python hpsc25_artifact/main_result/my_general_model_case.py --exp-ids goodput_vs_num_devices --output main_result.tsv --exp-name main_result --workload=azure_v1 --model-type=mixed --num-devices 12 --num_devices_per_node 4
# python hpsc25_artifact/main_result/my_general_model_case.py --exp-ids goodput_vs_num_devices --output main_result.tsv --exp-name main_result --workload=online --model-type=mixed --num-devices 12 --num_devices_per_node 4
# python hpsc25_artifact/main_result/my_general_model_case.py --exp-ids goodput_vs_num_devices --output main_result.tsv --exp-name main_result --workload=whole_process --model-type=mixed --num-devices 12 --num_devices_per_node 4
