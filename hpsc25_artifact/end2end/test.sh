# 分别在azure_v1、azure_v2 trace下进行实验

# num_devices
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_num_devices --output goodput_vs_num_devices.tsv --exp-name end2end_new --workload=azure_v1 --model-type=mixed  --num_devices_per_node 4
python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_num_devices --output goodput_vs_num_devices.tsv --exp-name end2end_new --workload=azure_v2 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_num_devices --output goodput_vs_num_devices.tsv --exp-name end2end_new --workload=synthetic --model-type=mixed  --num_devices_per_node 4

# slo_scales
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_slo --output goodput_vs_slo.tsv --exp-name end2end_new --workload=azure_v1 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_slo --output goodput_vs_slo.tsv --exp-name end2end_new --workload=azure_v2 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_slo --output goodput_vs_slo.tsv --exp-name end2end_new --workload=synthetic --model-type=mixed  --num_devices_per_node 4


# rate_scales
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_rate --output goodput_vs_rate.tsv --exp-name end2end_new --workload=azure_v1 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_rate --output goodput_vs_rate.tsv --exp-name end2end_new --workload=azure_v2 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_rate --output goodput_vs_rate.tsv --exp-name end2end_new --workload=synthetic --model-type=mixed  --num_devices_per_node 4


# cv_scales
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_cv --output goodput_vs_cv.tsv --exp-name end2end_new --workload=azure_v1 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_cv --output goodput_vs_cv.tsv --exp-name end2end_new --workload=azure_v2 --model-type=mixed  --num_devices_per_node 4
# python hpsc25_artifact/end2end/my_general_model_case.py --exp-ids goodput_vs_cv --output goodput_vs_cv.tsv --exp-name end2end_new --workload=synthetic --model-type=mixed  --num_devices_per_node 4

