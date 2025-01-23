# python monitor/my_general_model_case.py --trace-dir /home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl --exp-ids goodput_vs_num_devices --output azure_v1_mixed.tsv --exp-name sec6_2_data --workload=azure_v1 --model-type=mixed --policy mp-search-sep --parallel
# python monitor/monitor.py --trace-dir /home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl --output res_monitor_general_model_cases --exp-name monitor_exp_busiest_device_1 --workload=azure_v1 --model-type=mixed --policy mp-search-sep --rate_scale 0.01 --mem-budget 8
# python monitor/request_inspect.py --single_model_dir_path monitor/monitor_exp_busiest_device_1/res_monitor_general_model_cases_single_model_dir --plot_single_save_path monitor/monitor_exp_busiest_device_1/plot_res_monitor_general_model_cases_single_model_dir --plot_cluster_save_path monitor/monitor_exp_busiest_device_1
# python monitor/memory_usage_inspect.py --single_model_dir_path monitor/monitor_exp_busiest_device_1/res_monitor_general_model_cases_single_model_dir --plot_cluster_save_path monitor/monitor_exp_busiest_device_1

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --parallel --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy my-mp-ilp --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy my-mp-ilp-replace-600  --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy mp-ilp-replace-600  --scheduling_policy load_balance --rate 10

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy sr-replace-600  --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy mp-ilp-replace-600  --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy mp-ilp --scheduling_policy load_balance --rate 10
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output synthetic_mixed.tsv --exp-name synthetic_mixed --workload=synthetic --model-type=synthetic --policy my-mp-ilp --scheduling_policy load_balance --rate 10

# 探究不同rate下的性能
# python monitor/my_general_model_case.py --exp-ids goodput_vs_rate --output azure_v1_rate.tsv --exp-name azure_v1_exp --trace-dir azure_v1 --workload=azure_v1 --num_devices 4 --num_devices_per_node 2 --model-type=synthetic --parallel
# 探究不同slo_scale下的性能
# python monitor/my_general_model_case.py --exp-ids goodput_vs_slo --output synthetic_mixed_rate.tsv --exp-name synthetic_mixed_rate --workload=synthetic --model-type=synthetic --parallel
# 探究不同num_devices下的性能
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output azure_v2_mixed_rate.tsv --exp-name azure_v2_mixed_num_devices --workload=azure_v2 --model-type=synthetic --num_devices 8 --num_devices_per_node 4


# python monitor/my_general_model_case.py --exp-ids goodput_vs_rate --output azure_v1_rate.tsv --exp-name azure_v1_exp --trace-dir azure_v1 --workload=azure_v1 --num_devices 4 --num_devices_per_node 2 --model-type=synthetic --parallel --policy mp-search-sep

# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_azure_v1_convoy_ori.tsv --exp-name goodput_vs_cv_convoy_ori --workload=azure_v1 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate_scale 0.01 --num-devices 8 --num_devices_per_node 4
# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_synthetic.tsv --exp-name goodput_vs_cv --workload=synthetic --model-type=synthetic --policy my-mp-ilp --scheduling_policy load_balance --rate 100 --num-devices 8 --num_devices_per_node 4

# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_azure_v2.tsv --exp-name goodput_vs_cv_group_models --workload=azure_v2 --model-type=synthetic --policy my-mp-ilp-model-groups --scheduling_policy load_balance --rate 5 --num-devices 8 --num_devices_per_node 4 --model_groups_num 2
# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_azure_v2.tsv --exp-name goodput_vs_cv_group_models --workload=azure_v2 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 5 --num-devices 8 --num_devices_per_node 4
# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_azure_v2.tsv --exp-name goodput_vs_cv_group_models --workload=azure_v2 --model-type=synthetic --policy sr-greedy --scheduling_policy load_balance --rate 5 --num-devices 8 --num_devices_per_node 4

# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_synthetic.tsv --exp-name goodput_vs_cv_group_models --workload=synthetic --model-type=synthetic --policy my-mp-ilp-model-groups --scheduling_policy load_balance --rate 5 --num-devices 8 --num_devices_per_node 4 --monitor
# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output group_models_synthetic.tsv --exp-name goodput_vs_cv_group_models --workload=synthetic --model-type=synthetic --policy my-mp-ilp --scheduling_policy load_balance --rate 5 --num-devices 8 --num_devices_per_node 4 --monitor

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_synthetic_num_devices.tsv --exp-name goodput_vs_num_devices --workload=synthetic --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000
#python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_azure_v1_num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v1 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000
#python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_azure_v2_num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v2 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_azure_v2_num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v2 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_azure_v1_num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v1 --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output group_models_synthetic_num_devices.tsv --exp-name goodput_vs_num_devices --workload=synthetic --model-type=synthetic --policy mp-search-sep --scheduling_policy load_balance --rate 10 --num-devices 8 --num_devices_per_node 4 --duration 36000

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v1 --model-type=mixed --num_devices_per_node 8 --parallel
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name goodput_vs_num_devices --workload=azure_v2 --model-type=mixed --num_devices_per_node 8 --parallel

# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output cv.tsv --exp-name goodput_vs_cv --workload=azure_v1 --model-type=mixed --num_devices_per_node 8 --parallel
# python monitor/my_general_model_case.py --exp-ids goodput_vs_cv --output cv.tsv --exp-name goodput_vs_cv --workload=azure_v2 --model-type=mixed --num_devices_per_node 8

# python monitor/my_general_model_case.py --exp-ids goodput_vs_rate --output rate.tsv --exp-name goodput_vs_rate --workload=azure_v1 --model-type=mixed --num_devices_per_node 8 --parallel
# python monitor/my_general_model_case.py --exp-ids goodput_vs_rate --output rate.tsv --exp-name goodput_vs_rate --workload=azure_v2 --model-type=mixed --num_devices_per_node 8 --parallel

# 这是展示动态方式变化情况的实验设置
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=synthetic --policy heuristic-dynamic --model-type=mixed --rate 10 --num_devices_per_node 4

# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=synthetic --model-type=mixed --rate 30 --num-devices 16 --num_devices_per_node 8 --duration 3600
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=azure_v2 --model-type=mixed --num-devices 12 --num_devices_per_node 4

# 分别在offline、online、whole_process模式下进行实验
python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=azure_v1 --model-type=mixed --num-devices 12 --num_devices_per_node 4
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=online --model-type=mixed --num-devices 12 --num_devices_per_node 4
# python monitor/my_general_model_case.py --exp-ids goodput_vs_num_devices --output num_devices.tsv --exp-name test_goodput_vs_num_devices --workload=whole_process --model-type=mixed --num-devices 12 --num_devices_per_node 4
