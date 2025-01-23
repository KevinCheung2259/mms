"""The model placement policy"""

from alpa_serve.placement_policy.base_policy import ModelData, ClusterEnv
from alpa_serve.placement_policy.model_parallelism import (
    ModelParallelismILP, ModelParallelismRR, MyModelParallelismILP, MyModelParallelismILPReplacement, ModelParallelismILPReplacement,
    MyModelParallelismHeuReplacement, 
    ModelParallelismGreedy, ModelParallelismSearch,
    ModelParallelismEqual,
    ModelParallelismSearchReplacement)
from alpa_serve.placement_policy.selective_replication import (
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationUniform, SelectiveReplicationSearch,
    SelectiveReplicationReplacement,
    MySelectiveReplicationReplacement)
from alpa_serve.placement_policy.model_parallelism_RL import (
    MyModelParallelismDQNReplacement
)
