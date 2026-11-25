#!/bin/bash
# Stage 4 multi-layer re-run, launched on the GPU box.
#
# Logs land in results/ on cephfs, which the dev box shares, so this can run
# detached instead of inside the login session.
cd "$(dirname "$0")/.." || exit 1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29573
export PYTHONUNBUFFERED=1

step() { echo; echo "######## $* ########"; }

step env
python -V
python -c "import torch, live_eplb_cuda as c; print('torch', torch.__version__, \
'gpus', torch.cuda.device_count(), 'ext', c.__file__)" || exit 1

step "unit: moe_layer (stack views, graph)"
python tests/test_moe_layer.py

step "unit: controller (multi-layer state machine)"
python tests/test_controller.py

step "unit: p2p (2 ranks)"
torchrun --nproc_per_node=2 --master_port=29574 tests/test_p2p.py

step "unit: planner"
python tests/test_planner.py

step "smoke: 40 layers, static only, 20 steps -- memory and step cost"
torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py \
    --layers 40 --steps 20 --policies static || exit 1

step "full: 40 layers, 300 steps, three policies"
torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py \
    --layers 40 --steps 300

step "reference: 1 layer, 300 steps -- what Stage 4 originally reported"
torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py \
    --layers 1 --steps 300 --out results/stage4_timeline_1layer.json

echo; echo "######## ALL_DONE ########"
