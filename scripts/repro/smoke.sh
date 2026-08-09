#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {foraging|pp|sc2} [seed] [sc2_map]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage

target=$1
seed=${2:-1}
map=${3:-5m_vs_6m}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pymarl_dir="$repo_root/pymarl-master"

unset LD_LIBRARY_PATH PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

export SC2PATH=${SC2PATH:-"$pymarl_dir/3rdparty/StarCraftII"}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ ${CONDA_DEFAULT_ENV:-} != "macc" ]]; then
    echo "Activate the macc Conda environment before running this script." >&2
    exit 1
fi

common=(
    --config=macc
    --remarks=smoke
    with
    seed="$seed"
    t_max=20000
    test_interval=5000
    test_nepisode=4
    log_interval=5000
    runner_log_interval=5000
    learner_log_interval=5000
    save_model=False
    use_tensorboard=True
)

cd "$pymarl_dir"
case "$target" in
    foraging)
        python src/main.py --env-config=foraging "${common[@]}"
        ;;
    pp)
        python src/main.py --env-config=pred_prey_punish "${common[@]}"
        ;;
    sc2)
        python src/main.py --env-config=sc2 "${common[@]}" env_args.map_name="$map"
        ;;
    *)
        usage
        ;;
esac
