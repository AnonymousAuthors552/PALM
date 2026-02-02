#!/bin/bash
set -euo pipefail

# ---------------- CONFIG ----------------
attestation_with_model=("finetune" "eval" "eval_bleu" "inference")
attestation_no_model=("pretrain" "distribution" "preprocess" "bind")

models=("llama" "gemma" "phi")
model_size="L"

measure="--measure"   # leave empty "" if not using laminator measurement
device="cuda"
num_runs=5

python_exec="../.venv/bin/python3"
script="main_LLM.py"
# ----------------------------------------

run_common() {
    local attestation_type="$1"
    local model_arg="$2"
    local model_size_arg="$3"

    for use_in_memory in false true; do
        if [ "$use_in_memory" = true ]; then
            in_memory="--in_memory"
        else
            in_memory=""
        fi

        echo "=============================="
        echo " in_memory = $use_in_memory"
        echo "=============================="

        for i in $(seq 1 $num_runs); do
            echo "------------------------------"
            echo " Attestation: $attestation_type"
            echo " Run $i / $num_runs"
            echo " Measure: $measure | in_memory = $use_in_memory"
            echo "------------------------------"

            sudo $python_exec $script \
                --attestation_type "$attestation_type" \
                $model_arg \
                $model_size_arg \
                $measure \
                $in_memory \
                --device "$device"

            echo ""
        done
    done
}

# --------- Attestations WITH models ---------
for attestation_type in "${attestation_with_model[@]}"; do
    echo "########################################"
    echo " Running (with model): $attestation_type"
    echo "########################################"

    for model in "${models[@]}"; do
        echo "----------------------------------------"
        echo " Model: $model"
        echo "----------------------------------------"

        run_common \
            "$attestation_type" \
            "--model $model" \
            "--model_size $model_size"
    done
done

# --------- Attestations WITHOUT models ---------
for attestation_type in "${attestation_no_model[@]}"; do
    echo "########################################"
    echo " Running (no model): $attestation_type"
    echo "########################################"

    run_common \
        "$attestation_type" \
        "" \
        ""
done
