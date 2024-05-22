set -e
# allenai/tk-instruct-3b-def-pos
# export CUDA_VISIBLE_DEVICES=3
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
m=${3:-"t5-large"}
batch_size=${4:-"8"}
pos=${5:-"2"}
allenai=${6:-"0"}
kd=${7:-"0"}
m_path=${8:-"0"}
model=google/${m}-lm-adapt
run_file=run_s2s.py
data_dir=data/splits/default
task_dir=data/tasks
if [ "$allenai" == "allenai" ];then
    if [ "$m" == "t5-base" ];then
        model=allenai/tk-instruct-base-def-pos
    fi
    if [ "$m" == "t5-xl" ];then
        model=allenai/tk-instruct-3b-def-pos
        if [ "$pos" == "0" ];then
            model=allenai/tk-instruct-3b-def 
        fi
    fi
    if [ "$model" == "t5-xxl" ];then
        model=allenai/tk-instruct-11b-def-pos
        if [ "$pos" == "0" ];then
            model=allenai/tk-instruct-11b-def 
        fi
    fi
fi
out="output/${model}_eval_pos${pos}"
extra_args=""
if [ "$kd" == "kd" ];then
    t_model=output/${model}_lr5e-5
    if [ "$allenai" == "allenai" ];then
        if [ "$m" == "t5-base" ];then
            t_model=allenai/tk-instruct-base-def-pos
            if [ "$pos" == "0" ];then
                t_model=output_pos0/t5-base_lr1e-4_warm0.05
            fi
        fi
        if [ "$m" == "t5-xl" ];then
            t_model=allenai/tk-instruct-3b-def-pos
            if [ "$pos" == "0" ];then
                t_model=allenai/tk-instruct-3b-def 
            fi
        fi
        if [ "$m" == "t5-xxl" ];then
            t_model=allenai/tk-instruct-11b-def-pos
            if [ "$pos" == "0" ];then
                t_model=allenai/tk-instruct-11b-def 
            fi
        fi
    fi
    run_file=run_s2s_kd.py
    model=$m_path
    out="output/${m_path}_eval_pos${pos}"
    extra_args="${extra_args} --kd True --t_model ${t_model}"
fi
if [ "$kd" == "p3" ];then
    run_file=run_s2s_kd_ac.py
    data_dir=data_p3_eval
    task_dir=data_p3
    model=$m_path
    out="output_p3/${m_path}_eval_pos${pos}"
    if [ "$allenai" == "fid" ];then
        if [ "$m" == "t0-base" ];then
            model=qinyuany/my-t0-base
        fi
        if [ "$m" == "t0-large" ];then
            model=qinyuany/my-t0-large
        fi
        if [ "$m" == "t0-xl" ];then
            model=qinyuany/my-t0-3b
        fi
        name="${name}_fid"
        out="${out}_fid"
    fi
fi
if [ "$kd" == "kd_p3" ];then
    t_model=output/${model}_lr5e-5
    if [ "$allenai" == "fid" ];then
        if [ "$m" == "t5-base" ];then
            t_model=qinyuany/fid-icl-t5-lm-base
        fi
        if [ "$m" == "t5-large" ];then
            t_model=qinyuany/fid-icl-t5-lm-large
            gradient_accumulation_steps=4
        fi
        if [ "$m" == "t5-xl" ];then
            t_model=qinyuany/fid-icl-t5-lm-xl
            gradient_accumulation_steps=4
            max_num_instances=500
        fi
        if [ "$m" == "t0-base" ];then
            model=qinyuany/my-t0-base
            t_model=qinyuany/fid-icl-t0-base
        fi
        if [ "$m" == "t0-large" ];then
            model=qinyuany/my-t0-large
            t_model=qinyuany/fid-icl-t0-large
        fi
        if [ "$m" == "t0-xl" ];then
            model=qinyuany/my-t0-3b
            t_model=qinyuany/fid-icl-t0-3b
        fi
        name="${name}_fid"
        out="${out}_fid"
    fi
    run_file=run_s2s_kd_ac.py
    data_dir=data_p3_eval
    task_dir=data_p3
    model=$m_path
    out="output_p3/${m_path}_eval_pos${pos}"
    extra_args="${extra_args} --kd True --t_model ${t_model}"
fi
echo "model: ${model}"
echo "t_model: ${t_model}"
echo ${out}
port=$(shuf -i25000-30000 -n1)
# deepspeed --master_port $port -i localhost:${gpus} src/${run_file} \
CUDA_VISIBLE_DEVICES=${gpus} python src/${run_file} \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path $model \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples ${pos} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ${data_dir} \
    --task_dir ${task_dir} \
    --output_dir $out \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size ${batch_size} \
    --per_device_train_batch_size ${batch_size} \
    ${extra_args}