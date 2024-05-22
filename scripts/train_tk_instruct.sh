#!/bin/bash
set -e
export NCCL_P2P_DISABLE=1
# export HF_HOME=/mnt/publiccache/huggingface/
# export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
port=$(shuf -i25000-30000 -n1)
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
m=${3:-"t5-large"}
echo "model: ${m}"
bs=${4:-"4"}
echo "batch size: ${bs}"
lr=${5:-"5e-5"}
echo "lr: ${lr}"
e=${6:-"0"}
tune=${7:-"full"}
epoch=${8:-"3"}
warmup_ratio=${9:-"0.03"}
r=${10:-"16"}
allenai=${11:-"0"}
use_kl=${12:-"False"}
prompt=${13:-"0"}
ffn=${14:-"0"}
whitening=${15:-"0"}
pos=${16:-"2"}
s_pos=${17:-"10"}
custom=${18:-"0"}
hyper=${19:-"0"}
ko=${20:-"0"}
prefix=${21:-"0"}
gpt=${22:-"0"}
data_type=${23:-"0"}
loramse=${24:-"0"}
stand=${25:-"0"}
dataset=${26:-"0"}
do_sample=${27:-"0"}
cache="./cache"
echo epoch: ${epoch}
name=experiment_pos${pos}_pooler-${m}_lr${lr}_warm${warmup_ratio}_${epoch}
output_dir=output_pos${pos}_pooler/${m}_lr${lr}_warm${warmup_ratio}_${epoch}
extra_args="--evaluation_strategy no"
data_dir=data/splits/default
task_dir=data/tasks
run_file=run_s2s.py
max_num_instances=500
gradient_accumulation_steps=2

if [ "$e" == "eval" ];then
    cache="./cache_eval"
    name="${name}-eval"
    data_dir="${data_dir}_eval"
    output_dir="${output_dir}_eval"
    extra_args="--evaluation_strategy steps --do_eval --eval_steps 2500 --load_best_model_at_end True"
    echo name: ${name}
fi
if [ "$dataset" == "p3" ];then
    data_dir=data_p3_eval
    task_dir=data_p3
    name=experiment_p3_pooler-${m}_lr${lr}_warm${warmup_ratio}
    output_dir=output_p3_pooler/${m}_lr${lr}_warm${warmup_ratio}
    run_file=run_s2s_kd_ac.py
    max_num_instances=1000
fi

if [ "$tune" != "full" ];then
    name="${name}-${tune}_r${r}"
    output_dir="${output_dir}_${tune}_r${r}"
    run_file=run_s2s_${tune}.py
    extra_args="${extra_args} --r ${r}"
    if [ "$dataset" == "p3" ];then
        run_file=run_s2s_kd_ac.py
    fi
fi
model=google/${m}-lm-adapt
if [ "$tune" == "full" ];then
    if [ "$data_type" != "0" ];then
        # sed 's/[ ][ ]*/_/g' <<< $data_type
        run_file=run_s2s_kd.py
        name="${name}_$data_type"
        output_dir="output/$data_type/${output_dir}"
        if [ "$data_type" == "QAa" ];then
            data_type="QA,QG,SA,TLD,PE,Misc." #,TC
            max_num_instances=1200 
        fi
        if [ "$data_type" == "QAb" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU" #,TC
            max_num_instances=1000 
        fi
        if [ "$data_type" == "QAx" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU,Summarization,DG,WS,SCo,SI,PT,LP,FiTB,TQE,SD,GC,TtC,TS,Mathematics,CtT"
            max_num_instances=800 #700
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        if [ "$data_type" == "p3_5" ];then
            run_file=run_s2s_kd_ac.py
            data_type="cosmos_qa,kilt_tasks_hotpotqa,amazon_polarity,cnn_dailymail_3.0.0,common_gen"
            max_num_instances=8000 #700
            gradient_accumulation_steps=1
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        if [ "$data_type" == "p3_10" ];then
            run_file=run_s2s_kd_ac.py
            data_type="cosmos_qa,kilt_tasks_hotpotqa,amazon_polarity,cnn_dailymail_3.0.0,common_gen,glue_mrpc,ag_news,adversarial_qa_dbert,dream,gigaword"
            max_num_instances=4000 #700
            gradient_accumulation_steps=1
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        if [ "$data_type" == "p3_20" ];then
            run_file=run_s2s_kd_ac.py
            data_type="cosmos_qa,kilt_tasks_hotpotqa,amazon_polarity,cnn_dailymail_3.0.0,common_gen,glue_mrpc,ag_news,adversarial_qa_dbert,dream,gigaword,quoref,wiki_qa,wiki_bio,app_reviews,paws,dbpedia_14,quail,ropes,imdb,multi_news"
            max_num_instances=8000 #700
            gradient_accumulation_steps=1
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        extra_args="${extra_args} --data_type $data_type"
    fi
    if [ "$m" == "t5-large" ];then
        max_num_instances=500 
    fi
    if [ "$m" == "t5-xl" ];then
        gradient_accumulation_steps=1
    fi
    if [ "$m" == "t0-base" ];then
        model=qinyuany/my-t0-base
    fi
    if [ "$m" == "t0-large" ];then
        model=qinyuany/my-t0-large
        max_num_instances=500 
    fi
    if [ "$m" == "t0-xl" ];then
        model=qinyuany/my-t0-3b
        gradient_accumulation_steps=1
    fi
    extra_args="${extra_args} --do_predict"
fi

if [ "$tune" == "lora" ];then
    if [ "$allenai" == "allenai" ];then
        if [ "$m" == "t5-base" ];then
            model=allenai/tk-instruct-base-def-pos
            if [ "$pos" == "0" ];then
                model=output_pos0/t5-base_lr1e-4_warm0.05
            fi
        fi
        if [ "$m" == "t5-xl" ];then
            model=allenai/tk-instruct-3b-def-pos
            if [ "$pos" == "0" ];then
                model=allenai/tk-instruct-3b-def 
            fi
            gradient_accumulation_steps=8
        fi
        if [ "$m" == "t5-xxl" ];then
            model=allenai/tk-instruct-11b-def-pos
            if [ "$pos" == "0" ];then
                model=allenai/tk-instruct-11b-def 
            fi
            gradient_accumulation_steps=2
            max_num_instances=300
            if [ "$warmup_ratio" == "0.02" ];then
                gradient_accumulation_steps=2
                max_num_instances=200
            fi
        fi
        name="${name}_allenai"
        output_dir="${output_dir}_allenai"
    fi
    if [ "$data_type" != "0" ];then
        # sed 's/[ ][ ]*/_/g' <<< $data_type
        name="${name}_$data_type"
        output="output_meta/$data_type/${output_dir}"
        if [ "$pos" == "0" ];then
            output="output_meta_pos0/$data_type/${output_dir}"
        fi
        output_dir=${output}
        gradient_accumulation_steps=2
        extra_args="${extra_args} --data_type $data_type"
        max_num_instances=10000 # 10000 2000, 3000
        if [ "$data_type" == "QA" ];then
            max_num_instances=600
        fi
        if [ "$data_type" == "PE" ];then
            max_num_instances=1200
        fi
        if [ "$data_type" == "QG" ];then
            max_num_instances=2000
        fi
        if [ "$data_type" == "SA" ];then
            max_num_instances=3000
        fi
    fi
fi
if [ "$tune" == "lora_p3" ];then
    if [ "$m" == "t0-base" ];then
        model=qinyuany/my-t0-base
    fi
    if [ "$m" == "t0-large" ];then
        model=qinyuany/my-t0-large
    fi
    if [ "$m" == "t0-xl" ];then
        model=qinyuany/my-t0-3b
    fi
    if [ "$data_type" != "0" ];then
        # sed 's/[ ][ ]*/_/g' <<< $data_type
        name="${name}_$data_type"
        output="output_meta_p3_${m}/$data_type/${output_dir}"
        output_dir=${output}
        gradient_accumulation_steps=1
        extra_args="${extra_args} --data_type $data_type"
        run_file=run_s2s_lora.py
        max_num_instances=10000 # 10000 2000, 3000
        if [ "$bs" == "8" ];then
            gradient_accumulation_steps=2
        fi
        if [ "$bs" == "4" ];then
            gradient_accumulation_steps=4
        fi
        if [ "$m" == "t5-xl" ];then
            gradient_accumulation_steps=2
            max_num_instances=7000
            if [ "$bs" == "2" ];then
                max_num_instances=5000
            fi
        fi
        # if [ "$m" == "t5-large" ];then
        #     gradient_accumulation_steps=4
        # fi
        # if [ "$m" == "t0-large" ];then
        #     gradient_accumulation_steps=4
        # fi
        # if [ "$data_type" == "duorc_ParaphraseRC" ];then
        #     gradient_accumulation_steps=2
        # fi
    fi
fi

if [ "$tune" == "kd" ];then
    extra_args="${extra_args} --do_predict"
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
            max_num_instances=400
            gradient_accumulation_steps=4
        fi
        if [ "$m" == "t5-xxl" ];then
            t_model=allenai/tk-instruct-11b-def-pos
            if [ "$pos" == "0" ];then
                t_model=allenai/tk-instruct-11b-def 
            fi
            gradient_accumulation_steps=2
            max_num_instances=150
            if [ "$warmup_ratio" == "0.02" ];then
                gradient_accumulation_steps=2
                max_num_instances=200
            fi
        fi
        name="${name}_allenai"
        output_dir="${output_dir}_allenai"
    fi
    if [ "$use_kl" == "ce" ];then
        name="${name}_ce"
        output_dir="${output_dir}_ce"
        extra_args="${extra_args} --use_ce True"
    fi
    if [ "$use_kl" == "kl" ];then
        name="${name}_kl"
        output_dir="${output_dir}_kl"
        extra_args="${extra_args} --use_kl True"
    fi
    if [ "$use_kl" == "ce_kl" ];then
        name="${name}_ce_kl"
        output_dir="${output_dir}_ce_kl"
        extra_args="${extra_args} --use_kl True --use_ce True"
    fi
    if [ "$use_kl" == "all" ];then
        name="${name}_all"
        output_dir="${output_dir}_all"
        extra_args="${extra_args} --use_kl True --use_ce True --use_hd True --use_attn True"
    fi
    lora=hyperlora_kd
    if [ "$ffn" == "ffn" ];then
        name="${name}_ffn"
        output_dir="${output_dir}_ffn"
        lora="${lora}_ffn"
    fi
    if [ "$ko" == "ko" ];then
        name="${name}_ko"
        output_dir="${output_dir}_ko"
        lora="${lora}_ko"
    fi
    if [ "$prefix" != "0" ];then
        name="${name}_prefix${prefix}"
        output_dir="${output_dir}_prefix${prefix}"
        lora="${lora}_prefix"
        extra_args="${extra_args} --prefix_length ${prefix}"
        if [ "$gpt" == "gpt" ];then
            name="${name}_gpt"
            output_dir="${output_dir}_gpt"
            lora="${lora}_gpt"
        fi
    fi
    extra_args="${extra_args} --t_model ${t_model} --name ${lora} --temperature 3.0 --kd True --s_num_pos_examples ${s_pos}"
    if [ "$prompt" == "fullprompt" ];then
        name="${name}_${prompt}"
        output_dir="${output_dir}_${prompt}"
        extra_args="${extra_args} --prompt True"
    fi
    name="${name}_${whitening}"
    output_dir="${output_dir}_${whitening}"
    if [ "$whitening" == "whitening" ];then
        extra_args="${extra_args} --whitening True"
    fi
    if [ "$s_pos" == "10" ];then
        s_pos=${pos}
    fi
    if [ "$s_pos" != "$pos" ];then
        name="${name}_s_pos${s_pos}"
        output_dir="${output_dir}_s_pos${s_pos}"
    fi
    extra_args="${extra_args} --s_num_pos_examples ${s_pos}"
    if [ "$custom" == "custom" ];then
        name="${name}_custom"
        output_dir="${output_dir}_custom"
        extra_args="${extra_args} --custom_model True"
    fi
    if [ "$hyper" == "hyper" ];then
        name="${name}_hyper"
        output_dir="${output_dir}_hyper"
        extra_args="${extra_args} --hyperencoder True"
    fi
    if [ "$stand" == "stand" ];then
        name="${name}_stand"
        output_dir="${output_dir}_stand"
        extra_args="${extra_args} --logit_stand True"
    fi
    if [ "$loramse" == "loramse" ];then
        name="${name}_loramse"
        output_dir="${output_dir}_loramse"
        extra_args="${extra_args} --loramse True"
    fi
    if [ "$data_type" != "0" ];then
        # sed 's/[ ][ ]*/_/g' <<< $data_type
        name="${name}_$data_type"
        output_dir="${output_dir}_$data_type"
        max_num_instances=10000
        if [ "$data_type" == "QAa" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,TC"
            max_num_instances=1200
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        if [ "$data_type" == "QAb" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU" #,TC
            max_num_instances=1000 
        fi
        if [ "$data_type" == "QAx" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU,Summarization,DG,WS,SCo,SI,PT,LP,FiTB,TQE,SD,GC,TtC,TS,Mathematics,CtT"
            max_num_instances=800 #700
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=300
            fi
        fi
        if [ "$data_type" == "QAxx" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU,Summarization,DG,WS,SCo,SI,PT,LP,FiTB,TQE,SD,GC,TtC,TS,StC,Explanation,Mathematics,CtT,II"
            max_num_instances=800 #700
            if [ "$m" == "t5-xxl" ];then
                max_num_instances=200
            fi
        fi
        extra_args="${extra_args} --data_type $data_type"
        
    fi
    
fi

if [ "$tune" == "kd_p3" ];then
    t_model=output/${model}_lr5e-5
    if [ "$allenai" == "fid" ];then
        if [ "$m" == "t5-base" ];then
            t_model=qinyuany/fid-icl-t5-lm-base
        fi
        if [ "$m" == "t5-large" ];then
            t_model=qinyuany/fid-icl-t5-lm-large
            gradient_accumulation_steps=1
        fi
        if [ "$m" == "t5-xl" ];then
            t_model=qinyuany/fid-icl-t5-lm-xl
            gradient_accumulation_steps=2
            max_num_instances=500
        fi
        if [ "$m" == "t0-base" ];then
            model=qinyuany/my-t0-base
            t_model=qinyuany/fid-icl-t0-base
        fi
        if [ "$m" == "t0-large" ];then
            model=qinyuany/my-t0-large
            t_model=qinyuany/fid-icl-t0-large
            gradient_accumulation_steps=4
        fi
        if [ "$m" == "t0-xl" ];then
            model=qinyuany/my-t0-3b
            t_model=qinyuany/fid-icl-t0-3b
            gradient_accumulation_steps=2
            max_num_instances=500
        fi
        name="${name}_fid"
        output_dir="${output_dir}_fid"
    fi
    if [ "$use_kl" == "ce" ];then
        name="${name}_ce"
        output_dir="${output_dir}_ce"
        extra_args="${extra_args} --use_ce True"
    fi
    if [ "$use_kl" == "kl" ];then
        name="${name}_kl"
        output_dir="${output_dir}_kl"
        extra_args="${extra_args} --use_kl True"
    fi
    if [ "$use_kl" == "ce_kl" ];then
        name="${name}_ce_kl"
        output_dir="${output_dir}_ce_kl"
        extra_args="${extra_args} --use_kl True --use_ce True"
    fi
    if [ "$use_kl" == "all" ];then
        name="${name}_all"
        output_dir="${output_dir}_all"
        extra_args="${extra_args} --use_kl True --use_ce True --use_hd True --use_attn True"
    fi
    lora=hyperlora_kd
    if [ "$ffn" == "ffn" ];then
        name="${name}_ffn"
        output_dir="${output_dir}_ffn"
        lora="${lora}_ffn"
    fi
    if [ "$ko" == "ko" ];then
        name="${name}_ko"
        output_dir="${output_dir}_ko"
        lora="${lora}_ko"
    fi
    if [ "$prefix" != "0" ];then
        name="${name}_prefix${prefix}"
        output_dir="${output_dir}_prefix${prefix}"
        lora="${lora}_prefix"
        extra_args="${extra_args} --prefix_length ${prefix}"
        if [ "$gpt" == "gpt" ];then
            name="${name}_gpt"
            output_dir="${output_dir}_gpt"
            lora="${lora}_gpt"
        fi
    fi
    extra_args="${extra_args} --t_model ${t_model} --name ${lora} --temperature 3.0 --kd True --s_num_pos_examples ${s_pos}  --m_label ${m}"
    if [ "$prompt" == "fullprompt" ];then
        name="${name}_${prompt}"
        output_dir="${output_dir}_${prompt}"
        extra_args="${extra_args} --prompt True"
    fi
    name="${name}_${whitening}"
    output_dir="${output_dir}_${whitening}"
    if [ "$whitening" == "whitening" ];then
        extra_args="${extra_args} --whitening True"
    fi
    if [ "$s_pos" == "10" ];then
        s_pos=${pos}
    fi
    if [ "$s_pos" != "$pos" ];then
        name="${name}_s_pos${s_pos}"
        output_dir="${output_dir}_s_pos${s_pos}"
    fi
    extra_args="${extra_args} --s_num_pos_examples ${s_pos}"
    if [ "$custom" == "custom" ];then
        name="${name}_custom"
        output_dir="${output_dir}_custom"
        extra_args="${extra_args} --custom_model True"
    fi
    if [ "$hyper" == "hyper" ];then
        name="${name}_hyper"
        output_dir="${output_dir}_hyper"
        extra_args="${extra_args} --hyperencoder True"
    fi
    if [ "$stand" == "stand" ];then
        name="${name}_stand"
        output_dir="${output_dir}_stand"
        extra_args="${extra_args} --logit_stand True"
    fi
    if [ "$loramse" == "loramse" ];then
        name="${name}_loramse"
        output_dir="${output_dir}_loramse"
        extra_args="${extra_args} --loramse True"
    fi
    if [ "$data_type" != "0" ];then
        # sed 's/[ ][ ]*/_/g' <<< $data_type
        name="${name}_$data_type"
        output_dir="${output_dir}_$data_type"
        max_num_instances=10000
        if [ "$data_type" == "QAa" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,TC"
            max_num_instances=1200 
        fi
        if [ "$data_type" == "QAx" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU,Summarization,DG,WS,SCo,SI,PT,LP,FiTB,TQE,SD,GC,TtC,TS,Mathematics,CtT"
            max_num_instances=800 #700
        fi
        if [ "$data_type" == "QAxx" ];then
            data_type="QA,QG,SA,TLD,PE,Misc.,NER,TC,CC,CCl,TM,IE,WCG,TCo,QU,Summarization,DG,WS,SCo,SI,PT,LP,FiTB,TQE,SD,GC,TtC,TS,StC,Explanation,Mathematics,CtT,II"
            max_num_instances=800 #700
        fi
        extra_args="${extra_args} --data_type $data_type"
        
    fi
    extra_args="${extra_args} --do_predict"
fi

if [ "$do_sample" == "sample" ];then
    name="${name}_sample"
    output_dir="${output_dir}_sample"
    extra_args="${extra_args} --do_sample True"
fi
if [ "$epoch" != "3" ];then
    extra_args="${extra_args} --max_steps ${epoch} "
fi

echo name: ${name}
echo run_file: ${run_file}
echo output_dir: ${output_dir}
#     --do_predict \
deepspeed --master_port $port -i localhost:${gpus} src/${run_file} \
    --do_train \
    --predict_with_generate \
    --model_name_or_path ${model} \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task ${max_num_instances} \
    --max_num_instances_per_eval_task ${max_num_instances} \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples ${pos} \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ${data_dir} \
    --task_dir ${task_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --cache_dir ${cache} \
    --overwrite_cache \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size ${bs} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --pad_to_max_length False \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --lr_scheduler_type constant \
    --warmup_ratio ${warmup_ratio} \
    --logging_strategy steps \
    --logging_steps 500 \
    --save_strategy steps \
    --save_steps 2500 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --save_total_limit 1 \
    --seed 42 \
    --run_name ${name} \
    ${extra_args}
#    --report_to none \