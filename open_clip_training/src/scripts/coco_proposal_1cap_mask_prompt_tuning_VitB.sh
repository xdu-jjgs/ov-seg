cd open_clip_training
cd src

torchrun --nproc_per_node 2 -m training.main_mask_prompt_tuning ^
    --train-data ../openclip_data/coco_proposal_1cap.csv ^
    --train-num-samples 442117 ^
    --lr 0.05 ^
    --mask_wd 0.0 ^
    --warmup 100 ^
    --force-quick-gelu ^
    --dataset-type csv ^
    --batch-size 256 ^
    --precision amp ^
    --workers 0 ^
    --dist-backend gloo ^
    --with-mask ^
    --model  ViT-B-16 ^
    --mask-emb-depth 3 ^
    --lock-text ^
    --lock-image ^
    --lock-image-unlocked-groups 0 ^
    --zeroshot-frequency 1 ^
    --save-frequency 1 ^
    --epoch 5 ^
    --pretrained ./logs/2024_05_05-17_18_05-model_ViT-B-16-lr_5e-06-b_128-j_0-p_amp/checkpoints/epoch_3.pt ^
    --ade-val ../openclip_data/ade_gt_150cls_val