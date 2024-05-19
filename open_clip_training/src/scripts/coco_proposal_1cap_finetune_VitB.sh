cd open_clip_training
cd src

torchrun --nproc_per_node 2 -m training.main ^
    --train-data ..\openclip_data\coco_proposal_1cap.csv ^
    --train-num-samples 442117 ^
    --lr 0.000005 ^
    --warmup 100 ^
    --force-quick-gelu ^
    --dataset-type csv ^
    --batch-size 128 ^
    --precision amp ^
    --workers 0 ^
    --dist-backend gloo ^
    --model  ViT-B-16 ^
    --lock-text ^
    --zeroshot-frequency 1 ^
    --save-frequency 1 ^
    --epoch 5 ^
    --pretrained  openai ^
    --ade-val ..\openclip_data\ade_gt_150cls_val
