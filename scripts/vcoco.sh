#### training scripts for V-COCO with ViT-L backbone
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_tip_finetune.py --world-size 4 \
    --dataset vcoco --data-root vcoco/ --partitions trainval test \
    --pretrained  checkpoints/detr-r50-vcoco.pth \
    --output-dir checkpoints/vcoco_HO_adpt_default/hola \
    --use_insadapter \
    --num_classes 24 \
    --use_multi_hot \
    --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p  \
    --clip_dir_vit checkpoints/pretrained_CLIP/ViT-L-14-336px.pt \
    --port 1236 \
    --logits_type "HO+U"  \
    --llmtxt \
    --batch-size 4 \
    --epoch 13 \
    --img_align   \
    --self_adapt \
    --seperate_ho 1 \
    --basis_feat_enable \
    --basis_feat_constraint 'none' \
    --disentangle_basis \
    --ao_sep_basis \
    --act_txtdecrip \
    --kl_t 0.1 \
    --basis_feat_init 'pca' \
    --recon_ratio_pca 0.95 \
    --ho_pair_pt \
    --ho_pair_prior 1 \
    --pred_type  'ho+u+l' \
    --pt_init 'pos+detr' \
    --semloss \
    --unique_basis_weights \




# #### evaluation scripts for V-COCO with ViT-L backbone
# CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
#     --dataset vcoco --data-root vcoco/ --partitions trainval test \
#     --pretrained  checkpoints/detr-r50-vcoco.pth \
#     --output-dir checkpoints/vcoco_HO_adpt_default/lordhoi \
#     --use_insadapter \
#     --num_classes 24 \
#     --use_multi_hot \
#     --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p  \
#     --clip_dir_vit checkpoints/pretrained_CLIP/ViT-L-14-336px.pt \
#     --port 1236 \
#     --logits_type "HO+U"  \
#     --llmtxt \
#     --batch-size 4 \
#     --epoch 13 \
#     --img_align   \
#     --self_adapt \
#     --seperate_ho 1 \
#     --basis_feat_enable \
#     --basis_feat_constraint 'none' \
#     --disentangle_basis \
#     --ao_sep_basis \
#     --act_txtdecrip \
#     --kl_t 0.1 \
#     --basis_feat_init 'pca' \
#     --recon_ratio_pca 0.95 \
#     --ho_pair_pt \
#     --ho_pair_prior 1 \
#     --pred_type  'ho+u+l' \
#     --pt_init 'pos+detr' \
#     --semloss \
#     --unique_basis_weights \
#     --cache --resume <path_to_model>



# #### vcoco evaluation:    
# python eval_vcoco.py   <path_to_model_cache>