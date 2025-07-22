#### training script for HICO-DET with ViT-B backbone
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/hico_HO_adpt_uv_512_clipbase/hola \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
    --port 1235 \
    --logits_type "HO+U"  \
    --llmtxt   \
    --wo_unseen_pred \
    --batch-size 16 \
    --epoch 12 \
    --zs --zs_type "unseen_verb" \
    --use_insadapter \
    --img_align   \
    --self_adapt \
    --basis_feat_enable \
    --disentangle_basis \
    --basis_feat_init 'pca' \
    --recon_ratio_pca 71 \
    --unique_basis_weights \
    --seperate_ho 1 \
    --ao_sep_basis \
    --act_txtdecrip \
    --semloss \
    --ho_pair_pt \
    --ho_pair_prior 1 \
    --pred_type  'ho+u+l' \
    --pt_init 'pos+detr' \
    --pt_attn 'uniform' \



# #### evaluation script for HICO-DET with ViT-B backbone
# CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
#     --pretrained checkpoints/detr-r50-hicodet.pth \
#     --output-dir checkpoints/hico_HO_adpt_uv_512_clipbase/lordhoi \
#     --num_classes 117 \
#     --use_multi_hot \
#     --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
#     --clip_dir_vit checkpoints/pretrained_CLIP/ViT-B-16.pt \
#     --port 1235 \
#     --logits_type "HO+U"  \
#     --llmtxt   \
#     --wo_unseen_pred \
#     --batch-size 16 \
#     --epoch 12 \
#     --zs --zs_type "unseen_verb" \
#     --use_insadapter \
#     --img_align   \
#     --self_adapt \
#     --basis_feat_enable \
#     --disentangle_basis \
#     --basis_feat_init 'pca' \
#     --recon_ratio_pca 71 \
#     --unique_basis_weights \
#     --seperate_ho 1 \
#     --ao_sep_basis \
#     --act_txtdecrip \
#     --semloss \
#     --ho_pair_pt \
#     --ho_pair_prior 1 \
#     --pred_type  'ho+u+l' \
#     --pt_init 'pos+detr' \
#     --pt_attn 'uniform' \
#     --eval --resume <path_to_your_model>
    