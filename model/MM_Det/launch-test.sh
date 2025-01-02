CUDA_VISIBLE_DEVICES=3 \
python test.py \
    -d /home/dongryeol/frame_diffusion_detection/MM_Det/reconstruction/opensora/0_real \
    --classes inference  \
    --ckpt weights/MM-Det/current_model.pth \
    --mm-root  /home/dongryeol/frame_diffusion_detection/MM_Det/outputs/mm_representations/0_real/original/mm_representation.pth \
    --cache-mm \
    --sample-size -1