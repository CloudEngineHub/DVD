CKPT='ckpt/DVD'
FRAME_NUM=100 # 200,300
NUM=1
VIDEO_BASE_DATA_DIR='your_video_base_data_dir'

python test_script/test_from_trained_all_video.py --ckpt $CKPT --frame_num $FRAME_NUM --num $NUM --video_base_data_dir $VIDEO_BASE_DATA_DIR