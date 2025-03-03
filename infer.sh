gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
CKPT=checkpoints/ft-only-con-epoch=20-step=50946-layer3.ckpt

# wave directory
WAVS=/home/getsum/data/DIHARD/third_dihard_challenge_eval/data/wav
# result directory
OUT=/home/getsum/code/temp/reclustering/tmp/osd

python $EXP_DIR/code/infer.py --exp_dir $EXP_DIR --checkpoint_name $CKPT --wav_dir $WAVS --out_dir $OUT --gpus $gpus 