# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
python3 tools/train.py --config_file='configs/softmax_triplet_with_center_pga.yml' SOLVER.MAX_EPOCHS "(600)"  MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/zwy/dataset/')" MODEL.SGA_ON "('True')" MODEL.CGA_ON "('False')" MODEL.NUM_HEADS "(1)" OUTPUT_DIR "('/home/zwy/log/pga/market1501/test')"