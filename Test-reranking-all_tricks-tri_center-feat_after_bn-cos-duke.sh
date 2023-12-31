# Experiment all tricks without center loss with re-ranking : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 2: dukemtmc
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# with re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('dukemtmc')" TEST.RE_RANKING "('yes')"  DATASETS.ROOT_DIR "('/home/zwy/Baseline/dataset/')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/zwy/log/pga/dukemtmc/3all152/pga_resnet50_model_380.pth')"