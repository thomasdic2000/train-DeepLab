#code/.build_release/tools/caffe.bin train --solver=exper/pools/config/DeepLab-LargeFOV/solver_train.prototxt --gpu=0 --weights=exper/pools/model/DeepLab-LargeFOV/init.caffemodel
#code/.build_release/tools/caffe.bin train --solver=exper/roofs_new/config/DeepLab-LargeFOV/solver_train.prototxt --gpu=0 --weights=exper/roofs_new/model/DeepLab-LargeFOV/init.caffemodel
#code/.build_release/tools/caffe.bin train --solver=exper/roofs_Micaiah/config/DeepLab-LargeFOV/solver_train.prototxt --gpu=0 --weights=exper/roofs_Micaiah/model/DeepLab-LargeFOV/init.caffemodel
#code/.build_release/tools/caffe.bin train --solver=exper/roofs_stitchesZoom19/config/DeepLab-LargeFOV/solver_train.prototxt --gpu=0 --weights=exper/roofs_stitchesZoom19/model/DeepLab-LargeFOV/init.caffemodel
code/.build_release/tools/caffe.bin train --solver=exper/roofs_stitchesZoom20/config/DeepLab-LargeFOV/solver_train.prototxt --gpu=0 --weights=exper/roofs_stitchesZoom20/model/DeepLab-LargeFOV/init.caffemodel
