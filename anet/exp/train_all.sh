echo "single-class anet training";
echo "- train vgg16 model";
nohup matlab -nodesktop < train_voc07_vgg16.m > train_voc07_vgg16.out;
nohup matlab -nodesktop < train_voc12_vgg16.m > train_voc12_vgg16.out;
echo "- train vggm model in parallel";
nohup matlab -nodesktop < train_voc07_vggm.m > train_voc07_vggm.out &
nohup matlab -nodesktop < train_voc12_vggm.m > train_voc12_vggm.out;
echo "all done";
