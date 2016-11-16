echo "multi-class anet test";
echo "- test vgg16 model on voc07 in parallel";
nohup matlab -r 'test_voc07_vgg16( 4, 1, 1 )' > test_voc07_vgg16_1.out &
nohup matlab -r 'test_voc07_vgg16( 4, 2, 2 )' > test_voc07_vgg16_2.out &
nohup matlab -r 'test_voc07_vgg16( 4, 3, 3 )' > test_voc07_vgg16_3.out &
nohup matlab -r 'test_voc07_vgg16( 4, 4, 4 )' > test_voc07_vgg16_4.out;
echo "- test vgg16 model on voc12 in parallel";
nohup matlab -r 'test_voc12_vgg16( 4, 1, 1 )' > test_voc12_vgg16_1.out &
nohup matlab -r 'test_voc12_vgg16( 4, 2, 2 )' > test_voc12_vgg16_2.out &
nohup matlab -r 'test_voc12_vgg16( 4, 3, 3 )' > test_voc12_vgg16_3.out &
nohup matlab -r 'test_voc12_vgg16( 4, 4, 4 )' > test_voc12_vgg16_4.out;
echo "- test vggm model on voc07 in parallel";
nohup matlab -r 'test_voc07_vggm( 4, 1, 1 )' > test_voc07_vggm_1.out &
nohup matlab -r 'test_voc07_vggm( 4, 2, 2 )' > test_voc07_vggm_2.out &
nohup matlab -r 'test_voc07_vggm( 4, 3, 3 )' > test_voc07_vggm_3.out &
nohup matlab -r 'test_voc07_vggm( 4, 4, 4 )' > test_voc07_vggm_4.out;
echo "- test vggm model on voc12 in parallel";
nohup matlab -r 'test_voc12_vggm( 4, 1, 1 )' > test_voc12_vggm_1.out &
nohup matlab -r 'test_voc12_vggm( 4, 2, 2 )' > test_voc12_vggm_2.out &
nohup matlab -r 'test_voc12_vggm( 4, 3, 3 )' > test_voc12_vggm_3.out &
nohup matlab -r 'test_voc12_vggm( 4, 4, 4 )' > test_voc12_vggm_4.out;
