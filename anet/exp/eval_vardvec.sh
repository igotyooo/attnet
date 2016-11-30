 nohup matlab -nodisplay -r 'eval_voc07_vggm_0_vardvec( 3, [ 5, 10 ] ); exit;' > eval_voc07_vggm_0_vardvec_5_10.out & \
 nohup matlab -nodisplay -r 'eval_voc07_vggm_0_vardvec( 4, [ 15, 30, 50 ] ); exit;' > eval_voc07_vggm_0_vardvec_15_30_50.out;

 nohup matlab -nodisplay -r 'eval_voc07_vgg16_0_vardvec( 1, 15 ); exit;' > eval_voc07_vgg16_0_vardvec_15.out & \
 nohup matlab -nodisplay -r 'eval_voc07_vgg16_0_vardvec( 2, [ 30, 50 ] ); exit;' > eval_voc07_vgg16_0_vardvec_30_50.out & \
 nohup matlab -nodisplay -r 'eval_voc07_vgg16_0_vardvec( 3, 5 ); exit;' > eval_voc07_vgg16_0_vardvec_5.out & \
 nohup matlab -nodisplay -r 'eval_voc07_vgg16_0_vardvec( 4, 10 ); exit;' > eval_voc07_vgg16_0_vardvec_10.out;
