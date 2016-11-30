clc; clearvars -except numDiv divId gpuId; fclose all; close all;
addpath( genpath( '../..' ) ); init;
setting.db                                      = path.db.voc2007;
setting.prenet                                  = path.net.vgg16;
setting.anetdb.patchSide                        = 224;
setting.anetdb.stride                           = 32;
setting.anetdb.numScaling                       = 24;
setting.anetdb.dilate                           = 1 / 4;
setting.anetdb.normalizeImageMaxSide            = 500;
setting.anetdb.maximumImageSize                 = 9e6;
setting.anetdb.posGotoMargin                    = 2.4;
setting.anetdb.numQuantizeBetweenStopAndGoto    = 3;
setting.anetdb.negIntOverObjLessThan            = 0.1;
setting.train.gpus                              = 1;
setting.train.numCpu                            = 12;
setting.train.numSamplePerObj                   = [ 1; 14; 1; 16; ];
setting.train.shuffleSequance                   = false;
setting.train.useDropout                        = true;
setting.train.suppLearnRate                     = 1;
setting.train.learnRate                         = [ 0.01 * ones( 1, 10 ), 0.001 * ones( 1, 2 ), 0.0001 * ones( 1, 1 ) ] / 10;
setting.train.batchSize                         = 96;
setting.anetProp.gpu                            = 0;
setting.anetProp.flip                           = false;
setting.anetProp.numScaling                     = setting.anetdb.numScaling;
setting.anetProp.dilate                         = setting.anetdb.dilate * 2;
setting.anetProp.normalizeImageMaxSide          = setting.anetdb.normalizeImageMaxSide;
setting.anetProp.maximumImageSize               = setting.anetdb.maximumImageSize;
setting.anetProp.posGotoMargin                  = setting.anetdb.posGotoMargin;
setting.anetProp.directionVectorSize            = 30;
setting.anetDet0.batchSize                      = setting.train.batchSize * 2;
setting.anetDet0.rescaleBox                     = 1;
setting.anetDet0.rescaleBoxStd                  = 0;
setting.anetDet0.directionVectorSize            = setting.anetProp.directionVectorSize;
setting.anetMrg0.mergingOverlap                 = 1;
setting.anetMrg0.mergingType                    = 'NMS';
setting.anetMrg0.mergingMethod                  = 'MAX';
setting.anetMrg0.minimumNumSupportBox           = 0;
setting.anetDet1.batchSize                      = setting.train.batchSize * 2;
setting.anetDet1.rescaleBox                     = 3;
setting.anetDet1.rescaleBoxStd                  = 0.5;
setting.anetDet1.directionVectorSize            = 15;
setting.anetMrg1.mergingOverlap                 = 0.5;
setting.anetMrg1.mergingType                    = 'NMSIOU';
setting.anetMrg1.mergingMethod                  = 'WAVG';
setting.anetMrg1.minimumNumSupportBox           = 0;
db = Db( setting.db, path.dstDir );
db.genDb;
adb = AnetDb( db, setting.anetdb );
adb.init;
adb = adb.makeAnetDb;
anet = AnetTrain( db, adb, ...
    setting.prenet, setting.train );
anet = anet.train;
det = Anet( db, anet, ...
    setting.anetProp, ...
    setting.anetDet0, ...
    setting.anetMrg0, ...
    setting.anetDet1, ...
    setting.anetMrg1 );
det.init;
res1 = det.getSubDbDet1( 1, 1 );
res1.did2cid = 15 * ones( size( res1.did2score ) );
evalVoc( res1, db, 2 );
resultPath = fullfile( 'results', setting.db.name, 'Main/comp3_det_test_%s.txt' );
resultTarball = sprintf( '%s_DET1_%s_MO%.1f_MT%s_MM%s_MNSB%d.tar', ...
    setting.db.name, ...
    setting.prenet.name, ...
    setting.anetMrg1.mergingOverlap, ...
    setting.anetMrg1.mergingType, ...
    setting.anetMrg1.mergingMethod, ...
    setting.anetMrg1.minimumNumSupportBox );
resultTarball( resultTarball( 1 : end - 4 ) == '.' ) = 'P';
resultTarball( resultTarball( 1 : end - 4 ) == '-' ) = '';
system( sprintf( 'mkdir -p %s', fileparts( resultPath ) ) );
fprintf( 'Make result file.\n' );
makeResultFiles( res1, db.cid2name, db.iid2impath, resultPath );
fprintf( 'Done.\n' );
fprintf( 'Compress.\n' );
system( sprintf( 'tar cvf %s results', resultTarball ) );
system( 'rm -r results' );
fprintf( 'Done.\n' );