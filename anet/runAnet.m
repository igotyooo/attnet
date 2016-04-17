%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all;
addpath( genpath( '..' ) ); init;
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
setting.train.gpus                              = 1 : 4;
setting.train.numCpu                            = 12;
setting.train.numSamplePerObj                   = [ 1; 14; 1; 16; ];
setting.train.shuffleSequance                   = false;
setting.train.useDropout                        = true;
setting.train.suppLearnRate                     = 1;
setting.train.learnRate                         = [ 0.01 * ones( 1, 7 ), 0.001 * ones( 1, 2 ), 0.0001 * ones( 1, 1 ) ] / 10;
setting.train.batchSize                         = numel( setting.train.gpus ) * 24;
setting.anetProp.gpu                            = 1;
setting.anetProp.flip                           = false;
setting.anetProp.numScaling                     = setting.anetdb.numScaling;
setting.anetProp.dilate                         = setting.anetdb.dilate * 2;
setting.anetProp.normalizeImageMaxSide          = setting.anetdb.normalizeImageMaxSide;
setting.anetProp.maximumImageSize               = setting.anetdb.maximumImageSize;
setting.anetProp.posGotoMargin                  = setting.anetdb.posGotoMargin;
setting.anetProp.numTopClassification           = 1;
setting.anetProp.numTopDirection                = 1;
setting.anetProp.directionVectorSize            = 30;
setting.anetProp.minNumDetectionPerClass        = 0;
setting.anetDet0.batchSize                      = setting.train.batchSize * 2;
setting.anetDet0.type                           = 'DYNAMIC';
setting.anetDet0.rescaleBox                     = 1;
setting.anetDet0.numTopClassification           = setting.anetProp.numTopClassification;
setting.anetDet0.numTopDirection                = setting.anetProp.numTopDirection;
setting.anetDet0.directionVectorSize            = setting.anetProp.directionVectorSize;
setting.anetDet0.minNumDetectionPerClass        = 0;
setting.anetDet0.weightDirection                = 0.5;
setting.anetMrg0.mergingOverlap                 = 0.8;
setting.anetMrg0.mergingType                    = 'NMS';
setting.anetMrg0.mergingMethod                  = 'MAX';
setting.anetMrg0.minimumNumSupportBox           = 1;
setting.anetMrg0.classWiseMerging               = true;
setting.anetDet1.batchSize                      = setting.train.batchSize * 2;
setting.anetDet1.type                           = 'STATIC';
setting.anetDet1.rescaleBox                     = 2.5;
setting.anetDet1.onlyTargetAndBackground        = true;
setting.anetDet1.directionVectorSize            = 15;
setting.anetDet1.minNumDetectionPerClass        = 1;
setting.anetDet1.weightDirection                = setting.anetDet0.weightDirection;
setting.anetMrg1.mergingOverlap                 = 0.5;
setting.anetMrg1.mergingType                    = 'OV';
setting.anetMrg1.mergingMethod                  = 'MAX';
setting.anetMrg1.minimumNumSupportBox           = 0;
setting.anetMrg1.classWiseMerging               = true;

%% DO THE JOB.
db = Db( setting.db, path.dstDir );
db.genDb;
adb = AnetDb( db, setting.anetdb );
adb.init;
adb = adb.makeAnetDb;
anet = AnetTrain( db, adb, ...
    setting.prenet, setting.train );
[ anet, res ] = anet.train;
det = Anet( db, anet, ...
    setting.anetProp, ...
    setting.anetDet0, ...
    setting.anetMrg0, ...
    setting.anetDet1, ...
    setting.anetMrg1 );
det.init;

%% EVAL.
clearvars -except db adb anet res det setting path;
res0 = det.getSubDbDet0( 1, 1 );
res0 = evalVoc07( res0, db, 2 );
res1 = det.getSubDbDet1( 1, 1 );
res1 = evalVoc07( res1, db, 2 );
for cid = 1 : numel( db.cid2name ),
    plot( res1.cid2rec{ cid }, res1.cid2prec{ cid }, ' - ' ); grid;
    xlabel( 'recall' ); ylabel( 'precision' );
    title( sprintf( '%s, AP = %.2f', db.cid2name{ cid }, res1.cid2ap( cid ) * 100 ) );
    pause;
end;

%% DEMO.
close all;
iid = db.getTeiids;
iid = randsample( iid', 1 );
det.demoDet( iid, true );