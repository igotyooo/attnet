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
setting.train.learnRate                         = [ 0.01 * ones( 1, 10 ), 0.001 * ones( 1, 2 ), 0.0001 * ones( 1, 1 ) ] / 10;
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
setting.anetDet0.batchSize                      = setting.train.batchSize * 2;
setting.anetDet0.type                           = 'DYNAMIC';
setting.anetDet0.rescaleBox                     = 1;
setting.anetDet0.rescaleBoxStd                  = 0;
setting.anetDet0.numTopClassification           = setting.anetProp.numTopClassification;
setting.anetDet0.numTopDirection                = setting.anetProp.numTopDirection;
setting.anetDet0.directionVectorSize            = setting.anetProp.directionVectorSize;
setting.anetMrg0.mergingOverlap                 = 1;
setting.anetMrg0.mergingType                    = 'NMS';
setting.anetMrg0.mergingMethod                  = 'MAX';
setting.anetMrg0.minimumNumSupportBox           = 0;
setting.anetMrg0.classWiseMerging               = true;
setting.anetDet1.batchSize                      = setting.train.batchSize * 2;
setting.anetDet1.type                           = 'STATIC';
setting.anetDet1.rescaleBox                     = 3;
setting.anetDet1.rescaleBoxStd                  = 0.5;
setting.anetDet1.onlyTargetAndBackground        = false;
setting.anetDet1.directionVectorSize            = 15;
setting.anetMrg1.mergingOverlap                 = 0.45;
setting.anetMrg1.mergingType                    = 'NMSIOU';
setting.anetMrg1.mergingMethod                  = 'WAVG';
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

%% EVALUATION.
clearvars -except db adb anet res det setting path;
res1 = det.getSubDbDet1( 1, 1 );
res1 = evalVoc( res1, db, 2 );
for cid = 1 : numel( db.cid2name ),
    plot( res1.cid2rec{ cid }, res1.cid2prec{ cid }, '-' ); grid;
    xlabel( 'recall' ); ylabel( 'precision' );
    title( sprintf( '%s, AP = %.2f', db.cid2name{ cid }, res1.cid2ap( cid ) * 100 ) );
    waitforbuttonpress;
end;

%% DEMO.
close all;
iid = db.getTeiids;
iid = randsample( iid', 1 );
det.demoDet( iid, true );

%% ANALYSIS.
close all;
clearvars -except db adb anet res det setting path res1;
cname = 'motorbike';
cid = find( cellfun( @( s )strcmp( s, cname ), db.cid2name ) );
fpranks = find( res1.cid2rank2fp{ cid } );
for f = 1 : numel( fpranks ),
    r = fpranks( f );
    s = res1.cid2rank2score{ cid }( r );
    b = res1.cid2rank2tlbr{ cid }( :, r );
    i = res1.cid2rank2iid{ cid }( r );
    im = db.iid2impath{ i };
    figure( 1 );
    plot( res1.cid2rec{ cid }, res1.cid2prec{ cid }, '-' ); grid; hold on; drawnow;
    plot( res1.cid2rec{ cid }( r ), res1.cid2prec{ cid }( r ), 'ro' ); 
    title( sprintf( '%s, AP = %.2f', db.cid2name{ cid }, res1.cid2ap( cid ) * 100 ) );
    hold off;
    waitforbuttonpress;
    figure( 2 );
    plottlbr( b, im, false, 'r', { sprintf( '%s: %.2f', cname, s ); } );
    title( sprintf( 'IID%06d', i ) );
    drawnow; hold off;
    waitforbuttonpress;
    figure( 3 );
    plottlbr( db.oid2bbox( :, db.oid2iid == i ), im, false, 'b', db.cid2name( db.oid2cid( db.oid2iid == i ) ) );
    title( sprintf( 'IID%06d', i ) );
    drawnow; hold off;
    waitforbuttonpress;
end;
