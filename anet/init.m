% Set lib path only.
global path;
path.lib.matConvNet = '/home/dgyoo/workspace/lib/matconvnet-1.0-beta21_cudnn4/';
path.lib.vocDevKit = '/home/dgyoo/workspace/datain/PASCALVOC/VOCdevkit/';
% Set dst dir.
path.dstDir = '/home/dgyoo/workspace/dataout/attnet';
% Set image DB path only.
path.db.voc2007.name = 'VOC2007';
path.db.voc2007.funh = @DB_VOC2007;
path.db.voc2007.root = fullfile( path.lib.vocDevKit, 'VOC2007' );
path.db.voc2012.name = 'VOC2012';
path.db.voc2012.funh = @DB_VOC2012;
path.db.voc2012.root = fullfile( path.lib.vocDevKit, 'VOC2012' );
path.db.voc2007and2012.name = 'VOC2007AND2012';
path.db.voc2007and2012.funh = @DB_VOC2007AND2012;
% Set pre-trained CNN path only.
path.net.vgg_m.name = 'VGG-M';
path.net.vgg_m.path = '/home/dgyoo/workspace/nets/mat/imagenet-vgg-m.mat';
path.net.vgg16.name = 'VGG-VD-16';
path.net.vgg16.path = '/home/dgyoo/workspace/nets/mat/imagenet-vgg-verydeep-16.mat';
% Do not touch the following codes.
run( fullfile( path.lib.matConvNet, 'matlab/vl_setupnn.m' ) );      % MatConvnet.
addpath( genpath( fullfile( path.lib.matConvNet, 'examples' ) ) );  % MatConvnet.
addpath( fullfile( path.lib.vocDevKit, 'VOCcode' ) );               % VOC dev kit.