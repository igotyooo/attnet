classdef AnetTrain < handle
    properties
        db;
        anetdb;
        prenet;
        seq;
        gpus;
        numCpu;
        setting;
    end
    methods( Access = public )
        function this = AnetTrain( db, anetdb, prenet, setting )
            this.db = db;
            this.anetdb = anetdb;
            this.prenet = prenet;
            this.gpus = setting.gpus;
            this.numCpu = setting.numCpu;
            this.setting.numSamplePerObj = [ 1; 2; 1; 4; ];
            this.setting.shuffleSequance = false;
            this.setting.useDropout = true;
            this.setting.suppLearnRate = 0.1;
            this.setting.learnRate = logspace( -2, -4, 10 );
            this.setting.batchSize = 32;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function [ anet, info ] = train( this )
            anetPath = fullfile( this.getNetDir, 'net-deployed.mat' );
            infoPath = fullfile( this.getNetDir, 'info.mat' );
            try
                anet = load( anetPath );
                info = load( infoPath );
            catch
                % Form anet.
                anet = this.makeAnet;
                % Make initial seq.
                this.makeSeq;
                % Compute patch statistics.
                path = this.getPatchStatPath;
                try
                    data = load( path );
                    rgbMean = data.rgbMean;
                    rgbCovariance = data.rgbCovariance;
                catch
                    [ rgbMean, rgbCovariance ] = this.getImageStats( anet.meta );
                    save( path, 'rgbMean', 'rgbCovariance' );
                end;
                anet.meta.normalization.averageImage = rgbMean;
                [ v, d ] = eig( rgbCovariance );
                anet.meta.augmentation.rgbVariance = 0.1 * sqrt( d ) * v';
                % Learn.
                this.seq.makeSeq = @this.makeSeq;
                opts.train.gpus = this.gpus;
                opts.train.prefetch = true;
                opts.train.errorFunction = @this.errorFun;
                opts.train.errorLabels = { 'err' };
                [ anet, info ] = my_cnn_train( ...
                    anet, this.seq, ...
                    @( x, y )this.getBatch( anet.meta, x, y ), ...
                    'expDir', this.getNetDir, ...
                    anet.meta.trainOpts, ...
                    opts.train );
                % Deploy.
                fprintf( '%s: Save anet.\n', upper( mfilename ) );
                anet.meta.directions = this.anetdb.directions;
                anet.meta.directions.dimDir = anet.layers{ end }.dimDir;
                anet.meta.directions.dimCls = anet.layers{ end }.dimCls;
                anet.meta.name = this.getNetName;
                anet = cnn_imagenet_deploy( anet );
                anet.layers = anet.layers( 1 : end - 2 );
                for l = 1 : numel( anet.layers ),
                    if isfield( anet.layers{ l }, 'opts' ), ...
                            anet.layers{ l }.opts = {  }; end;
                end;
                save( anetPath, '-struct', 'anet' );
                save( infoPath, '-struct', 'info' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%
    % Private interface. %
    %%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        function [ rgbMean, rgbCovariance ] = getImageStats( this, meta )
            sids = find( this.seq.images.set == 1 );
            numSample = numel( sids );
            maxNumSample = 1e6;
            if numSample > maxNumSample,
                sel = randperm( numSample );
                sids = sort( sids( sel( 1 : maxNumSample ) ) );
                numSample = maxNumSample;
            end;
            bs = 256;
            fn = @( x, y )this.getBatch( meta, x, y );
            numiter = ( numSample - mod( numSample, bs ) ) / bs + 1;
            rgbm1 = cell( numiter, 1 );
            rgbm2 = cell( numiter, 1 );
            for t = 1 : bs : numSample,
                i = ( t - 1 ) / bs + 1;
                btime = tic;
                batch = sids( t : min( t + bs - 1, numel( sids ) ) );
                z = fn( this.seq, batch );
                z = reshape( permute( z, [ 3, 1, 2, 4 ] ), 3, [  ] );
                z = z( :, logical( sum( z, 1 ) ) );
                n = size( z, 2 );
                rgbm1{ i } = sum( z, 2 ) / n;
                rgbm2{ i } = z * z' / n;
                btime = toc( btime );
                fprintf( '%s: Compute patch stats: %d/%d, %.1fims/s\n', ...
                    upper( mfilename ), i, numiter, numel( batch ) / btime );
            end
            rgbm1 = mean( cat( 2, rgbm1{ : } ), 2 );
            rgbm2 = mean( cat( 3, rgbm2{ : } ), 3 );
            rgbMean = rgbm1;
            rgbCovariance = rgbm2 - rgbm1 * rgbm1';
        end
        function anet = makeAnet( this )
            % Set params.
            suppLearnRate = this.setting.suppLearnRate;
            useDropout = this.setting.useDropout;
            name = upper( this.prenet.name );
            numDimPerDirLyr = 5;
            numOutDim = numDimPerDirLyr * 2;
            % Initialize anet and copy params.
            fprintf( '%s: Form %s-based anet.\n', ...
                upper( mfilename ), name );
            ptnet = load( this.prenet.path );
            anet = cnn_imagenet_init( 'model', lower( name ) );
            for al = 1 : numel( anet.layers ),
                if isfield( anet.layers{ al }, 'weights' ),
                    altype = anet.layers{ al }.type;
                    alname = anet.layers{ al }.name;
                    for pl = 1 : numel( ptnet.layers ),
                        pltype = ptnet.layers{ pl }.type;
                        plname = ptnet.layers{ pl }.name;
                        if strcmp( altype, pltype ) && strcmp( alname, plname ),
                            anet.layers{ al }.stride = ptnet.layers{ pl }.stride;
                            anet.layers{ al }.pad = ptnet.layers{ pl }.pad;
                            lastconv = al;
                            lastwei = anet.layers{ al }.weights;
                            anet.layers{ al }.weights = ptnet.layers{ pl }.weights;
                            ptnet.layers{ pl } = rmfield( ptnet.layers{ pl }, 'weights' );
                            if al < numel( anet.layers ) - 2,
                                anet.layers{ al }.learningRate = ...
                                    anet.layers{ al }.learningRate * suppLearnRate;
                            end;
                            fprintf( '%s: Copy %s param.\n', upper( mfilename ), plname );
                        end;
                    end;
                end;
            end;
            for al = 1 : numel( ptnet.layers ),
                assert( ~isfield( ptnet.layers{ al }, 'weights' ) );
            end;
            anet.layers{ lastconv }.weights{ 1 } = lastwei{ 1 }( :, :, :, 1 : numOutDim );
            anet.layers{ lastconv }.weights{ 2 } = lastwei{ 2 }( 1 : numOutDim );
            anet.layers{ lastconv }.learningRate = anet.layers{ lastconv }.learningRate * 2; % Compansation to 1/2 in loss.
            % Initialize the output layer.
            anet.layers{ end }.type = 'custom';
            anet.layers{ end }.forward = @AnetTrain.forward;
            anet.layers{ end }.backward = @AnetTrain.backward;
            % Set train options.
            anet.meta.trainOpts.learningRate = this.setting.learnRate;
            anet.meta.trainOpts.numEpochs = numel( this.setting.learnRate );
            anet.meta.trainOpts.batchSize = this.setting.batchSize;
            % Set meta.
            anet.meta.augmentation = rmfield( anet.meta.augmentation, 'transformation' );
            anet.meta.normalization = rmfield( anet.meta.normalization, 'border' );
            anet.meta.normalization = rmfield( anet.meta.normalization, 'keepAspect' );
            fprintf( '%s: Compute patch side and stride between in/out of %s.\n', ...
                upper( mfilename ), name );
            tmp.layers = anet.layers( 1 : lastconv );
            tmp.meta.normalization = anet.meta.normalization;
            [ anet.meta.map.patchSide, anet.meta.map.stride ] = getNetProperties( tmp );
            anet.classes.name = {  };
            anet.classes.name{ end + 1, 1 } = sprintf( 'top-left: go to down' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'top-left: go to right-down' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'top-left: go to right' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'top-left: stop' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'top-left: false' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'bottom-right: go to up' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'bottom-right: go to up-left' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'bottom-right: go to left' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'bottom-right: stop' );
            anet.classes.name{ end + 1, 1 } = sprintf( 'bottom-right: false' );
            anet.classes.description = anet.classes.name;
            anet.name = this.getNetName;
            % Remove dropout if unnecessary.
            if ~useDropout, anet.layers = ...
                    anet.layers( ~cellfun( @( l )strcmp( l.type, 'dropout' ), anet.layers ) ); end;
        end
        function [ ims, labels ] = getBatch( this, meta, foo, batch )
            numThreads = this.numCpu;
            imageSize = meta.normalization.imageSize;
            rgbMean = meta.normalization.averageImage;
            rgbVar = meta.augmentation.rgbVariance;
            interpolation = meta.normalization.interpolation;
            inside = imageSize( 1 );
            if isempty( batch ),
                ims = zeros( inside, inside, 3, 0, 'single' );
                labels = zeros( 1, 1, size( this.seq.sid2gt, 3 ), 0, 'single' );
                return;
            end;
            if ~isempty( rgbVar ) && isempty( rgbMean ),
                rgbMean = zeros( 1, 1, 3 );
            end;
            if numel( rgbMean ) == 3,
                rgbMean = reshape( rgbMean, 1, 1, 3 );
            end;
            if isempty( this.seq ), this.makeSeq; end;
            iid2impath = this.seq.iid2impath;
            sid2iid = this.seq.sid2iid( batch );
            sid2tlbr = this.seq.sid2tlbr( :, batch );
            sid2flip = this.seq.sid2flip( batch );
            sid2gt = this.seq.sid2gt( :, batch );
            [ idx2iid, ~, sid2idx ] = unique( sid2iid );
            idx2impath = iid2impath( idx2iid );
            fetch = ischar( idx2impath{ 1 } );
            prefetch = fetch & nargout == 0;
            if prefetch,
                vl_imreadjpeg( idx2impath, 'numThreads', numThreads, 'prefetch' );
                ims = zeros( inside, inside, 3, 0, 'single' );
                labels = zeros( 1, 1, size( this.seq.sid2gt, 3 ), 0, 'single' );
                return;
            end;
            if fetch,
                idx2im = vl_imreadjpeg( idx2impath, 'numThreads', numThreads );
            else
                idx2im = images;
            end;
            numSmpl = numel( sid2iid );
            sid2im = zeros( inside, inside, 3, numSmpl, 'single' );
            for sid = 1 : numSmpl;
                wind = sid2tlbr( :, sid );
                im = idx2im{ sid2idx( sid ) };
                if ~isempty( rgbMean ) && ~isempty( rgbVar ),
                    im = normalizeAndCropImage( im, wind, rgbMean, rgbVar );
                elseif ~isempty( rgbMean )
                    im = normalizeAndCropImage( im, wind, rgbMean );
                else
                    im = normalizeAndCropImage( im, wind );
                end;
                if sid2flip( sid ), im = fliplr( im ); end;
                sid2im( :, :, :, sid ) = imresize...
                    ( im, [ inside, inside ], interpolation );
            end;
            ims = sid2im;
            labels = reshape( sid2gt, [ 1, 1, size( sid2gt, 1 ), numSmpl ] );
        end
        function makeSeq( this )
            fprintf( '%s: Make train seq.\n', upper( mfilename ) );
            trseq = this.getSeqPerEpch( 1 );
            fprintf( '%s: Make val seq.\n', upper( mfilename ) );
            valseq = this.getSeqPerEpch( 2 );
            this.seq.iid2impath = cat( 1, trseq.iid2impath, valseq.iid2impath );
            this.seq.sid2iid = cat( 1, trseq.sid2iid, valseq.sid2iid + numel( trseq.iid2impath ) );
            this.seq.sid2tlbr = cat( 2, trseq.sid2tlbr, valseq.sid2tlbr );
            this.seq.sid2flip = cat( 1, trseq.sid2flip, valseq.sid2flip );
            this.seq.sid2gt = cat( 2, trseq.sid2gt, valseq.sid2gt );
            this.seq.images.set = cat( 1, ones( size( trseq.sid2iid ) ), 2 * ones( size( valseq.sid2iid ) ) );
        end
        function subseq = getSeqPerEpch( this, setid )
            % Setting.
            if setid == 1, 
                subPatchdb = this.anetdb.tr; 
            else
                subPatchdb = this.anetdb.val; 
            end;
            shuffleSequance = this.setting.shuffleSequance;
            numSamplePerObj = this.setting.numSamplePerObj;
            dpid2dp = this.anetdb.directions.dpid2dp;
            % Do the job.
            numGo = numSamplePerObj( 1 );
            numAnyDir = numSamplePerObj( 2 );
            numStop = numSamplePerObj( 3 );
            numBgd = numSamplePerObj( 4 );
            numObj = numel( subPatchdb.oid2iid );
            numAugPerObj = sum( numSamplePerObj );
            numSample = numObj * numAugPerObj;
            bgdClsId = max( subPatchdb.oid2cid ) + 1;
            dpidBasis = 4 .^ ( 1 : -1 : 0 );
            dpidAllGo = dpidBasis * ( [ 2; 2; ] - 1 ) + 1;
            dpidAllStop = dpidBasis * ( [ 4; 4; ] - 1 ) + 1;
            numLyr = max( subPatchdb.oid2cid ) * 2 + 1;
            sid2iid = zeros( numSample, 1, 'single' );
            sid2tlbr = zeros( 4, numSample, 'single' );
            sid2flip = zeros( numSample, 1, 'single' );
            sid2gt = zeros( numLyr, numSample, 'single' );
            sid = 0;
            for oid = 1 : numObj,
                iid = subPatchdb.oid2iid( oid );
                cid = subPatchdb.oid2cid( oid );
                lids = [ cid * 2 - 1, cid * 2 ];
                % Sample positive regions - for initial proposal.
                for n = 1 : numGo,
                    dpid = dpidAllGo;
                    flip = round( rand );
                    if flip,
                        regns = subPatchdb.oid2dpid2posregnsFlip{ oid }{ dpid };
                    else
                        regns = subPatchdb.oid2dpid2posregns{ oid }{ dpid };
                    end;
                    numRegn = size( regns, 2 );
                    if numRegn,
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( lids, sid ) = dpid2dp( :, dpid );
                        sid2gt( end, sid ) = cid;
                    end;
                end;
                % Sample positive regions - for various directions.
                for n = 1 : numAnyDir,
                    flip = round( rand );
                    if flip,
                        dpid2posregns = subPatchdb.oid2dpid2posregnsFlip{ oid };
                    else
                        dpid2posregns = subPatchdb.oid2dpid2posregns{ oid };
                    end;
                    dpid2ok = ~cellfun( @isempty, dpid2posregns );
                    dpid2ok( dpidAllGo ) = false;
                    dpid2ok( dpidAllStop ) = false;
                    if sum( dpid2ok ),
                        dpid = find( dpid2ok );
                        dpid = dpid( ceil( numel( dpid ) * rand ) );
                        regns = dpid2posregns{ dpid };
                        numRegn = size( regns, 2 );
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( lids, sid ) = dpid2dp( :, dpid );
                        sid2gt( end, sid ) = cid;
                    end;
                end;
                % Sample positive regions - for stop.
                for n = 1 : numStop,
                    dpid = dpidAllStop;
                    flip = round( rand );
                    if flip,
                        regns = subPatchdb.oid2dpid2posregnsFlip{ oid }{ dpid };
                    else
                        regns = subPatchdb.oid2dpid2posregns{ oid }{ dpid };
                    end;
                    numRegn = size( regns, 2 );
                    if numRegn,
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = flip;
                        sid2gt( lids, sid ) = dpid2dp( :, dpid );
                        sid2gt( end, sid ) = cid;
                    end;
                end;
                % Sample negative regions.
                for n = 1 : numBgd,
                    regns = subPatchdb.iid2sid2negregns{ iid };
                    s2ok = ~cellfun( @isempty, regns );
                    if sum( s2ok ),
                        s = find( s2ok );
                        s = s( ceil( numel( s ) * rand ) );
                        regns = regns{ s };
                        numRegn = size( regns, 2 );
                        ridx = ceil( rand * numRegn );
                        sid = sid + 1;
                        sid2iid( sid ) = iid;
                        sid2tlbr( :, sid ) = regns( :, ridx );
                        sid2flip( sid ) = round( rand );
                        sid2gt( end, sid ) = bgdClsId;
                    end;
                end;
            end;
            if shuffleSequance,
                sids = randperm( numSample )';
                sid2iid = sid2iid( sids );
                sid2tlbr = sid2tlbr( :, sids );
                sid2gt = sid2gt( :, sids );
            end;
            subseq.iid2impath = subPatchdb.iid2impath;
            subseq.sid2iid = sid2iid;
            subseq.sid2tlbr = sid2tlbr;
            subseq.sid2flip = sid2flip;
            subseq.sid2gt = sid2gt;
        end
        function err = errorFun( this, foo, gts, res )
            output = gather( res( end - 1 ).x );
            gts = gather( gts );
            % Compute direction error.
            tcid = 15;
            numDimPerLyr = 5;
            signBgd = 5;
            numDirLyr = 2;
            errDir = 0;
            for lid = 1 : numDirLyr,
                gt = gts( :, :, ( tcid - 1 ) * numDirLyr + lid, : );
                gt( gt == 0 ) = signBgd;
                dims = ( lid - 1 ) * numDimPerLyr + 1;
                dime = lid * numDimPerLyr;
                p = output( :, :, dims : dime, : );
                [ ~, p ] = sort( p, 3, 'descend' );
                e = ~bsxfun( @eq, p, gt );
                e = e( :, :, 1, : );
                e = sum( e( : ) );
                errDir = errDir + e;
            end;
            errDir = errDir / 2; % Normalization in Eq (2).
            err = errDir;
        end
        % Functions for file IO.
        function name = getPatchStatName( this )
            numSamplePerObj = this.setting.numSamplePerObj;
            name = sprintf( 'PSTAT_%s_OF_%s', ...
                mat2str( numSamplePerObj( : )' ), this.anetdb.name );
            name( name == ' ' ) = '_';
            name( name == '[' ) = '';
            name( name == ']' ) = '';
        end
        function dir = getPatchStatDir( this )
            dir = this.db.getDir;
        end
        function dir = makePatchStatDir( this )
            dir = this.getPatchStatDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getPatchStatPath( this )
            fname = strcat( this.getPatchStatName, '.mat' );
            path = fullfile( this.getPatchStatDir, fname );
        end
        function name = getNetName( this )
            name = sprintf( 'ANET_%s_%s_OF_%s', ...
                upper( this.prenet.name ), this.setting.changes, this.anetdb.name );
            name( strfind( name, '-' ) ) = '';
	    name( strfind( name, ';' ) ) = '_';
	    name( strfind( name, '[' ) ) = '_';
	    name( strfind( name, ']' ) ) = '_';
        name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getNetDir( this )
            dir = fullfile( this.db.getDir, this.getNetName );
        end
        function dir = makeNetDir( this )
            dir = this.getNetDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
    end
    methods( Static )
        function res2 = forward( ly, res1, res2 )
            X = res1.x;
            gts = ly.class;
            tcid = 15;
            numDimPerLyr = 5;
            signBgd = 5;
            numDirLyr = 2;
            ydir = 0;
            for lid = 1 : numDirLyr,
                gt = gts( :, :, ( tcid - 1 ) * numDirLyr + lid, : );
                gt( gt == 0 ) = signBgd;
                dims = ( lid - 1 ) * numDimPerLyr + 1;
                dime = lid * numDimPerLyr;
                ydir_ = vl_nnsoftmaxloss( X( :, :, dims : dime, : ), gt );
                ydir = ydir + ydir_;
            end;
            ydir = ydir / 2;
            res2.x = ydir;
        end
        function res1 = backward( ly, res1, res2 )
            X = res1.x;
            gts = ly.class;
            Y = gpuArray( zeros( size( X ), 'single' ) );
            tcid = 15;
            numDimPerLyr = 5;
            signBgd = 5;
            numDirLyr = 2;
            dzdyDir = res2.dzdx / 2;
            for lid = 1 : numDirLyr,
                gt = gts( :, :, ( tcid - 1 ) * numDirLyr + lid, : );
                gt( gt == 0 ) = signBgd;
                dims = ( lid - 1 ) * numDimPerLyr + 1;
                dime = lid * numDimPerLyr;
                ydir = vl_nnsoftmaxloss...
                    ( X( :, :, dims : dime, : ), gt, dzdyDir );
                Y( :, :, dims : dime, : ) = ydir;
            end;
            res1.dzdx = Y;
        end
    end
end
