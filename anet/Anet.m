classdef Anet < handle
    properties
        db;
        anet;
        scales;
        settingProp;
        settingDet0;
        settingMrg0;
        settingDet1;
        settingMrg1;
    end
    methods( Access = public )
        function this = Anet...
                ( db, anet, settingProp, settingDet0, settingMrg0, settingDet1, settingMrg1 )
            this.db                                   = db;
            this.anet                                 = anet;
            this.settingProp.gpu                      = settingProp.gpu;
            this.settingProp.flip                     = false;
            this.settingProp.numScaling               = 12;
            this.settingProp.dilate                   = 1 / 4;
            this.settingProp.normalizeImageMaxSide    = 500;
            this.settingProp.maximumImageSize         = 9e6;
            this.settingProp.posGotoMargin            = 2;
            this.settingProp.numTopClassification     = 1;
            this.settingProp.numTopDirection          = 1;
            this.settingProp.onlyTargetAndBackground  = false;
            this.settingProp.directionVectorSize      = 30;
            this.settingDet0.batchSize                = settingDet0.batchSize;
            this.settingDet0.type                     = 'DYNAMIC';
            this.settingDet0.rescaleBox               = 1;
            this.settingDet0.numTopClassification     = 1;          % Ignored if 'STATIC'.
            this.settingDet0.numTopDirection          = 1;          % Ignored if 'STATIC'.
            this.settingDet0.onlyTargetAndBackground  = false;      % Ignored if 'DYNAMIC'.
            this.settingDet0.directionVectorSize      = 30;
            this.settingMrg0.weightDirection 	      = 0.5;
            this.settingMrg0.mergingOverlap           = 0.8;
            this.settingMrg0.mergingType              = 'OV';
            this.settingMrg0.mergingMethod            = 'WAVG';
            this.settingMrg0.minimumNumSupportBox     = 1;          % Ignored if mergingOverlap = 1.
            this.settingMrg0.classWiseMerging         = true;
            this.settingDet1.batchSize                = settingDet1.batchSize;
            this.settingDet1.type                     = 'DYNAMIC';
            this.settingDet1.rescaleBox               = 2.5;
            this.settingDet1.numTopClassification     = 1;          % Ignored if 'STATIC'.
            this.settingDet1.numTopDirection          = 1;          % Ignored if 'STATIC'.
            this.settingDet1.onlyTargetAndBackground  = false;      % Ignored if 'DYNAMIC'.
            this.settingDet1.directionVectorSize      = 30;
            this.settingMrg1.weightDirection          = 0.5;
            this.settingMrg1.mergingOverlap           = 0.6;
            this.settingMrg1.mergingType              = 'OV';
            this.settingMrg1.mergingMethod            = 'WAVG';
            this.settingMrg1.minimumNumSupportBox     = 0;          % Ignored if mergingOverlap = 1.
            this.settingMrg1.classWiseMerging         = true;
            this.settingProp = setChanges...
                ( this.settingProp, settingProp, upper( mfilename ) );
            this.settingDet0 = setChanges...
                ( this.settingDet0, settingDet0, upper( mfilename ) );
            this.settingMrg0 = setChanges...
                ( this.settingMrg0, settingMrg0, upper( mfilename ) );
            this.settingDet1 = setChanges...
                ( this.settingDet1, settingDet1, upper( mfilename ) );
            this.settingMrg1 = setChanges...
                ( this.settingMrg1, settingMrg1, upper( mfilename ) );
        end
        function init( this )
            % Determine scaling factors.
            patchSide = this.anet.meta.map.patchSide;
            fpath = this.getScaleFactorPath;
            try
                fprintf( '%s: Try to load scaling factors.\n', upper( mfilename ) );
                data = load( fpath );
                this.scales = data.data.scales;
            catch
                fprintf( '%s: Determine scaling factors.\n', ...
                    upper( mfilename ) );
                posGotoMargin = this.settingProp.posGotoMargin;
                maxSide = this.settingProp.normalizeImageMaxSide;
                numScaling = this.settingProp.numScaling;
                posIntOverRegnMoreThan = 1 / ( posGotoMargin ^ 2 );
                setid = 1;
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == setid );
                if maxSide,
                    oid2iid = this.db.oid2iid( this.db.iid2setid( this.db.oid2iid ) == setid );
                    oid2imsize = this.db.iid2size( :, oid2iid );
                    numRegn = size( oid2tlbr, 2 );
                    for oid = 1 : numRegn,
                        [ ~, oid2tlbr( :, oid ) ] = normalizeImageSize...
                            ( maxSide, oid2imsize( :, oid ), oid2tlbr( :, oid ) );
                    end;
                end;
                referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr, numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Fetch net on GPU.
            fprintf( '%s: Fetch anet on GPU.\n', upper( mfilename ) );
            gpuDevice( this.settingProp.gpu );
            this.anet = vl_simplenn_move( this.anet, 'gpu') ;
            fprintf( '%s: Done.\n', upper( mfilename ) );
        end
        function [ rid2tlbr, nid2rid, nid2cid ] = iid2prop( this, iid )
            cidx2cid = 1 : this.db.getNumClass;
            fpath = this.getPropPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                nid2rid = data.nid2rid;
                nid2cid = data.nid2cid;
            catch
                [ rid2tlbr, nid2rid, nid2cid ] = this.iid2propWrapper( iid, cidx2cid );
                this.makePropDir;
                save( fpath, 'rid2tlbr', 'nid2rid', 'nid2cid' );
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det0( this, iid )
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
                rid2cid = data.rid2cid;
            catch
                % 1. Get regions.
                [ rid2tlbr, nid2rid, nid2cid ] = this.iid2prop( iid );
                % 2. Tighten regions.
                [ rid2tlbr, rid2out, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2out', 'rid2cid' );
            end;
            if isempty( rid2tlbr ), rid2score = zeros( 0, 1 ); end;
            if nargout && ~isempty( rid2tlbr ),
                % 3. Merge regions.
                rid2score = this.scoring( rid2out, rid2cid );
                [ rid2tlbr, rid2score, rid2cid ] = this.merge...
                    ( rid2tlbr, rid2score, rid2cid, this.settingMrg0 );
                if isempty( rid2tlbr ), return; end;
                imbnd = [ 1; 1; this.db.iid2size( :, iid ); ];
                [ rid2tlbr, idx ] = bndtlbr( rid2tlbr, imbnd );
                if numel( rid2score ) ~= numel( idx ),
                    rid2tlbr_ = repmat( imbnd, 1, numel( rid2score ) );
                    rid2tlbr_( :, idx ) = rid2tlbr;
                    rid2tlbr = rid2tlbr_;
                end;
            end;
        end
        function [ rid2tlbr, rid2score, rid2cid ] = iid2det1( this, iid )
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
                rid2cid = data.rid2cid;
            catch
                % 1. Get regions.
                [ rid2tlbr, ~, rid2cid ] = this.iid2det0( iid );
                nid2rid = 1 : numel( rid2cid );
                nid2cid = rid2cid;
                % 2. Tighten regions.
                [ rid2tlbr, rid2out, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2out', 'rid2cid' );
            end;
            if isempty( rid2tlbr ), rid2score = zeros( 0, 1 ); end;
            if nargout && ~isempty( rid2tlbr ),
                % 3. Merge regions.
                rid2score = this.scoring( rid2out, rid2cid );
                [ rid2tlbr, rid2score, rid2cid ] = this.merge...
                    ( rid2tlbr, rid2score, rid2cid, this.settingMrg1 );
                if isempty( rid2tlbr ), return; end;
                imbnd = [ 1; 1; this.db.iid2size( :, iid ); ];
                [ rid2tlbr, idx ] = bndtlbr( rid2tlbr, imbnd );
                if numel( rid2score ) ~= numel( idx ),
                    rid2tlbr_ = repmat( imbnd, 1, numel( rid2score ) );
                    rid2tlbr_( :, idx ) = rid2tlbr;
                    rid2tlbr = rid2tlbr_;
                end;
            end;
        end
        function rid2score = scoring( this, rid2out, rid2cid )
            weightDirection = 0.8;
            signStop = 4;
            numCls = numel( this.db.cid2name );
            numDimPerDirLyr = 4;
            numDimClsLyr = numCls + 1;
            numDirDim = ( numDimPerDirLyr * 2 ) * numCls;
            dimCls = ( numDirDim + 1 ) : ( numDirDim + numDimClsLyr );
            rid2score = zeros( size( rid2out, 2 ), 1, 'single' );
            for cid = 1 : numCls,
                rid2tar = rid2cid == cid;
                if ~any( rid2tar ), continue; end;
                dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                dimBr = dimTl + numDimPerDirLyr;
                outsTl = rid2out( dimTl, rid2tar );
                outsBr = rid2out( dimBr, rid2tar );
                outsCls = rid2out( dimCls, rid2tar );
                % outsTl = exp( bsxfun( @minus, outsTl, max( outsTl, [  ], 1 ) ) );
                % outsBr = exp( bsxfun( @minus, outsBr, max( outsBr, [  ], 1 ) ) );
                % outsCls = exp( bsxfun( @minus, outsCls, max( outsCls, [  ], 1 ) ) );
                % outsTl = bsxfun( @times, outsTl, 1 ./ sum( outsTl, 1 ) );
                % outsBr = bsxfun( @times, outsBr, 1 ./ sum( outsBr, 1 ) );
                % outsCls = bsxfun( @times, outsCls, 1 ./ sum( outsCls, 1 ) );
                scoresTl = outsTl( signStop, : );
                scoresBr = outsBr( signStop, : );
                scoresCls = outsCls( cid, : ) - outsCls( end, : );
                rid2score( rid2tar ) = weightDirection * ( scoresTl + scoresBr )' / 2 + ...
                    ( 1 - weightDirection ) * scoresCls';
            end;
        end;
        function demoDet( this, iid, wait )
            flip = this.settingProp.flip;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            % Demo 1: proposals.
            [ rid2tlbr, nid2rid, nid2cid ] = this.iid2prop( iid );
            rid2tlbr_ = bndtlbr( rid2tlbr, scaleBoxes( rid2tlbr, 2, 2 ) );
            rid2tlbr_ = round( rid2tlbr_ );
            figure( 1 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr_, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Proposals, IID%06d (Boxes are bounded)', iid ) ); hold off; drawnow;
            % Demo 2: detection0.
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
                rid2cid = data.rid2cid;
            catch
                [ rid2tlbr, rid2out, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2out', 'rid2cid' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure( 2 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection0, IID%06d', iid ) ); hold off; drawnow;
            % Demo 3: Merge0.
            rid2score = this.scoring( rid2out, rid2cid );
            [ rid2tlbr, ~, rid2cid ] = this.merge...
                ( rid2tlbr, rid2score, rid2cid, this.settingMrg0 );
            imbnd = [ 1; 1; this.db.iid2size( :, iid ); ];
            [ rid2tlbr, idx ] = bndtlbr( rid2tlbr, imbnd );
            if numel( rid2cid ) ~= numel( idx ),
                rid2tlbr_ = repmat( imbnd, 1, numel( rid2score ) );
                rid2tlbr_( :, idx ) = rid2tlbr;
                rid2tlbr = rid2tlbr_;
            end;
            rid2tlbr = round( rid2tlbr );
            figure( 3 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, 'c', this.db.cid2name( rid2cid ) );
            title( sprintf( 'Merge0, IID%06d', iid ) ); hold off; drawnow;
            % Demo 4: detection1.
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
                rid2cid = data.rid2cid;
            catch
                nid2rid = 1 : numel( rid2cid );
                nid2cid = rid2cid;
                [ rid2tlbr, rid2out, rid2cid ] = this.iid2det...
                    ( iid, rid2tlbr, nid2rid, nid2cid, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2out', 'rid2cid' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure( 4 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection1, IID%06d', iid ) ); hold off; drawnow;
            % Demo 5: Merge1.
            rid2score = this.scoring( rid2out, rid2cid );
            [ rid2tlbr, rid2score, rid2cid ] = this.merge...
                ( rid2tlbr, rid2score, rid2cid, this.settingMrg1 );
            imbnd = [ 1; 1; this.db.iid2size( :, iid ); ];
            [ rid2tlbr, idx ] = bndtlbr( rid2tlbr, imbnd );
            if numel( rid2score ) ~= numel( idx ),
                rid2tlbr_ = repmat( imbnd, 1, numel( rid2score ) );
                rid2tlbr_( :, idx ) = rid2tlbr;
                rid2tlbr = rid2tlbr_;
            end;
            rid2tlbr = round( rid2tlbr );
            [ ~, rank2rid ] = sort( rid2score, 'descend' );
            rid2tlbr = rid2tlbr( :, rank2rid );
            rid2score = rid2score( rank2rid );
            rid2cid = rid2cid( rank2rid );
            cids = unique( rid2cid, 'stable' );
            cids = cids( : );
            figure( 5 ); set( gcf, 'color', 'w' );
            for cid = cids',
                rid2ok = rid2cid == cid;
                if ~wait, rid2ok = true( size( rid2cid ) ); end;
                rid2tlbr_ = rid2tlbr( :, rid2ok );
                rid2score_ = rid2score( rid2ok );
                rid2cid_ = rid2cid( rid2ok );
                rid2title_ = cell( size( rid2score_ ) );
                for rid = 1 : numel( rid2score_ ),
                    cname = this.db.cid2name{ rid2cid_( rid ) };
                    score = rid2score_( rid );
                    rid2title_{ rid } = sprintf( '%s(%.1f)', cname, score );
                end;
                plottlbr( rid2tlbr_, im, false, 'c', rid2title_ );
                title( sprintf( 'Merge1, IID%06d', iid ) ); hold off; drawnow;
                if ~wait, break; end;
                waitforbuttonpress;
            end;
        end
        function subDbDet0( this, numDiv, divId )
            iids = this.db.getTeiids;
            iids = iids( divId : numDiv : numel( iids ) );
            fprintf( '%s: Check if detections exist.\n', upper( mfilename ) );
            paths = arrayfun( @( iid )this.getDet0Path( iid ), iids, 'UniformOutput', false );
            exists = cellfun( @( path )exist( path, 'file' ), paths );
            if all( exists ), fprintf( '%s: All done.\n', upper( mfilename ) ); return; end;
            this.makeDet0Dir;
            iids = iids( ~exists );
            numIm = numel( iids );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2det0( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Det0 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
        end
        function subDbDet1( this, numDiv, divId )
            iids = this.db.getTeiids;
            iids = iids( divId : numDiv : numel( iids ) );
            fprintf( '%s: Check if detections exist.\n', upper( mfilename ) );
            paths = arrayfun( @( iid )this.getDet1Path( iid ), iids, 'UniformOutput', false );
            exists = cellfun( @( path )exist( path, 'file' ), paths );
            if all( exists ), fprintf( '%s: All done.\n', upper( mfilename ) ); return; end;
            this.makeDet1Dir;
            iids = iids( ~exists );
            numIm = numel( iids );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2det1( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Det1 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
        end
        function res = getSubDbDet0( this,  numDiv, divId )
            iids = this.db.getTeiids;
            idx2iid = iids( divId : numDiv : numel( iids ) );
            numIm = numel( idx2iid );
            rid2tlbr = cell( numIm, 1 );
            rid2score = cell( numIm, 1 );
            rid2cid = cell( numIm, 1 );
            rid2iid = cell( numIm, 1 );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = idx2iid( iidx );
                [ rid2tlbr{ iidx }, rid2score{ iidx }, rid2cid{ iidx } ] = this.iid2det0( iid );
                rid2iid{ iidx } = iid * ones( size( rid2score{ iidx } ) );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Get det0 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
            rid2tlbr = cat( 2, rid2tlbr{ : } );
            rid2score = cat( 1, rid2score{ : } );
            rid2cid = cat( 1, rid2cid{ : } );
            rid2iid = cat( 1, rid2iid{ : } );
            res.did2tlbr = rid2tlbr;
            res.did2score = rid2score;
            res.did2cid = rid2cid;
            res.did2iid = rid2iid;
        end;
        function res = getSubDbDet1( this,  numDiv, divId )
            iids = this.db.getTeiids;
            idx2iid = iids( divId : numDiv : numel( iids ) );
            numIm = numel( idx2iid );
            rid2tlbr = cell( numIm, 1 );
            rid2score = cell( numIm, 1 );
            rid2cid = cell( numIm, 1 );
            rid2iid = cell( numIm, 1 );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = idx2iid( iidx );
                [ rid2tlbr{ iidx }, rid2score{ iidx }, rid2cid{ iidx } ] = this.iid2det1( iid );
                rid2iid{ iidx } = iid * ones( size( rid2score{ iidx } ) );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Get det1 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
            rid2tlbr = cat( 2, rid2tlbr{ : } );
            rid2score = cat( 1, rid2score{ : } );
            rid2cid = cat( 1, rid2cid{ : } );
            rid2iid = cat( 1, rid2iid{ : } );
            res.did2tlbr = rid2tlbr;
            res.did2score = rid2score;
            res.did2cid = rid2cid;
            res.did2iid = rid2iid;
        end;
    end
    methods( Access = private )
        function [ rid2tlbr, nid2rid, nid2cid ] = iid2propWrapper( this, iid )
            % Initial guess.
            flip = this.settingProp.flip;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            [ rid2out, rid2tlbr ] = this.initGuess( im );
            % Compute each region score.
            patchSide = this.anet.meta.map.patchSide;
            dvecSize = this.settingProp.directionVectorSize;
            numTopCls = this.settingProp.numTopClassification;
            numTopDir = this.settingProp.numTopDirection;
            onlyTarAndBgd = this.settingProp.onlyTargetAndBackground;
            signStop = 4;
            signDiag = 2;
            numCls = numel( this.db.cid2name );
            numDimPerDirLyr = 4;
            numDimClsLyr = numCls + 1;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            rid2outCls = rid2out( dimCls, : );
            [ ~, rid2rank2cidx ] = sort( rid2outCls, 1, 'descend' );
            rid2tlbrProp = cell( numCls, 1 );
            for cid = 1 : numCls,
                % Direction: DD condition.
                dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                dimBr = dimTl + numDimPerDirLyr;
                rid2outTl = rid2out( dimTl, : );
                rid2outBr = rid2out( dimBr, : );
                [ ~, rid2rank2ptl ] = sort( rid2outTl, 1, 'descend' );
                [ ~, rid2rank2pbr ] = sort( rid2outBr, 1, 'descend' );
                rid2ptl = rid2rank2ptl( 1, : );
                rid2pbr = rid2rank2pbr( 1, : );
                rid2okTl = any( rid2rank2ptl( 1 : numTopDir, : ) == signDiag, 1 );
                rid2okBr = any( rid2rank2pbr( 1 : numTopDir, : ) == signDiag, 1 );
                rid2ss = rid2ptl == signStop & rid2pbr == signStop;
                rid2dd = rid2okTl & rid2okBr;
                rid2dd = rid2dd & ( ~rid2ss );
                rid2dd = rid2dd & ( rid2ptl == signDiag | rid2pbr == signDiag );
                % Classification.
                if onlyTarAndBgd,
                    rid2bgdScore = rid2outCls( numCls + 1, : );
                    rid2tarScore = rid2outCls( cid, : );
                    rid2okCls = rid2tarScore > rid2bgdScore;
                else
                    rid2bgd = rid2rank2cidx( 1, : ) == ( numCls + 1 );
                    rid2okCls = any( rid2rank2cidx( 1 : min( numTopCls, numDimClsLyr ), : ) == cid, 1 );
                    rid2okCls = rid2okCls & ( ~rid2bgd );
                end;
                % Update.
                rid2cont = rid2dd & rid2okCls;
                numCont = sum( rid2cont );
                if ~numCont, continue; end;
                idx2tlbr = rid2tlbr( 1 : 4, rid2cont );
                idx2ptl = rid2ptl( rid2cont );
                idx2pbr = rid2pbr( rid2cont );
                idx2tlbrWarp = [ ...
                    this.anet.meta.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                    this.anet.meta.directions.did2vecBr( :, idx2pbr ) * dvecSize + patchSide; ];
                for idx = 1 : numCont,
                    w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                    h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                    tlbrWarp = idx2tlbrWarp( :, idx );
                    tlbr = resizeTlbr( tlbrWarp, [ patchSide, patchSide ], [ h, w ] );
                    idx2tlbr( :, idx ) = tlbr - 1 + ...
                        [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                end;
                idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                rid2tlbrProp{ cid } = idx2tlbr;
            end;
            rid2tlbr = round( cat( 2, rid2tlbrProp{ : } ) );
            if isempty( rid2tlbr ), rid2tlbr = zeros( 5, 0 ); end;
            [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
            nid2cid = rid2tlbr( 5, : )';
            rid2tlbr = rid2tlbr_';
            if isempty( rid2tlbr ),
                rid2tlbr = zeros( 4, 0 );
                nid2rid = zeros( 0, 1 );
                nid2cid = zeros( 0, 1 );
            end;
        end
        function [ rid2tlbr, rid2out, rid2cid, fid2boxes ] = iid2det...
                ( this, iid, rid2tlbr0, nid2rid0, nid2cid0, detParams )
            numCls = numel( this.db.cid2name );
            numDimPerDirLyr = 4;
            numDimClsLyr = numCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numCls + numDimClsLyr;
            if isempty( rid2tlbr0 ),
                rid2tlbr = zeros( 4, 0 );
                rid2out = zeros( numOutDim, 0 );
                rid2cid = zeros( 0, 1 );
                return;
            end;
            % Pre-processing: box re-scaling.
            detType = detParams.type;
            rescaleBox = detParams.rescaleBox;
            rid2tlbr0 = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            rid2tlbr0 = round( rid2tlbr0 );
            rgbMean = reshape( this.anet.meta.normalization.averageImage, [ 1, 1, 3 ] );
            % Do detection on each region.
            flip = this.settingProp.flip;
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0( 1 : 4, : ) = bsxfun( @minus, rid2tlbr0( 1 : 4, : ), [ imTl; imTl; ] ) + 1;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], rgbMean );
            switch detType,
                case 'STATIC',
                    if nargout < 4
                        [ rid2tlbr, rid2out, rid2cid ] = this.staticFitting...
                            ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
                    else
                        [ rid2tlbr, rid2out, rid2cid, fid2boxes ] = this.staticFitting...
                            ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
                    end;
                case 'DYNAMIC',
                    if nargout < 4
                        [ rid2tlbr, rid2out, rid2cid ] = this.dynamicFitting...
                            ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
                    else
                        [ rid2tlbr, rid2out, rid2cid, fid2boxes ] = this.dynamicFitting...
                            ( rid2tlbr0, nid2rid0, nid2cid0, imGlobal, detParams );
                    end;
            end;
            if isempty( rid2tlbr ),
                rid2tlbr = zeros( 4, 0 );
                rid2out = zeros( numOutDim, 0 );
                rid2cid = zeros( 0, 1 );
                return;
            end;
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
            if nargout == 4,
                for fid = 1 : numel( fid2boxes ),
                    fid2boxes{ fid }( 1 : 4, : ) = bsxfun( @minus, fid2boxes{ fid }( 1 : 4, : ), 1 - [ imTl; imTl; ] );
                end;
            end;
        end
        function [ rid2out, rid2tlbr ] = initGuess( this, im )
            patchSide = this.anet.meta.map.patchSide;
            dilate = this.settingProp.dilate;
            maxSide = this.settingProp.normalizeImageMaxSide;
            maximumImageSize = this.settingProp.maximumImageSize;
            [ r, c, ~ ] = size( im );
            imSize0 = [ r; c; ];
            if maxSide, imSize = normalizeImageSize( maxSide, imSize0 ); else imSize = imSize0; end;
            sid2size = round( bsxfun( @times, this.scales, imSize ) );
            rid2tlbr = extractDenseRegions...
                ( imSize, sid2size, patchSide, this.anet.meta.map.stride, dilate, maximumImageSize );
            rid2tlbr = round( resizeTlbr( rid2tlbr, imSize, imSize0 ) );
            rid2out = this.extractDenseActivations( im, sid2size );
            if size( rid2out, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
        end
        function rid2out = extractDenseActivations...
                ( this, originalImage, targetImageSizes )
            patchSide = this.anet.meta.map.patchSide;
            regionDilate = this.settingProp.dilate;
            maximumImageSize = this.settingProp.maximumImageSize;
            imageDilate = round( patchSide * regionDilate );
            rgbMean = reshape( this.anet.meta.normalization.averageImage, [ 1, 1, 3 ] );
            interpolation = this.anet.meta.normalization.interpolation;
            numSize = size( targetImageSizes, 2 );
            rid2out = cell( numSize, 1 );
            for sid = 1 : numSize,
                imSize = targetImageSizes( :, sid );
                if min( imSize ) + 2 * imageDilate < patchSide, continue; end;
                if prod( imSize + imageDilate * 2 ) > maximumImageSize,
                    fprintf( '%s: Warning) Im of %s rejected.\n', ...
                        upper( mfilename ), mat2str( imSize ) ); continue;
                end;
                im = imresize( ...
                    originalImage, imSize', ...
                    'method', interpolation );
                im = single( im );
                roi = [ ...
                    1 - imageDilate; ...
                    1 - imageDilate; ...
                    imSize( : ) + imageDilate; ];
                im = normalizeAndCropImage( im, roi, rgbMean );
                fprintf( '%s: Feed im of %dX%d size.\n', ...
                    upper( mfilename ), size( im, 1 ), size( im, 2 ) );
                y = this.feedforward( im );
                [ nr, nc, z ] = size( y );
                y = reshape( permute( y, [ 3, 1, 2 ] ), z, nr * nc );
                rid2out{ sid } = y;
            end;
            rid2out = cat( 2, rid2out{ : } );
        end
        function y = feedforward( this, im )
            im = gpuArray( im );
            res = vl_simplenn( this.anet, im, [  ], [  ], ...
                'accumulate', false, ...
                'mode', 'test', ...
                'conserveMemory', true );
            y = res( end ).x; clear res im tmp;      
            y = gather( y );
        end
        function [ did2tlbr, did2out, did2cid, fid2boxes ] = dynamicFitting...
                ( this, rid2tlbr, nid2rid, nid2cid, im, detParams )
            % Preparing for data.
            inputSide = this.anet.meta.inputSize( 1 );
            numTopCls = detParams.numTopClassification;
            numTopDir = detParams.numTopDirection;
            dvecSize = detParams.directionVectorSize;
            testBatchSize = detParams.batchSize;
            numMaxFeed = 50;
            interpolation = 'bilinear';
            inputCh = size( im, 3 );
            numCls = numel( this.db.cid2name );
            numDimPerDirLyr = 4;
            numDimClsLyr = numCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numCls + numDimClsLyr;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            signStop = numDimPerDirLyr;
            signDiag = 2;
            numRegn = size( rid2tlbr, 2 );
            buffSize = 5000;
            if ~numRegn,
                did2tlbr = zeros( 4, 0, 'single' );
                did2out = zeros( numOutDim, 0, 'single' );
                did2cid = zeros( 0, 1, 'single' );
                return;
            end;
            % Detection on each region.
            did2tlbr = zeros( 4, buffSize, 'single' );
            did2out = zeros( numOutDim, buffSize, 'single' );
            did2cid = zeros( buffSize, 1, 'single' );
            did2fill = false( 1, buffSize );
            did = 1;
            if nargout == 4, fid2boxes = cell( numMaxFeed, 1 ); end;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                for r = 1 : testBatchSize : numRegn,
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2im = zeros( inputSide, inputSide, inputCh, bsize, 'single' );
                    for brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ inputSide, inputSide ], 'method', interpolation );
                    end;
                    brid2out = this.feedforward( brid2im );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                nrid2tlbr = cell( numCls, 1 );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2rank2cidx ] = sort( rid2outCls, 1, 'descend' );
                rid2cidx = rid2rank2cidx( 1, : );
                for cid = 1 : numCls,
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    rid2outTl = rid2out( dimTl, : );
                    rid2outBr = rid2out( dimBr, : );
                    [ ~, rid2rank2ptl ] = sort( rid2outTl, 1, 'descend' );
                    [ ~, rid2rank2pbr ] = sort( rid2outBr, 1, 'descend' );
                    rid2ptl = rid2rank2ptl( 1, : );
                    rid2pbr = rid2rank2pbr( 1, : );
                    rid2ss = rid2ptl == signStop & rid2pbr == signStop;
                    rid2okTl = any( rid2rank2ptl( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2okBr = any( rid2rank2pbr( 1 : numTopDir, : ) == signDiag, 1 );
                    rid2dd = rid2okTl & rid2okBr;
                    rid2dd = rid2dd & ( ~rid2ss );
                    rid2dd = rid2dd & ( rid2ptl == signDiag | rid2pbr == signDiag );
                    rid2bgd = rid2cidx == ( numCls + 1 );
                    rid2high = any( rid2rank2cidx( 1 : min( numTopCls, numDimClsLyr ), : ) == cid, 1 );
                    rid2high = rid2high & ( ~rid2bgd );
                    rid2top = rid2cidx == cid;
                    nid2purebred = nid2cid == cid;
                    rid2purebred = false( 1, numRegn );
                    rid2purebred( nid2rid( nid2purebred ) ) = true;
                    % Find and store detections.
                    rid2det = rid2ss & rid2purebred & rid2top;
                    numDet = sum( rid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = rid2tlbr( :, rid2det );
                    did2cid( dids ) = cid;
                    did2out( :, dids ) = rid2out( :, rid2det );
                    did2fill( dids ) = true;
                    did = did + numDet;
                    if nargout == 4,
                        fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, cat( 1, did2tlbr( :, did2fill ), ones( 1, sum( did2fill ) ) ) );
                    end;
                    % Find and store regiones to be continued.
                    rid2purebredCont = ( ~rid2det ) & rid2high & rid2purebred & ( ~rid2ss );
                    rid2branchCont = rid2high & rid2dd & ~rid2purebred;
                    rid2cont = rid2purebredCont | rid2branchCont;
                    numCont = sum( rid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = rid2tlbr( :, rid2cont );
                    idx2ptl = rid2ptl( rid2cont );
                    idx2pbr = rid2pbr( rid2cont );
                    idx2tlbrWarp = [ ...
                        this.anet.meta.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.anet.meta.directions.did2vecBr( :, idx2pbr ) * dvecSize + inputSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ inputSide, inputSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                    nrid2tlbr{ cid } = idx2tlbr;
                end;
                rid2tlbr = round( cat( 2, nrid2tlbr{ : } ) );
                if isempty( rid2tlbr ), break; end;
                [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
                nid2cid = rid2tlbr( 5, : )';
                rid2tlbr = rid2tlbr_';
                numRegn = size( rid2tlbr, 2 );
                if nargout == 4,
                    fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, cat( 1, rid2tlbr, zeros( 1, numRegn ) ) );
                end;
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2out = did2out( :, did2fill );
            did2cid = did2cid( did2fill );
            if nargout == 4, fid2boxes = fid2boxes( ~cellfun( @isempty, fid2boxes ) ); end;
        end
        function [ did2tlbr, did2out, did2cid, fid2boxes ] = staticFitting...
                ( this, rid2tlbr, nid2rid, nid2cid, im, detParams )
            % Preparing for data.
            inputSide = this.anet.meta.inputSize( 1 );
            onlyTarAndBgd = detParams.onlyTargetAndBackground;
            dvecSize = detParams.directionVectorSize;
            testBatchSize = detParams.batchSize;
            numMaxFeed = 50;
            interpolation = 'bilinear';
            inputCh = size( im, 3 );
            numCls = numel( this.db.cid2name );
            numDimPerDirLyr = 4;
            numDimClsLyr = numCls + 1;
            numOutDim = ( numDimPerDirLyr * 2 ) * numCls + numDimClsLyr;
            dimCls = numCls * numDimPerDirLyr * 2 + ( 1 : numDimClsLyr );
            signStop = numDimPerDirLyr;
            numRegn = size( rid2tlbr, 2 );
            buffSize = numel( nid2rid );
            if ~numRegn,
                did2tlbr = zeros( 4, 0, 'single' );
                did2out = zeros( numOutDim, 0, 'single' );
                did2cid = zeros( 0, 1, 'single' );
                return;
            end;
            % Detection on each region.
            did2tlbr = zeros( 4, buffSize, 'single' );
            did2out = zeros( numOutDim, buffSize, 'single' );
            did2cid = zeros( buffSize, 1, 'single' );
            did2fill = false( 1, buffSize );
            did = 1;
            if nargout == 4, fid2boxes = cell( numMaxFeed, 1 ); end;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                for r = 1 : testBatchSize : numRegn,
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2im = zeros( inputSide, inputSide, inputCh, bsize, 'single' );
                    for brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ inputSide, inputSide ], 'method', interpolation );
                    end;
                    brid2out = this.feedforward( brid2im );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                nrid2tlbr = cell( numCls, 1 );
                rid2outCls = rid2out( dimCls, : );
                [ ~, rid2cidx ] = max( rid2outCls, [  ], 1 );
                for cid = 1 : numCls,
                    rid2tar = false( size( nid2rid ) );
                    rid2tar( nid2rid( nid2cid == cid ) ) = true;
                    if ~sum( rid2tar ), continue; end;
                    crid2tlbr = rid2tlbr(  :, rid2tar );
                    crid2out = rid2out( :, rid2tar );
                    crid2outCls = rid2outCls( :, rid2tar );
                    crid2cidx = rid2cidx( rid2tar );
                    if onlyTarAndBgd,
                        crid2bgdScore = crid2outCls( numCls + 1, : );
                        crid2tarScore = crid2outCls( cid, : );
                        crid2fgd = crid2tarScore > crid2bgdScore;
                    else
                        crid2fgd = crid2cidx ~= ( numCls + 1 );
                    end;
                    dimTl = ( cid - 1 ) * numDimPerDirLyr * 2 + 1;
                    dimTl = dimTl : dimTl + numDimPerDirLyr - 1;
                    dimBr = dimTl + numDimPerDirLyr;
                    crid2outTl = crid2out( dimTl, : );
                    crid2outBr = crid2out( dimBr, : );
                    [ ~, crid2ptl ] = max( crid2outTl, [  ], 1 );
                    [ ~, crid2pbr ] = max( crid2outBr, [  ], 1 );
                    crid2ss = crid2ptl == signStop & crid2pbr == signStop;
                    % Find and store detections.
                    crid2det = crid2ss & crid2fgd; % Add more conditions!!!
                    numDet = sum( crid2det );
                    dids = did : did + numDet - 1;
                    did2tlbr( :, dids ) = crid2tlbr( :, crid2det );
                    did2out( :, dids ) = crid2out( :, crid2det );
                    did2cid( dids ) = cid;
                    did2fill( dids ) = true;
                    did = did + numDet;
                    if nargout == 4,
                        fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, ...
                            cat( 1, did2tlbr( :, did2fill ), ones( 1, sum( did2fill ) ) ) );
                    end;
                    % Find and store regiones to be continued.
                    crid2cont = ~crid2det & ~crid2ss;
                    numCont = sum( crid2cont );
                    if ~numCont, continue; end;
                    idx2tlbr = crid2tlbr( :, crid2cont );
                    idx2ptl = crid2ptl( crid2cont );
                    idx2pbr = crid2pbr( crid2cont );
                    idx2tlbrWarp = [ ...
                        this.anet.meta.directions.did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                        this.anet.meta.directions.did2vecBr( :, idx2pbr ) * dvecSize + inputSide; ];
                    for idx = 1 : numCont,
                        w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                        h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                        tlbrWarp = idx2tlbrWarp( :, idx );
                        tlbr = resizeTlbr( tlbrWarp, [ inputSide, inputSide ], [ h, w ] );
                        idx2tlbr( :, idx ) = tlbr - 1 + ...
                            [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                    end;
                    idx2tlbr = cat( 1, idx2tlbr, cid * ones( 1, numCont ) );
                    nrid2tlbr{ cid } = idx2tlbr;
                end;
                rid2tlbr = round( cat( 2, nrid2tlbr{ : } ) );
                if isempty( rid2tlbr ), break; end;
                [ rid2tlbr_, ~, nid2rid ] = unique( rid2tlbr( 1 : 4, : )', 'rows' );
                nid2cid = rid2tlbr( 5, : )';
                rid2tlbr = rid2tlbr_';
                numRegn = size( rid2tlbr, 2 );
                if nargout == 4,
                    fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, ...
                        cat( 1, rid2tlbr, zeros( 1, numRegn ) ) );
                end;
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2out = did2out( :, did2fill );
            did2cid = did2cid( did2fill );
            if nargout == 4, fid2boxes = fid2boxes( ~cellfun( @isempty, fid2boxes ) ); end;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functions for file identification %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 1. Scale factor file.
        function name = getScaleFactorName( this )
            numScaling = this.settingProp.numScaling;
            posGotoMargin = this.settingProp.posGotoMargin;
            maxSide = this.settingProp.normalizeImageMaxSide;
            piormt = 1 / ( posGotoMargin ^ 2 );
            piormt = num2str( piormt );
            piormt( piormt == '.' ) = 'P';
            name = sprintf( 'SFTE_N%03d_PIORMT%s_NIMS%d', ...
                numScaling, piormt, maxSide );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getScaleFactorDir( this )
            dir = this.db.getDir;
        end
        function dir = makeScaleFactorDir( this )
            dir = this.getScaleFactorDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getScaleFactorPath( this )
            fname = strcat( this.getScaleFactorName, '.mat' );
            path = fullfile( this.getScaleFactorDir, fname );
        end
        % 2. Proposal file.
        function name = getPropName( this )
            name = sprintf( 'PROP_%s_OF_%s', ...
                this.settingProp.changes, this.anet.meta.name );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getPropDir( this )
            name = this.getPropName;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'PROP_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makePropDir( this )
            dir = this.getPropDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPropPath( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getPropDir, fname );
        end
        % 3. Detection0 file.
        function name = getDet0Name( this )
            name = sprintf( 'DET0_%s_OF_%s', ...
                this.settingDet0.changes, this.getPropName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDet0Dir( this )
            name = this.getDet0Name;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'DET0_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makeDet0Dir( this )
            dir = this.getDet0Dir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDet0Path( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getDet0Dir, fname );
        end
        % 4. Detection1 file.
        function name = getDet1Name( this )
            name = sprintf( 'DET1_%s_OF_%s_OF_%s', ...
                this.settingDet1.changes, this.settingMrg0.changes, this.getDet0Name );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDet1Dir( this )
            name = this.getDet1Name;
            if length( name ) > 150,
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) );
                name = sprintf( '%010d', name );
                name = strcat( 'DET1_', name );
            end
            dir = fullfile( this.db.dstDir, name );
        end
        function dir = makeDet1Dir( this )
            dir = this.getDet1Dir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getDet1Path( this, iid )
            fname = sprintf( 'ID%06d.mat', iid );
            fpath = fullfile( this.getDet1Dir, fname );
        end
    end
    methods( Static )
        function [ rid2tlbr, rid2score, rid2cid ] = merge...
                ( rid2tlbr, rid2score, rid2cid, mrgParams )
            mergingOverlap = mrgParams.mergingOverlap;
            mergingType = mrgParams.mergingType;
            mergingMethod = mrgParams.mergingMethod;
            minNumSuppBox = mrgParams.minimumNumSupportBox;
            classWiseMerging = mrgParams.classWiseMerging;
            if mergingOverlap == 1, return; end;
            if classWiseMerging,
                cids = unique( rid2cid );
                numCls = numel( cids );
                rid2tlbr_ = cell( numCls, 1 );
                rid2score_ = cell( numCls, 1 );
                rid2cid_ = cell( numCls, 1 );
                for cidx = 1 : numCls,
                    cid = cids( cidx );
                    rid2ok = rid2cid == cid;
                    switch mergingType,
                        case 'NMS',
                            rid2score = rid2score( : )';
                            [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = nms( ...
                                [ rid2tlbr( :, rid2ok ); rid2score( rid2ok ); ]', ...
                                mergingOverlap, minNumSuppBox, mergingMethod );
                            rid2tlbr_{ cidx } = rid2tlbr_{ cidx }';
                        case 'NMSIOU',
                            rid2score = rid2score( : )';
                            [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = nms_iou( ...
                                [ rid2tlbr( :, rid2ok ); rid2score( rid2ok ); ]', ...
                                mergingOverlap, minNumSuppBox, mergingMethod );
                            rid2tlbr_{ cidx } = rid2tlbr_{ cidx }';
                        case 'OV',
                            [ rid2tlbr_{ cidx }, rid2score_{ cidx } ] = ov( ...
                                rid2tlbr( :, rid2ok ), rid2score( rid2ok ), ...
                                mergingOverlap, minNumSuppBox, mergingMethod );
                    end;
                    rid2cid_{ cidx } = cid * ones( size( rid2score_{ cidx } ) );
                end;
                rid2tlbr = cat( 2, rid2tlbr_{ : } );
                rid2score = cat( 1, rid2score_{ : } );
                rid2cid = cat( 1, rid2cid_{ : } );
            else
            end;
            [ rid2score, idx ] = sort( rid2score, 'descend' );
            rid2tlbr = rid2tlbr( :, idx );
            rid2cid = rid2cid( idx );
        end
    end
end
