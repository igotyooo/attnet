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
            this.settingProp.directionVectorSize      = 30;
            this.settingDet0.batchSize                = settingDet0.batchSize;
            this.settingDet0.rescaleBox               = 1;
            this.settingDet0.rescaleBoxStd            = 0;
            this.settingDet0.directionVectorSize      = 30;
            this.settingMrg0.mergingOverlap           = 0.8;
            this.settingMrg0.mergingType              = 'OV';
            this.settingMrg0.mergingMethod            = 'WAVG';
            this.settingMrg0.minimumNumSupportBox     = 1;          % Ignored if mergingOverlap = 1.
            this.settingDet1.batchSize                = settingDet1.batchSize;
            this.settingDet1.rescaleBox               = 2.5;
            this.settingDet1.rescaleBoxStd            = 0;
            this.settingDet1.directionVectorSize      = 30;
            this.settingMrg1.mergingOverlap           = 0.6;
            this.settingMrg1.mergingType              = 'OV';
            this.settingMrg1.mergingMethod            = 'WAVG';
            this.settingMrg1.minimumNumSupportBox     = 0;          % Ignored if mergingOverlap = 1.
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
                oid2cid = this.db.oid2cid( this.db.iid2setid( this.db.oid2iid ) == setid );
                if maxSide,
                    oid2iid = this.db.oid2iid( this.db.iid2setid( this.db.oid2iid ) == setid );
                    oid2imsize = this.db.iid2size( :, oid2iid );
                    numRegn = size( oid2tlbr, 2 );
                    for oid = 1 : numRegn,
                        [ ~, oid2tlbr( :, oid ) ] = normalizeImageSize...
                            ( maxSide, oid2imsize( :, oid ), oid2tlbr( :, oid ) );
                    end;
                end;
                tcid = 15;
                referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr( :, oid2cid == tcid ), numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Fetch net on GPU.
            if this.settingProp.gpu > 0,
                fprintf( '%s: Fetch anet on GPU.\n', upper( mfilename ) );
                gpuDevice( this.settingProp.gpu );
                this.anet = vl_simplenn_move( this.anet, 'gpu') ;
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end;
        end
        function rid2tlbr = iid2prop( this, iid )
            fpath = this.getPropPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
            catch
                rid2tlbr = this.iid2propWrapper( iid );
                this.makePropDir;
                save( fpath, 'rid2tlbr' );
            end;
        end
        function [ rid2tlbr, rid2score ] = iid2det0( this, iid )
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
            catch
                % 1. Get regions.
                rid2tlbr = this.iid2prop( iid );
                % 2. Tighten regions.
                [ rid2tlbr, rid2out ] = this.iid2det...
                    ( iid, rid2tlbr, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2out' );
            end;
            if isempty( rid2tlbr ), rid2score = zeros( 0, 1 ); end;
            if nargout && ~isempty( rid2tlbr ),
                % 3. Merge regions.
                rid2score = this.scoring( rid2out );
                [ rid2tlbr, rid2score ] = this.merge...
                    ( rid2tlbr, rid2score, this.settingMrg0 );
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
        function [ rid2tlbr, rid2score ] = iid2det1( this, iid )
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
            catch
                % 1. Get regions.
                rid2tlbr = this.iid2det0( iid );
                % 2. Tighten regions.
                [ rid2tlbr, rid2out ] = this.iid2det...
                    ( iid, rid2tlbr, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2out' );
            end;
            if isempty( rid2tlbr ), rid2score = zeros( 0, 1 ); end;
            if nargout && ~isempty( rid2tlbr ),
                % 3. Merge regions.
                rid2score = this.scoring( rid2out );
                [ rid2tlbr, rid2score ] = this.merge...
                    ( rid2tlbr, rid2score, this.settingMrg1 );
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
        function rid2score = scoring( this, rid2out )
            signStop = 4;
            numDimPerLyr = 5;
            dimTl = 1 : numDimPerLyr;
            dimBr = dimTl + numDimPerLyr;
            outsTl = rid2out( dimTl, : );
            outsBr = rid2out( dimBr, : );
            scoresTl = outsTl( signStop, : );
            scoresBr = outsBr( signStop, : );
            rid2score = ( 2 * ( scoresTl + scoresBr ) - sum( outsTl + outsBr, 1 ) )' / 2;
        end;
        function demoDet( this, iid )
            flip = this.settingProp.flip;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            % Demo 1: proposals.
            rid2tlbr = this.iid2prop( iid );
            figure( 1 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Proposals, IID%06d (Boxes are bounded)', iid ) ); hold off; drawnow;
            % Demo 2: detection0.
            fpath = this.getDet0Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
            catch
                [ rid2tlbr, rid2out ] = this.iid2det...
                    ( iid, rid2tlbr, this.settingDet0 );
                this.makeDet0Dir;
                save( fpath, 'rid2tlbr', 'rid2out' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure( 2 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection0, IID%06d', iid ) ); hold off; drawnow;
            % Demo 3: Merge0.
            rid2score = this.scoring( rid2out );
            rid2tlbr = this.merge...
                ( rid2tlbr, rid2score, this.settingMrg0 );
            imbnd = [ 1; 1; this.db.iid2size( :, iid ); ];
            rid2tlbr = bndtlbr( rid2tlbr, imbnd );
            rid2tlbr = round( rid2tlbr );
            figure( 3 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, 'c' );
            title( sprintf( 'Merge0, IID%06d', iid ) ); hold off; drawnow;
            % Demo 4: detection1.
            fpath = this.getDet1Path( iid );
            try
                data = load( fpath );
                rid2tlbr = data.rid2tlbr;
                rid2out = data.rid2out;
            catch
                [ rid2tlbr, rid2out ] = this.iid2det...
                    ( iid, rid2tlbr, this.settingDet1 );
                this.makeDet1Dir;
                save( fpath, 'rid2tlbr', 'rid2out' );
            end;
            rid2tlbr = round( rid2tlbr );
            figure( 4 ); set( gcf, 'color', 'w' );
            plottlbr( rid2tlbr, im, false, { 'r'; 'g'; 'b'; 'y' } );
            title( sprintf( 'Detection1, IID%06d', iid ) ); hold off; drawnow;
            % Demo 5: Merge1.
            rid2score = this.scoring( rid2out );
            [ rid2tlbr, rid2score ] = this.merge...
                ( rid2tlbr, rid2score, this.settingMrg1 );
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
            figure( 5 ); set( gcf, 'color', 'w' );
            rid2title = cell( size( rid2score ) );
            for rid = 1 : numel( rid2score ),
                score = rid2score( rid );
                rid2title{ rid } = sprintf( '%.1f',  score );
            end;
            plottlbr( rid2tlbr, im, false, 'c', rid2title );
            title( sprintf( 'Merge1, IID%06d', iid ) ); hold off; drawnow;
        end
        function subDbDet0( this, numDiv, divId )
            iids = find( this.db.iid2setid == 3 );
            if isempty( iids ), iids = find( this.db.iid2setid == 2 ); end;
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
            iids = find( this.db.iid2setid == 3 );
            if isempty( iids ), iids = find( this.db.iid2setid == 2 ); end;
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
        function res = getSubDbDet0( this, numDiv, divId )
            iids = find( this.db.iid2setid == 3 );
            if isempty( iids ), iids = find( this.db.iid2setid == 2 ); end;
            idx2iid = iids( divId : numDiv : numel( iids ) );
            numIm = numel( idx2iid );
            rid2tlbr = cell( numIm, 1 );
            rid2score = cell( numIm, 1 );
            rid2iid = cell( numIm, 1 );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = idx2iid( iidx );
                [ rid2tlbr{ iidx }, rid2score{ iidx } ] = this.iid2det0( iid );
                rid2iid{ iidx } = iid * ones( size( rid2score{ iidx } ) );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Get det0 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
            rid2tlbr = cat( 2, rid2tlbr{ : } );
            rid2score = cat( 1, rid2score{ : } );
            rid2iid = cat( 1, rid2iid{ : } );
            res.did2tlbr = rid2tlbr;
            res.did2score = rid2score;
            res.did2iid = rid2iid;
        end;
        function res = getSubDbDet1( this, numDiv, divId )
            iids = find( this.db.iid2setid == 3 );
            if isempty( iids ), iids = find( this.db.iid2setid == 2 ); end;
            idx2iid = iids( divId : numDiv : numel( iids ) );
            numIm = numel( idx2iid );
            rid2tlbr = cell( numIm, 1 );
            rid2score = cell( numIm, 1 );
            rid2iid = cell( numIm, 1 );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = idx2iid( iidx );
                [ rid2tlbr{ iidx }, rid2score{ iidx } ] = this.iid2det1( iid );
                rid2iid{ iidx } = iid * ones( size( rid2score{ iidx } ) );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, sprintf( 'Get det1 on IID%d in %dth(/%d) div.', iid, divId, numDiv ), cummt );
            end;
            rid2tlbr = cat( 2, rid2tlbr{ : } );
            rid2score = cat( 1, rid2score{ : } );
            rid2iid = cat( 1, rid2iid{ : } );
            res.did2tlbr = rid2tlbr;
            res.did2score = rid2score;
            res.did2iid = rid2iid;
        end;
    end
    methods( Access = private )
        function rid2tlbr = iid2propWrapper( this, iid )
            % Initial guess.
            flip = this.settingProp.flip;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            [ rid2out, rid2tlbr ] = this.initGuess( im );
            % Compute each region score.
            patchSide = this.anet.meta.map.patchSide;
            dvecSize = this.settingProp.directionVectorSize;
            signDiag = 2;
            numDimPerLyr = 5;
            % Direction: DD condition.
            dimTl = 1 : numDimPerLyr;
            dimBr = dimTl + numDimPerLyr;
            rid2outTl = rid2out( dimTl, : );
            rid2outBr = rid2out( dimBr, : );
            [ ~, rid2ptl ] = max( rid2outTl, [  ], 1 );
            [ ~, rid2pbr ] = max( rid2outBr, [  ], 1 );
            rid2cont = ( rid2ptl == signDiag ) & ( rid2pbr == signDiag );
            % Update.
            numCont = sum( rid2cont );
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
            rid2tlbr = round( idx2tlbr );
            if isempty( rid2tlbr ), rid2tlbr = zeros( 4, 0 ); return; end;
            rid2tlbr_ = unique( rid2tlbr', 'rows' );
            rid2tlbr = rid2tlbr_';
        end
        function [ rid2tlbr, rid2out, fid2boxes ] = iid2det...
                ( this, iid, rid2tlbr0, detParams )
            numDimPerLyr = 5;
            numOutDim = ( numDimPerLyr * 2 );
            if isempty( rid2tlbr0 ),
                rid2tlbr = zeros( 4, 0 );
                rid2out = zeros( numOutDim, 0 );
                return;
            end;
            % Pre-processing: box re-scaling.
            rescaleBox = detParams.rescaleBox;
            rescaleBoxStd = detParams.rescaleBoxStd;
            rid2tlbr0_ = scaleBoxes( rid2tlbr0, sqrt( rescaleBox ), sqrt( rescaleBox ) );
            if rescaleBoxStd,
                sigh = ( rid2tlbr0_( 3, : ) - rid2tlbr0_( 1, : ) ) * ( rescaleBox - 1 ) / rescaleBox / 3;
                sigw = ( rid2tlbr0_( 4, : ) - rid2tlbr0_( 2, : ) ) * ( rescaleBox - 1 ) / rescaleBox / 3;
                noise = randn( size( rid2tlbr0_ ) ) .* [ sigh; sigw; sigh; sigw ] * rescaleBoxStd;
                rid2tlbr0_ = rid2tlbr0_ + noise;
                rid2tlbr0_( 1, : ) = min( [ rid2tlbr0_( 1, : ); rid2tlbr0( 1, : ) ], [  ], 1 );
                rid2tlbr0_( 2, : ) = min( [ rid2tlbr0_( 2, : ); rid2tlbr0( 2, : ) ], [  ], 1 );
                rid2tlbr0_( 3, : ) = max( [ rid2tlbr0_( 3, : ); rid2tlbr0( 3, : ) ], [  ], 1 );
                rid2tlbr0_( 4, : ) = max( [ rid2tlbr0_( 4, : ); rid2tlbr0( 4, : ) ], [  ], 1 );
            end;
            rid2tlbr0 = round( rid2tlbr0_ );
            rgbMean = reshape( this.anet.meta.normalization.averageImage, [ 1, 1, 3 ] );
            % Do detection on each region.
            flip = this.settingProp.flip;
            imTl = min( rid2tlbr0( 1 : 2, : ), [  ], 2 );
            imBr = max( rid2tlbr0( 3 : 4, : ), [  ], 2 );
            rid2tlbr0 = bsxfun( @minus, rid2tlbr0, [ imTl; imTl; ] ) + 1;
            im = imread( this.db.iid2impath{ iid } );
            if flip, im = fliplr( im ); end;
            imGlobal = normalizeAndCropImage...
                ( single( im ), [ imTl; imBr ], rgbMean );
            if nargout < 3
                [ rid2tlbr, rid2out ] = this.staticFitting...
                    ( rid2tlbr0, imGlobal, detParams );
            else
                [ rid2tlbr, rid2out, fid2boxes ] = this.staticFitting...
                    ( rid2tlbr0, imGlobal, detParams );
            end;
            if isempty( rid2tlbr ),
                rid2tlbr = zeros( 4, 0 );
                rid2out = zeros( numOutDim, 0 );
                return;
            end;
            % Convert to original image domain.
            rid2tlbr = bsxfun( @minus, rid2tlbr, 1 - [ imTl; imTl; ] );
            if nargout == 3,
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
            interpolation = 'bilinear';
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
        function [ did2tlbr, did2out, fid2boxes ] = staticFitting...
                ( this, rid2tlbr, im, detParams )
            % Preparing for data.
            did2vecTl = [ this.anet.meta.directions.did2vecTl, [ 0; 0; ] ];
            did2vecBr = [ this.anet.meta.directions.did2vecBr, [ 0; 0; ] ];
            inputSide = this.anet.meta.inputSize( 1 );
            dvecSize = detParams.directionVectorSize;
            testBatchSize = detParams.batchSize;
            numMaxFeed = 200; 
            interpolation = 'bilinear';
            inputCh = size( im, 3 );
            numDimPerLyr = 5;
            numOutDim = numDimPerLyr * 2;
            signStop = 4;
            signBgd = 5;
            numRegn = size( rid2tlbr, 2 );
            if ~numRegn,
                did2tlbr = zeros( 4, 0, 'single' );
                did2out = zeros( numOutDim, 0, 'single' );
                return;
            end;
            % Detection on each region.
            rid2flip = rand( numRegn, 1 ) > 0.5;
            did2tlbr = zeros( 4, numRegn, 'single' );
            did2out = zeros( numOutDim, numRegn, 'single' );
            did2fill = false( 1, numRegn );
            did = 1;
            if nargout == 3, fid2boxes = cell( numMaxFeed, 1 ); end;
            for feed = 1 : numMaxFeed,
                % Feedforward.
                fprintf( '%s: %dth feed. %d regions.\n', ...
                    upper( mfilename ), feed, numRegn );
                rid2out = zeros( numOutDim, numRegn, 'single' );
                for r = 1 : testBatchSize : numRegn,
                    rids = r : min( r + testBatchSize - 1, numRegn );
                    bsize = numel( rids );
                    brid2tlbr = rid2tlbr( :, rids );
                    brid2flip = rid2flip( rids );
                    brid2im = zeros( inputSide, inputSide, inputCh, bsize, 'single' );
                    for brid = 1 : bsize,
                        roi = brid2tlbr( :, brid );
                        imRegn = im( roi( 1 ) : roi( 3 ), roi( 2 ) : roi( 4 ), : );
                        if brid2flip( brid ), imRegn = fliplr( imRegn ); end;
                        brid2im( :, :, :, brid ) = imresize...
                            ( imRegn, [ inputSide, inputSide ], 'method', interpolation );
                    end;
                    brid2out = this.feedforward( brid2im );
                    brid2out = permute( brid2out, [ 3, 4, 1, 2 ] );
                    rid2out( :, rids ) = brid2out;
                end;
                % Do the job.
                dimTl = 1 : numDimPerLyr;
                dimBr = dimTl + numDimPerLyr;
                rid2outTl = rid2out( dimTl, : );
                rid2outBr = rid2out( dimBr, : );
                [ ~, rid2ptl ] = max( rid2outTl, [  ], 1 );
                [ ~, rid2pbr ] = max( rid2outBr, [  ], 1 );
                rid2det = rid2ptl == signStop & rid2pbr == signStop;
                rid2false = rid2ptl == signBgd & rid2pbr == signBgd;
                % Find and store detections.
                numDet = sum( rid2det );
                dids = did : did + numDet - 1;
                did2tlbr( :, dids ) = rid2tlbr( :, rid2det );
                did2out( :, dids ) = rid2out( :, rid2det );
                did2fill( dids ) = true;
                did = did + numDet;
                if nargout == 3,
                    fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, ...
                        cat( 1, did2tlbr( :, did2fill ), ones( 1, sum( did2fill ) ) ) );
                end;
                % Find and store regiones to be continued.
                rid2cont = ( ~rid2det ) & ( ~rid2false );
                numCont = sum( rid2cont );
                idx2tlbr = rid2tlbr( :, rid2cont );
                idx2flip = rid2flip( rid2cont );
                idx2ptl = rid2ptl( rid2cont );
                idx2pbr = rid2pbr( rid2cont );
                idx2tlbrWarp = [ ...
                    did2vecTl( :, idx2ptl ) * dvecSize + 1; ...
                    did2vecBr( :, idx2pbr ) * dvecSize + inputSide; ];
                for idx = 1 : numCont,
                    w = idx2tlbr( 4, idx ) - idx2tlbr( 2, idx ) + 1;
                    h = idx2tlbr( 3, idx ) - idx2tlbr( 1, idx ) + 1;
                    tlbrWarp = idx2tlbrWarp( :, idx );
                    if idx2flip( idx ), tlbrWarp = flipTlbr( tlbrWarp, inputSide ); end;
                    tlbr = resizeTlbr( tlbrWarp, [ inputSide, inputSide ], [ h, w ] );
                    idx2tlbr( :, idx ) = tlbr - 1 + ...
                        [ idx2tlbr( 1 : 2, idx ); idx2tlbr( 1 : 2, idx ) ];
                end;
                rid2tlbr = round( idx2tlbr );
                if isempty( rid2tlbr ), break; end;
                rid2tlbr_ = unique( rid2tlbr', 'rows' );
                rid2tlbr = rid2tlbr_';
                numRegn = size( rid2tlbr, 2 );
                rid2flip = rand( numRegn, 1 ) > 0.5;
                if nargout == 3,
                    fid2boxes{ feed } = cat( 2, fid2boxes{ feed }, ...
                        cat( 1, rid2tlbr, zeros( 1, numRegn ) ) );
                end;
            end;
            did2tlbr = did2tlbr( :, did2fill );
            did2out = did2out( :, did2fill );
            if nargout == 3, fid2boxes = fid2boxes( ~cellfun( @isempty, fid2boxes ) ); end;
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
        function [ rid2tlbr, rid2score ] = merge...
                ( rid2tlbr, rid2score, mrgParams )
            mergingOverlap = mrgParams.mergingOverlap;
            mergingType = mrgParams.mergingType;
            mergingMethod = mrgParams.mergingMethod;
            minNumSuppBox = mrgParams.minimumNumSupportBox;
            if mergingOverlap == 1, return; end;
            switch mergingType,
                case 'NMS',
                    rid2score = rid2score( : )';
                    [ rid2tlbr, rid2score ] = nms( ...
                        [ rid2tlbr; rid2score; ]', ...
                        mergingOverlap, minNumSuppBox, mergingMethod );
                    rid2tlbr = rid2tlbr';
                case 'NMSIOU',
                    rid2score = rid2score( : )';
                    [ rid2tlbr, rid2score ] = nms_iou( ...
                        [ rid2tlbr; rid2score; ]', ...
                        mergingOverlap, minNumSuppBox, mergingMethod );
                    rid2tlbr = rid2tlbr';
                case 'OV',
                    [ rid2tlbr, rid2score ] = ov( ...
                        rid2tlbr, rid2score, ...
                        mergingOverlap, minNumSuppBox, mergingMethod );
            end;
            [ rid2score, idx ] = sort( rid2score, 'descend' );
            rid2tlbr = rid2tlbr( :, idx );
        end
    end
end
