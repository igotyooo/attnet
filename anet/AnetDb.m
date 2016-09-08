classdef AnetDb < handle
    properties
        db;
        scales;
        directions;
        setting;
        settingBatch;
    end
    methods( Access = public )
        function this = AnetDb( db, setting )
            this.db = db;
            this.setting.patchSide = 224;
            this.setting.stride = 32;
            this.setting.numScaling = 256;
            this.setting.dilate = 1 / 4;
            this.setting.normalizeImageMaxSide = 0;
            this.setting.maximumImageSize = 9e6;
            this.setting.posGotoMargin = 2;
            this.setting.numQuantizeBetweenStopAndGoto = 3;
            this.setting.negIntOverObjLessThan = 0.1;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this )
            % Define directions.
            fprintf( '%s: Define directions.\n', upper( mfilename ) );
            numDirection = 3;
            angstep = ( pi / 2 ) / ( numDirection - 1 );
            did2angTl = ( 0 : angstep : ( pi / 2 ) )';
            did2angBr = ( pi : angstep : ( pi * 3 / 2 ) )';
            this.directions.did2vecTl = [ [ cos( did2angTl' ); sin( did2angTl' ); ], [ 0; 0; ] ];
            this.directions.did2vecBr = [ [ cos( did2angBr' ); sin( did2angBr' ); ], [ 0; 0; ] ];
            numDirPerSide = 4;
            numSide = 2;
            numPair = numDirPerSide ^ numSide;
            dpid2dp = dec2base( ( 0 : numPair - 1 )', numDirPerSide, numSide )';
            dpid2dp = mod( double( dpid2dp ), double( '0' ) - 1 );
            this.directions.dpid2dp = dpid2dp;
            fprintf( '%s: Done.\n', upper( mfilename ) );
            % Determine scaling factors.
            fpath = this.getScaleFactorPath;
            try
                fprintf( '%s: Try to load scaling factors.\n', upper( mfilename ) );
                data = load( fpath );
                this.scales = data.data.scales;
            catch
                fprintf( '%s: Determine scaling factors.\n', ...
                    upper( mfilename ) );
                patchSide = this.setting.patchSide;
                posGotoMargin = this.setting.posGotoMargin;
                maxSide = this.setting.normalizeImageMaxSide;
                numScaling = this.setting.numScaling;
                posIntOverRegnMoreThan = 1 / ( posGotoMargin ^ 2 );
                setid = 1;
                tcid = 15;
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
                referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
                [ scalesRow, scalesCol ] = determineImageScaling...
                    ( oid2tlbr( :, oid2cid == tcid ), numScaling, referenceSide, true );
                data.scales = [ scalesRow, scalesCol ]';
                fprintf( '%s: Done.\n', upper( mfilename ) );
                save( fpath, 'data' );
                this.scales = data.scales;
            end;
            fprintf( '%s: Done.\n', upper( mfilename ) );
        end
        function anetdb = makeAnetDb( this )
            % Reform the general db to a task-specific format.
            fpath = this.getAnetDbPath;
            try
                fprintf( '%s: Try to load anet-db.\n', upper( mfilename ) );
                data = load( fpath );
                anetdb = data.data.anetdb;
                fprintf( '%s: Done.\n', upper( mfilename ) );
            catch
                fprintf( '%s: Gen anet-db.\n', upper( mfilename ) );
                data.anetdb.tr = this.makeSubAnetDb( 1 );
                data.anetdb.val = this.makeSubAnetDb( 2 );
                anetdb = data.anetdb;
                fprintf( '%s: Save anet-db.\n', upper( mfilename ) );
                this.makeAnetDbDir;
                save( fpath, 'data', '-v7.3' );
                fprintf( '%s: Done.\n', upper( mfilename ) );
            end;
            anetdb.directions = this.directions;
            anetdb.name = this.getAnetDbName;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%
    % Private interface. %
    %%%%%%%%%%%%%%%%%%%%%%
    methods( Access = private )
        function subAnetDb = makeSubAnetDb( this, setid )
            % Set parameters.
            patchSide = this.setting.patchSide;
            stride = this.setting.stride;
            numSize = size( this.scales, 2 );
            numDirPair = size( this.directions.dpid2dp, 2 );
            dilate = this.setting.dilate;
            numDirPerSide = 4;
            dpidBasis = numDirPerSide .^ ( 1 : -1 : 0 )';
            domainWarp = [ patchSide; patchSide; ];
            % Parameters for positive mining.
            posGotoMargin = this.setting.posGotoMargin;
            maxSide = this.setting.normalizeImageMaxSide;
            numQtz = this.setting.numQuantizeBetweenStopAndGoto;
            step = sqrt( ( posGotoMargin ) ^ ( 2 / ( numQtz + 1 ) ) );
            numMaxRegionPerDirectionPair = 16;
            % Parameters for negative mining.
            negIntOverObjLessThan = this.setting.negIntOverObjLessThan;
            % Do the job.
            newiid2iid = find( this.db.iid2setid == setid );
            newiid2iid = newiid2iid( randperm( numel( newiid2iid ) )' );
            newoid2oid = cat( 1, this.db.iid2oids{ newiid2iid } );
            newoid2newiid = zeros( size( newoid2oid ), 'single' );
            newoid2dpid2posregns = cell( size( newoid2oid ) );
            newoid2dpid2posregnsFlip = cell( size( newoid2oid ) );
            newoid2cid = zeros( size( newoid2oid ), 'single' );
            newiid2sid2negregns = cell( size( newiid2iid ) );
            newiid2impath = this.db.iid2impath( newiid2iid );
            numIm = numel( newiid2iid );
            newoid = 1; cummt = 0;
            for newiid = 1 : numIm;
                itime = tic;
                iid = newiid2iid( newiid );
                oid2tlbr = this.db.oid2bbox( :, this.db.iid2oids{ iid } );
                oid2cid = this.db.oid2cid( this.db.iid2oids{ iid } );
                numObj = size( oid2tlbr, 2 );
                oid2dpid2posregns = cell( numObj, 1 );
                oid2dpid2posregnsFlip = cell( numObj, 1 );
                % Positive mining.
                for oid = 1 : numObj,
                    tlbr = oid2tlbr( :, oid );
                    tlbrGo = zeros( 4, numQtz + 1, 'single' );
                    for g = 1 : numQtz + 1,
                        s = step ^ g;
                        tlbrGo( :, g ) = scaleBoxes( tlbr, s, s );
                    end;
                    tlbrs = [ tlbr, tlbrGo ];
                    n = size( tlbrs, 2 );
                    rid2tlbr = zeros( 4, n ^ 4, 'single' );
                    rid2dp = zeros( 2, n ^ 4, 'single' );
                    rid2dpFlip = zeros( 2, n ^ 4, 'single' );
                    cnt = 0;
                    for it = 1 : n,
                        t = tlbrs( 1, it );
                        for il = 1 : n,
                            l = tlbrs( 2, il );
                            for ib = 1 : n,
                                b = tlbrs( 3, ib );
                                for ir = 1 : n,
                                    r = tlbrs( 4, ir );
                                    cnt = cnt + 1;
                                    regn = [ t; l; b; r; ];
                                    [ didTl, didBr, didTlFlip, didBrFlip, ~ ] = ...
                                        getGtCornerDirection( regn, tlbr, ...
                                        this.directions.did2vecTl, ...
                                        this.directions.did2vecBr, ...
                                        0, domainWarp );
                                    rid2tlbr( :, cnt ) = regn;
                                    rid2dp( :, cnt ) = [ didTl; didBr; ];
                                    rid2dpFlip( :, cnt ) = [ didTlFlip; didBrFlip; ];
                                end;
                            end;
                        end;
                    end;
                    rid2dpid = sum( bsxfun( @times, dpidBasis, rid2dp - 1 ), 1 ) + 1;
                    rid2dpidFlip = sum( bsxfun( @times, dpidBasis, rid2dpFlip - 1 ), 1 ) + 1;
                    rid2tlbr = round( rid2tlbr );
                    oid2dpid2posregns{ oid } = cell( numDirPair, 1 );
                    oid2dpid2posregnsFlip{ oid } = cell( numDirPair, 1 );
                    for dpid = 1 : numDirPair,
                        regns = rid2tlbr( :, rid2dpid == dpid );
                        numRegn = size( regns, 2 );
                        ok = randperm( numRegn, min( numRegn, numMaxRegionPerDirectionPair ) );
                        oid2dpid2posregns{ oid }{ dpid } = regns( :, ok );
                        regnsFlip = rid2tlbr( :, rid2dpidFlip == dpid );
                        numRegnFlip = size( regnsFlip, 2 );
                        okFlip = randperm( numRegnFlip, min( numRegnFlip, numMaxRegionPerDirectionPair ) );
                        oid2dpid2posregnsFlip{ oid }{ dpid } = regnsFlip( :, okFlip );
                    end;
                end;
                % Negative mining.
                imSize0 = this.db.iid2size( :, iid );
                if maxSide, imSize = normalizeImageSize( maxSide, imSize0 ); else imSize = imSize0; end;
                sid2size = round( bsxfun( @times, this.scales, imSize ) );
                maximumImageSize = this.setting.maximumImageSize;
                rid2tlbr = ...
                    extractDenseRegions( ...
                    imSize, ...
                    sid2size, ...
                    patchSide, ...
                    stride, ...
                    dilate, ...
                    maximumImageSize );
                if isempty( rid2tlbr ),
                    rid2tlbr = zeros( 5, 0 );
                    rid2rect = zeros( 5, 0 );
                else
                    rid2tlbr = round( resizeTlbr( rid2tlbr, imSize, imSize0 ) );
                    rid2rect = tlbr2rect( rid2tlbr );
                end;
                oid2rect = tlbr2rect( oid2tlbr );
                oid2area = prod( oid2rect( 3 : 4, : ), 1 )';
                rid2oid2int = rectint( rid2rect', oid2rect' );
                rid2oid2ioo = bsxfun( @times, rid2oid2int, 1 ./ oid2area' );
                numMaxRegnPerSize = max( 1, round( numMaxRegionPerDirectionPair * numDirPair / numSize ) );
                rid2ok = all( rid2oid2ioo <= negIntOverObjLessThan, 2 );
                nrid2tlbr = rid2tlbr( :, rid2ok );
                sid2nregns = cell( numSize, 1 );
                for sid = 1 : numSize,
                    nrids = find( nrid2tlbr( 5, : ) == sid );
                    if ~isempty( nrids ),
                        nrids = randsample( nrids, min( numel( nrids ), numMaxRegnPerSize ) );
                        sid2nregns{ sid } = nrid2tlbr( 1 : 4, nrids );
                    end;
                end;
                % Accumulate results.
                newoids = ( newoid : newoid + numObj - 1 )';
                newoid2newiid( newoids ) = newiid;
                newoid2cid( newoids ) = oid2cid;
                newoid2dpid2posregns( newoids ) = oid2dpid2posregns;
                newoid2dpid2posregnsFlip( newoids ) = oid2dpid2posregnsFlip;
                newiid2sid2negregns{ newiid } = sid2nregns;
                newoid = newoid + numObj;
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, newiid, ...
                    sprintf( 'make regions on im %d', newiid ), cummt );
            end;
            subAnetDb.iid2impath = newiid2impath;
            subAnetDb.oid2iid = newoid2newiid;
            subAnetDb.oid2cid = newoid2cid;
            subAnetDb.oid2dpid2posregns = newoid2dpid2posregns;
            subAnetDb.oid2dpid2posregnsFlip = newoid2dpid2posregnsFlip;
            subAnetDb.iid2sid2negregns = newiid2sid2negregns;
        end
        % Functions for file IO.
        function name = getScaleFactorName( this )
            numScaling = this.setting.numScaling;
            posGotoMargin = this.setting.posGotoMargin;
            maxSide = this.setting.normalizeImageMaxSide;
            piormt = 1 / ( posGotoMargin ^ 2 );
            piormt = num2str( piormt );
            piormt( piormt == '.' ) = 'P';
            name = sprintf( 'SFTR_N%03d_PIORMT%s_NIMS%d', ...
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
        function name = getAnetDbName( this )
            name = sprintf( 'ADB_%s', ...
                this.setting.changes );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getAnetDbDir( this )
            dir = this.db.getDir;
        end
        function dir = makeAnetDbDir( this )
            dir = this.getAnetDbDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function path = getAnetDbPath( this )
            fname = strcat( this.getAnetDbName, '.mat' );
            path = fullfile( this.getAnetDbDir, fname );
        end
    end
end