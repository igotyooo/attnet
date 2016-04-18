function res = evalVoc( det, db, setid )
% Set parameters
minoverlap = 0.5;
minDetArea = 0;
% Prepare data.
iidx2iid = find( db.iid2setid == setid );
numTeIm = numel( iidx2iid );
% Do the job.
res.cid2prec = cell( size( db.cid2name ) );
res.cid2rec = cell( size( db.cid2name ) );
res.cid2ap = zeros( size( db.cid2name ) );
res.cid2rank2tp = cell( size( db.cid2name ) );
res.cid2rank2fp = cell( size( db.cid2name ) );
res.cid2rank2iid = cell( size( db.cid2name ) );
res.cid2rank2tlbr = cell( size( db.cid2name ) );
res.cid2rank2score = cell( size( db.cid2name ) );
for cid = 1 : numel( db.cid2name ),
    oid2target = db.oid2cid == cid;
    % Get ground-truth.
    iidx2oidx2bbox = cell( numTeIm, 1 );
    iidx2oidx2diff = cell( numTeIm, 1 );
    iidx2oidx2det = cell( numTeIm, 1 );
    numPos = 0;
    for iidx = 1 : numTeIm;
        iid = iidx2iid( iidx );
        oidx2oid = ( db.oid2iid == iid ) & oid2target;
        iidx2oidx2bbox{ iidx } = db.oid2bbox( :, oidx2oid );
        iidx2oidx2diff{ iidx } = db.oid2diff( oidx2oid );
        iidx2oidx2det{ iidx } = false( sum( oidx2oid ), 1 );
        numPos = numPos + sum( ~iidx2oidx2diff{ iidx } );
    end;
    % Get detections.
    cdid2tlbr = det.did2tlbr( :, det.did2cid == cid );
    cdid2score = det.did2score( det.did2cid == cid );
    cdid2iid = det.did2iid( det.did2cid == cid );
    % Cut detection results by minimum area.
    did2rect = tlbr2rect( cdid2tlbr );
    did2area = prod( did2rect( 3 : 4, : ), 1 );
    did2ok = did2area >= minDetArea;
    did2score_ = cdid2score( did2ok );
    did2iid_ = cdid2iid( did2ok );
    did2tlbr_ = cdid2tlbr( :, did2ok );
    % Sort detection results.
    [ rank2score, rank2did ] = sort( did2score_, 'descend' );
    rank2iid = did2iid_( rank2did );
    rank2bbox = did2tlbr_( :, rank2did );
    % Determine TP/FP/DONT-CARE.
    numDet = numel( rank2did );
    rank2tp = zeros( numDet, 1 );
    rank2fp = zeros( numDet, 1 );
    for r = 1 : numDet,
        iid = rank2iid( r );
        iidx = find( iidx2iid == iid );
        detBbox = rank2bbox( :, r );
        ovMax = -Inf;
        for oidx = 1 : size( iidx2oidx2bbox{ iidx }, 2 ),
            gtBbox = iidx2oidx2bbox{ iidx }( :, oidx );
            insectBbox = [  ...
                max( detBbox( 1 ), gtBbox( 1 ) ); ...
                max( detBbox( 2 ), gtBbox( 2 ) ); ...
                min( detBbox( 3 ), gtBbox( 3 ) ); ...
                min( detBbox( 4 ), gtBbox( 4 ) ); ];
            insectW = insectBbox( 3 ) - insectBbox( 1 ) + 1;
            insectH = insectBbox( 4 ) - insectBbox( 2 ) + 1;
            if insectW > 0 && insectH > 0,
                union = ...
                    ( detBbox( 3 ) - detBbox( 1 ) + 1 ) * ...
                    ( detBbox( 4 ) - detBbox( 2 ) + 1 ) + ...
                    ( gtBbox( 3 ) - gtBbox( 1 ) + 1 ) * ...
                    ( gtBbox( 4 ) - gtBbox( 2 ) + 1 ) - ...
                    insectW * insectH;
                ov = insectW * insectH / union;
                if ov > ovMax, ovMax = ov; oidxMax = oidx; end;
            end;
        end;
        if ovMax >= minoverlap,
            if ~iidx2oidx2diff{ iidx }( oidxMax ),
                if ~iidx2oidx2det{ iidx }( oidxMax ),
                    rank2tp( r ) = 1;
                    iidx2oidx2det{ iidx }( oidxMax ) = true;
                else
                    rank2fp( r ) = 1;
                end;
            end;
        else
            rank2fp( r ) = 1;
        end;
    end;
    % Compute AP.
    rank2fpCum = cumsum( rank2fp );
    rank2tpCum = cumsum( rank2tp );
    rec = rank2tpCum / numPos;
    prec = rank2tpCum ./ ( rank2fpCum + rank2tpCum );
    mrec=[ 0; rec; 1 ];
    mpre=[ 0; prec; 0 ];
    for i = numel( mpre ) - 1 : -1 : 1,
        mpre( i ) = max( mpre( i ), mpre( i + 1 ) );
    end;
    i = find( mrec( 2 : end ) ~= mrec( 1 : end - 1 ) ) + 1;
    ap = sum( ( mrec( i ) - mrec( i - 1  ) ) .* mpre( i ) );
    res.cid2rec{ cid } = rec;
    res.cid2prec{ cid } = prec;
    res.cid2ap( cid ) = ap;
    res.cid2rank2tp{ cid } = rank2tp;
    res.cid2rank2fp{ cid } = rank2fp;
    res.cid2rank2iid{ cid } = rank2iid;
    res.cid2rank2tlbr{ cid } = rank2bbox;
    res.cid2rank2score{ cid } = rank2score;
    fprintf( '%s: %s) %.2f%%.\n', upper( mfilename ), db.cid2name{ cid }, ap * 100 );
end;
fprintf( '%s: mAP) %.2f%%.\n', upper( mfilename ), mean( res.cid2ap ) * 100 );