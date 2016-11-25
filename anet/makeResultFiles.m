function makeResultFiles( results, cid2name, iid2impath, resultPath )
for cid = 1 : numel( cid2name ),
    cname = cid2name{ cid };
    fid = fopen( sprintf( resultPath, cname ), 'w' );
    if cid == 15,
        did2true = true( size( results.did2score ) );
    else
        did2true = false( size( results.did2score ) );
    end
    idx2score = results.did2score( did2true );
    idx2tlbr = results.did2tlbr( :, did2true );
    idx2impath = iid2impath( results.did2iid( did2true ) );
    for idx = 1 : sum( did2true );
        [ ~, imcode ] = fileparts( idx2impath{ idx } );
        tlbr = idx2tlbr( :, idx );
        score = idx2score( idx );
        fprintf( fid, '%s %f %f %f %f %f\n', imcode, score, tlbr( 2 ), tlbr( 1 ), tlbr( 4 ), tlbr( 3 ) );
    end;
    fclose( fid );
end;