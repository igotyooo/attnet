function [  cid2name, ...
            iid2impath, ...
            iid2size, ...
            iid2setid, ...
            oid2cid, ...
            oid2diff, ...
            oid2iid, ...
            oid2bbox, ...
            oid2cont ] = DB_VOC2012
    global path;
    pathtr = fullfile( path.db.voc2012.root, 'ImageSets', 'Main', 'trainval.txt' );
    pathte = fullfile( path.db.voc2012.root, 'ImageSets', 'Main', 'test.txt' );
    fptr = fopen( pathtr, 'r' );
    fpte = fopen( pathte, 'r' );
    trlist = textscan( fptr, '%s\n' );
    telist = textscan( fpte, '%s\n' );
    fclose( fptr );
    fclose( fpte );
    numIm = length( trlist{ 1 } ) + length( telist{ 1 } );
    imdir = fullfile( path.db.voc2012.root, 'JPEGImages' );
    annodir = fullfile( path.db.voc2012.root, 'Annotations' );
    triid2impath = fullfile( imdir, strcat( trlist{ 1 }, '.jpg' ) );
    teiid2impath = fullfile( imdir, strcat( telist{ 1 }, '.jpg' ) );
    iid2impath = cat( 1, triid2impath, teiid2impath );
    iid2annopath = fullfile( annodir, strcat( trlist{ 1 }, '.xml' ) );
    numTrIm = numel( triid2impath );
    iid2setid = 2 * ones( size( iid2impath ) );
    iid2setid( 1 : numTrIm ) = 1;
    iid2size = cell( numIm, 1 );
    oid2name = cell( numTrIm, 1 );
    oid2diff = cell( numTrIm, 1 );
    oid2bbox_ = cell( numTrIm, 1 );
    for iid = 1 : numIm
        if iid > numTrIm,
            [ r, c, ~ ] = size( imread( iid2impath{ iid } ) );
            iid2size{ iid } = [ r; c; ];
        else
            anno = VOCreadxml( iid2annopath{ iid } );
            anno = anno.annotation;
            iid2size{ iid } = [ str2double( anno.size.height ); str2double( anno.size.width ); ];
            oid2name{ iid } = { anno.object.name }';
            oid2diff{ iid } = logical( str2double( { anno.object.difficult }' ) );
            oid2bbox_{ iid } = { anno.object.bndbox }';
        end;
    end
    oid2name = cat( 1, oid2name{ : } );
    oid2diff = cat( 1, oid2diff{ : } );
    iid2size = cat( 2, iid2size{ : } );
    numObj = numel( oid2diff );

    oid2bbox = cell( numObj, 1 );
    oid2iid = zeros( numObj, 1 );
    oid = 0;
    for iid = 1 : numTrIm,
        no = numel( oid2bbox_{ iid } );
        for oidx = 1 : no
            oid = oid + 1;
            oid2iid( oid ) = iid;
            oid2bbox{ oid } = ...
                [   str2double( oid2bbox_{ iid }{ oidx }.ymin ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.xmin ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.ymax ); ...
                    str2double( oid2bbox_{ iid }{ oidx }.xmax ); ];
        end
    end
    oid2bbox = cat( 2, oid2bbox{ : } );
    [  cid2name, ~, oid2cid ] = unique( oid2name );
    oid2cont = cell( size( oid2cid ) );
end