function [ patchSide, stride ] = ...
    getNetProperties( net )
    patchSide = net.meta.normalization.imageSize( 1 );
    numChannel = net.meta.normalization.imageSize( 3 );
    % Determine patch size.
    while true,
        try
            im = zeros...
                ( patchSide, patchSide, numChannel, 'single' );
            vl_simplenn( net, im );
            clear im; clear ans;
            patchSide = patchSide - 1;
        catch
            patchSide = patchSide + 1;
            break;
        end;
    end
    % Determine patch stride.
    stride = 0;
    while true,
        im = zeros...
            ( patchSide + stride, patchSide, numChannel, 'single' );
        res = vl_simplenn( net, im );
        desc = res( end ).x;
        clear im; clear res;
        if size( desc, 1 ) == 2, break; end;
        stride = stride + 1;
    end;
end