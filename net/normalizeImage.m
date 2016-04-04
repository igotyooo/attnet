function image = normalizeImage( image, rgbMean, rgbVar )
    offset = bsxfun( @plus, rgbMean, reshape( rgbVar * randn( 3, 1 ), 1, 1, 3 ) );
    image = bsxfun( @minus, image, offset );
end