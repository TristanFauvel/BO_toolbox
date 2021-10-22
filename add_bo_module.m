currentFile = mfilename( 'fullpath' );
[pathname,~,~] = fileparts( currentFile );
addpath(genpath(pathname))

addpath(genpath([fileparts(pathname), '/GP_toolbox']))