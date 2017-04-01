%% create symlinks for mgz files
% note -- need to be running matlab in admin mode

%% paths
addpath(genpath('C:/Users/adalca/Dropbox (Personal)/MATLAB/external_toolboxes/enhanced_rdir'));
addpath(genpath('C:/Users/adalca/Dropbox (Personal)/MATLAB/toolboxes'));

origDataFolder = 'D:\Dropbox (MIT)\Research\fsCNN\data\orig';
coreLinksFolder = 'D:\Dropbox (MIT)\Research\fsCNN\data\simlinks';

%% run
% g = rdir(fullfile(origDataFolder, '**', 'norm.mgz'));

% split between train (50%), validate (30%) and test (20%)
split = [0.5, 0.3, 0.2];
assert(isclose(sum(split), 1))
r = randperm(numel(g));

% make folders
mkdir(fullfile(coreLinksFolder, 'asegs'))
mkdir(fullfile(coreLinksFolder, 'vols'))
mkdir(fullfile(coreLinksFolder, 'train', 'asegs'))
mkdir(fullfile(coreLinksFolder, 'train', 'vols'))
mkdir(fullfile(coreLinksFolder, 'test', 'asegs'))
mkdir(fullfile(coreLinksFolder, 'test', 'vols'))
mkdir(fullfile(coreLinksFolder, 'validate', 'asegs'))
mkdir(fullfile(coreLinksFolder, 'validate', 'vols'))

% go through files
vi = verboseIter(r, 2);
while vi.hasNext
    [i, idx] = vi.next();
    srcVolFileName = g(i).name;
    
    % get file parts to create file name
    [fpath, fname, fext] = fileparts(g(i).name);
    subFolders = fpath(numel(origDataFolder) + 2:end);
    subFolders = strrep(subFolders, '\', '_');
    subFolders = strrep(subFolders, '/', '_');
    
    % make sure the seg exists
    srcSegFileName = fullfile(fpath, ['aseg', fext]);
    if ~sys.isfile(srcSegFileName)
        warning('skipping %s', g(i).name);
        continue;
    end
    
    % create links in core folder
    dstFileName = fullfile(coreLinksFolder, 'vols', sprintf('%s_%s%s', subFolders, 'norm', fext));
    str = sprintf('mklink "%s" "%s"', dstFileName, srcVolFileName);
    [s, ~] = system(str); assert(s == 0, 'vol simlink failed');
    
    dstFileName = fullfile(coreLinksFolder, 'asegs', sprintf('%s_%s%s', subFolders, 'aseg', fext));
    str = sprintf('mklink "%s" "%s"', dstFileName, srcSegFileName);
    [s, ~] = system(str); assert(s == 0, 'seg simlink failed');
    
    % create links in sel folder
    if idx/numel(g) < split(1)
        folder = fullfile(coreLinksFolder, 'train');
    elseif idx/numel(g) < sum(split(1:2))
        folder = fullfile(coreLinksFolder, 'validate');
    else 
        folder = fullfile(coreLinksFolder, 'test');
    end
    
    dstFileName = fullfile(folder, 'vols', sprintf('%s_%s%s', subFolders, 'norm', fext));
    str = sprintf('mklink "%s" "%s"', dstFileName, srcVolFileName);
    [s, ~] = system(str); assert(s == 0, 'vol simlink failed');
    
    dstFileName = fullfile(folder, 'asegs', sprintf('%s_%s%s', subFolders, 'aseg', fext));
    str = sprintf('mklink "%s" "%s"', dstFileName, srcSegFileName);
    [s, ~] = system(str); assert(s == 0, 'seg simlink failed');
end

%% test loading a mgh file
% go through links, resize and 

coreProcFolder = 'D:\Dropbox (MIT)\Research\fsCNN\data_proc64';
[vol, M, mr_parms, volsz] = load_mgh('F:\Dropbox (MIT)\Research\fsCNN\ABIDE_50002_norm.nii.gz');
