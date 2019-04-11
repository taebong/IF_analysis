import skimage
from skimage import measure
from skimage import segmentation

import numpy as np
import scipy as sp
import pandas as pd
import re
import os
import gc
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics
import sys


if len(sys.argv) != 4:
    print("Use: python measureNucleiProperties.py input_dir output_dir binning_factor")
    sys.exit()
else:
    data_dir = sys.argv[1]  #data pth
    analysis_dir = sys.argv[2]    #analysis pth
    binning_factor = int(sys.argv[3])

all_files = os.listdir(data_dir)
regexp = re.compile('Well(?P<Well>\d*?)_(?P<Pos>\d*?)_w(?P<Ch>\d{1}).*.tif',re.IGNORECASE)
meta_list = []
for f in all_files:
    r = regexp.match(f)
    if r:
        meta=r.groupdict()
        meta['Filename'] = f
        meta_list.append(meta)
        
df_meta = pd.DataFrame(meta_list)
df_meta = df_meta.sort_values(['Well','Pos','Ch'])
df_meta = df_meta.reset_index(drop=True)

seg_files = os.listdir(analysis_dir+'segm/')
seg_files_dapi = [f for f in seg_files if 'DAPI' in f]


def downsample(im,binning_factor):
    sz = np.array(im.shape)
    
    # if sz[0] and sz[1] are not multiples of BINNING_FACTOR, reduce them to the largest multiple of BINNING_FACTOR and crop image
    newsz = sz//binning_factor
    cropsz = newsz*binning_factor
    im = im[0:cropsz[0],0:cropsz[1]]

    newim = im.reshape((newsz[0],binning_factor,newsz[1],binning_factor))
    return newim.mean(-1).mean(1)

def normalize_image(im,low=None,high=None):
    if low==None:
        low = np.min(im)
    if high==None:
        high = np.max(im)
    
    im = np.minimum(high,im)
    im = np.maximum(low,im)
    
    im = (im-low)/(high-low)
    
    return im    

def subtractBg(im,method='min'):
    if method=='mode':
        filtered = skimage.filters.gaussian(im,sigma=2)
        h = np.histogram(filtered,bins=500);
        bg = h[1][h[0].argmax()]
    elif method=='min':
        bg = im.flatten().min()
    im[im<bg] = bg
    im = im-bg
    return im
    
def getLabelColorMap():
    colors = plt.cm.jet(range(256))
    np.random.shuffle(colors)
    colors[0] = (0.,0.,0.,1.)
    #rmap = c.ListedColormap(colors)
    return colors

def showSegmentation(label_im,norm_im1,norm_im2,rmap,zoom=3,fig=None):
    
    sz = label_im.shape
    
    combined = np.moveaxis([norm_im1,norm_im2,np.zeros(sz)],0,2)
    
    if fig:
        axs = fig.subplots(2, 2,sharex=True, sharey=True);
    else:
        fig,axs = plt.subplots(2, 2,sharex=True, sharey=True);
    
    w,h = plt.figaspect(sz[0]/sz[1]);
    fig.set_size_inches(w * zoom, h * zoom);
    
    axs[0][0].imshow(rmap[label_im%256]);    
    axs[0][1].imshow(segmentation.mark_boundaries(norm_im1,label_im,mode='inner',color=None,outline_color=[1,0,0]));
    axs[1][0].imshow(segmentation.mark_boundaries(norm_im2,label_im,mode='inner',color=None,outline_color=[1,0,0]));
    axs[1][1].imshow(segmentation.mark_boundaries(combined,label_im,mode='inner',color=None,outline_color=[1,1,1]));
    
    rps = measure.regionprops(label_im)
    
    #X,Y = np.meshgrid(np.arange(sz[1]),np.arange(sz[0]))
    
    for rp in rps:
        yc,xc = rp.centroid

        for ax in axs.flatten()[:3]:
            ax.text(xc,yc,rp.label,fontsize=3*zoom,
                     horizontalalignment='center',
                     verticalalignment='center',color='k');
            
    for ax in axs.flatten():
        ax.set_yticks([]);
        ax.set_xticks([]);
    
    
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.02,wspace=0.02);
    
    fig.tight_layout()

    return fig,axs

savedir = analysis_dir+'combined_images/'
os.makedirs(savedir,exist_ok=True)

grp = df_meta.groupby(['Well','Pos'])

rmap = getLabelColorMap()
fig = plt.figure();
df_props_all = pd.DataFrame()
for ind,df in grp:
    well,pos = ind
    
    fname_dapi = df.iloc[np.where(df['Ch']=='1')[0][0]]['Filename']
    fname_gfp = df.iloc[np.where(df['Ch']=='2')[0][0]]['Filename']

    #downsample and bg-subtract
    im_dapi = downsample(imageio.imread(os.path.join(data_dir,fname_dapi)),binning_factor)
    
    im_gfp = downsample(imageio.imread(os.path.join(data_dir,fname_gfp)),binning_factor)
    im_gfp = subtractBg(im_gfp)
    
    label_im = imageio.imread(analysis_dir+'segm/'+fname_dapi[:-3]+'png')
    
    rps = measure.regionprops(label_im,im_gfp)
    prop_list = []
    for rp in rps:
        l = rp.label

        prop = dict([(i, rp[i]) for i in ('label','area','eccentricity','perimeter',
                                          'orientation','mean_intensity','min_intensity',
                                          'max_intensity')])
        yc,xc = rp.centroid
        prop['xc'] = xc
        prop['yc'] = yc

        perimeter_ints = im_gfp[segmentation.find_boundaries(label_im == l)]
        prop['mean_edge_intensity_gfp'] = perimeter_ints.mean()
        prop['min_edge_intensity_gfp'] = perimeter_ints.min()
        prop['max_edge_intensity_gfp'] = perimeter_ints.max()

        prop_list.append(prop)

    df_props = pd.DataFrame(prop_list)
    df_props['Well'] = well
    df_props['Pos'] = pos
    df_props_all = pd.concat((df_props_all,df_props))
    
    
    fig,axs = showSegmentation(label_im,normalize_image(im_dapi),normalize_image(im_gfp),rmap,zoom=3,fig=fig);
    
    fig.savefig(savedir+'Well%s_Pos%s.jpg' %(well,pos),
               frameon=False,facecolor=None,edgecolor=None,quality=80);

    plt.clf() 

    gc.collect()


df_props_all.to_csv(analysis_dir+'nuclei_properties.csv',index=False)

