import numpy as np
from peaknet.Peaknet import Peaknet
from peaknet.peaknet_utils import conv_peaks_to_psana, conv_peaks_to_array
from psana import *
from psalgos.pypsalgos import PyAlgos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import time

expName = 'cxic0515'
runNum = 13
detName = 'DsdCsPad'
evtNum = 811
rank=3 # psana peak finder window rank
doPlot = 1
windowSize=float(rank)*2+1

def plotPeaks(calib, peaks, peaks1):
    for seg in range(calib.shape[0]):
        fig = plt.figure()
        ax = plt.gca()
        plt.imshow(calib[seg,:,:], cmap=cm.gray, vmin=0)
        plt.title("seg: {}".format(seg))
        rect = None
        circ = None
        for peak in peaks:
            if peak[0] == seg:
                if rect is None:
                    rect = patches.Rectangle((peak[2]-windowSize/2,peak[1]-windowSize/2),
                           windowSize,windowSize,linewidth=1,edgecolor='c',facecolor='none', label="psana")
                else:
                    rect = patches.Rectangle((peak[2]-windowSize/2,peak[1]-windowSize/2),
                           windowSize,windowSize,linewidth=1,edgecolor='c',facecolor='none')
                ax.add_patch(rect)
        for peak in peaks1:
            if peak[0] == seg:
                if circ is None:
                    circ = plt.plot(peak[2], peak[1], "om", fillstyle="none", label="peaknet")
                else:
                    circ = plt.plot(peak[2], peak[1], "om", fillstyle="none")
        if rect is not None or circ is not None:
            plt.legend()
        plt.show()

t0=time.time()
peaknet = Peaknet()
t1=time.time()
peaknet.loadDefaultCFG() # ~liponan/ai/peaknet4antfarm_v2/model_init.pt
t2=time.time()
ds = DataSource('exp='+expName+':run='+str(runNum)+':idx')
det = Detector(detName)
run = ds.runs().next()
times = run.times()
t3=time.time()
alg = PyAlgos()
alg.set_peak_selection_pars(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
mask=det.mask(run, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True, unbondnbrs8=False).astype(np.uint16)
print("Init Peaknet {0:.3f} sec".format(t1-t0))
print("Load Peaknet config {0:.3f} sec".format(t2-t1))
print("Psana setup {0:.3f} sec".format(t3-t2))

for nevt,t in enumerate(times[evtNum:evtNum+1]):
   t4=time.time()
   evt = run.event(t)
   t5=time.time()
   calib = det.calib(evt)
   if calib is None:
       print 'None'
       continue
   [seg,row,col] = calib.shape
   imgs = calib * mask
   img_h = imgs.shape[1]
   img_w = imgs.shape[2]

   t6=time.time()
   # psana peak finder
   psana_peaks = alg.peak_finder_v3r3(imgs, rank=rank, r0=3, dr=2, nsigm=10, mask=mask)
   t7=time.time()

   # peaknet peak finder
   _imgs = imgs.reshape(-1, 1, imgs.shape[1], imgs.shape[2])
   results = peaknet.predict(_imgs, conf_thresh=0.5)
   t8=time.time()

   peaknet_peaks = conv_peaks_to_psana(results, img_h, img_w)

   print("Get evt {0:.3f} sec".format(t5-t4))
   print("det.calib {0:.3f} sec".format(t6-t5))
   print("psana peaks {0:.3f} sec".format(t7-t6))
   print("peaknet peaks {0:.3f} sec".format(t8-t7))
   print("psana peaks:\n {}".format(psana_peaks[:,0:3]))
   print("peaknet peaks:\n {}".format(peaknet_peaks[:,0:3]))

   if doPlot:
       plotPeaks(imgs, psana_peaks, peaknet_peaks)


