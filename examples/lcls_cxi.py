import numpy as np
from peaknet.Peaknet import Peaknet
from psana import *
from psalgos.pypsalgos import PyAlgos

import time

t0=time.time()
peaknet = Peaknet()
t1=time.time()
peaknet.loadDefaultCFG() # ~liponan/ai/peaknet4antfarm_v2/model_init.pt
t2=time.time()
ds = DataSource('exp=cxic0515:run=14:idx')
det = Detector('DsdCsPad')
run = ds.runs().next()
times = run.times()
t3=time.time()
alg = PyAlgos()
alg.set_peak_selection_pars(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
mask=det.mask(run, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True, unbondnbrs8=False).astype(np.uint16)

print("init Peaknet: ",t1-t0)
print("load config: ",t2-t1)
print("psana setup: ",t3-t2)
# 10, 7, 6, 10 peaks according to psocake
for nevt,t in enumerate(times[174:178]):
   t4=time.time()
   evt = run.event(t)
   t5=time.time()
   calib = det.calib(evt)
   t6=time.time()
   if calib is None:
       print 'None'
       continue

   peaks = alg.peak_finder_v3r3(calib, rank=3, r0=3, dr=2, nsigm=10, mask=mask)

   newimg = calib[np.newaxis, ...]
   t7=time.time()
   images = peaknet.predict(newimg, conf_thresh=0.4)
   t8=time.time()
   print("+++++ nevt:",nevt)
   npeak = 0
   #print("images:")
   #print(images)
   for img in images:
       #print('*img')
       for segment in img:
           #print('*seg')
           for peak in segment:
               npeak += 1
               #print('*peak')
               # seven numbers: x,y,w,h,confidence score,category score,category id
               #print '***',peak
   #print('npeaks:',npeak)
   print("get evt: ",t5-t4)
   print("det.calib: ",t6-t5)
   print("psana peaks: ",t7-t6)
   print("peaknet predict: ",t8-t7)

