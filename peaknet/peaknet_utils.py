import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.cm as cm
import psana
import json
import pandas as pd


def load_cxi_labels_yxhw( filename, total_size=-1, shuffle=False ):
    # H5 file IO
    with h5py.File(filename, 'r') as f:
        nPeaks = f["entry_1/result_1/nPeaks"].value
        #nStreaks = f["entry_1/result_1/nStreaks"].value
        dataset_hits = len(nPeaks)
        print('hits: ' + str(dataset_hits))
        if total_size == -1:
            n_total = dataset_hits
        else:
            n_total = total_size
        eventIdx = f["LCLS/eventNumber"][:n_total]
        peak_x_label = f['entry_1/result_1/peakXPosRaw'][:n_total,:]
        peak_y_label = f['entry_1/result_1/peakYPosRaw'][:n_total,:]
#         streak_p_label = f["entry_1/result_1/streakPanel"][:n_total,:]
#         streak_x_label = f["entry_1/result_1/streakXPos"][:n_total,:]
#         streak_y_label = f["entry_1/result_1/streakYPos"][:n_total,:]
#         streak_w_label = f["entry_1/result_1/streakWidth"][:n_total,:]
#         streak_h_label = f["entry_1/result_1/streakHeight"][:n_total,:]
        peak_x_center = f['entry_1/result_1/peak2'][:n_total,:]
        peak_y_center = f['entry_1/result_1/peak1'][:n_total,:]
        peak_x_min = f['entry_1/result_1/peak5'][:n_total,:]
        peak_x_max = f['entry_1/result_1/peak6'][:n_total,:]
        peak_y_min = f['entry_1/result_1/peak3'][:n_total,:]
        peak_y_max = f['entry_1/result_1/peak4'][:n_total,:]
    # image reshaping

    # label formatting
    labels = []
    for i in range(n_total):
        nObj = nPeaks[i] #+ nStreaks[i]
        offset = nPeaks[i]
        cls = np.zeros( (nObj,) )
        s = np.zeros( (nObj,) )
        r = np.zeros( (nObj,) )
        c = np.zeros( (nObj,) )
        ww = np.zeros( (nObj,) )
        hh = np.zeros( (nObj,) )
        for j in range(nPeaks[i]):
#             print(peak_y_min[i,j], peak_y_max[i,j], peak_x_min[i,j], peak_x_max[i,j])
            my_s = (int(peak_y_label[i,j])/185) + (int(peak_x_label[i,j])/388)*8
            my_r = np.fmod( 0.5*(peak_y_min[i,j] + peak_y_max[i,j]), 185.0 )
            my_c = np.fmod( 0.5*(peak_x_min[i,j] + peak_x_max[i,j]), 388.0 )
            s[j] = my_s
            r[j] = my_r
            c[j] = my_c
            hh[j] = 1*(peak_y_max[i,j] - peak_y_min[i,j]) + 3
            ww[j] = 1*(peak_x_max[i,j] - peak_x_min[i,j]) + 3
#             print(r[i], c[i], hh[i], ww[i])
        my_label = (cls, s, r, c, hh, ww)
        labels.append( my_label )
    # randomize    
    if shuffle:
        rand_idxs = np.random.permutation( n_total )
        labels = [labels[i] for i in rand_idxs]
        eventIdx = [eventIdx[i] for i in rand_idxs]
    return labels, eventIdx

def psanaImageLoader(rows):
    print("rows", rows)
#     print( str(rows.loc[0,"exp"]) )
    # rows is a subset of a df
    psana_runs = {}
    n = len(rows)
    imgs = np.zeros( (n, 32, 185, 388) )
#     print("rows.iloc[:,0]", rows.iloc[:,0])
    for i, j  in enumerate(rows.index):
        # i: 0,1,2,...
        # j: DF index
        print("i", i)
        exp = str(rows.loc[j,"exp"])
        run = str(rows.loc[j,"run"])
        det = str(rows.loc[j,"detector"])
        
        if (exp, run, det) in psana_runs:
            (this_run, det, times) = psana_runs[(exp, run, det)]
        else:
            (this_run, detector, times) = psanaRun(exp, run, det)
            psana_runs[(exp, run, det)] = (this_run, detector, times)
            
        event_idx = int(rows.loc[j,"event"]) 
        evt = this_run.event(times[event_idx])
        calib = detector.calib(evt) * detector.mask(evt, calib=True, status=True,
                              edges=True, central=True,
                              unbond=True, unbondnbrs=True)
        imgs[i,:,:,:] = calib
    imgs = imgs#/np.max(imgs)    
    return imgs    
    

def json_parser(filename, mode="validate", subset=False):
    #data = json.load( open("/reg/neh/home/liponan/ai/peaknet4antfarm/val_and_test.json") )
    data = json.load( open(filename) )
    data = data[mode]
    n = len(data["experiment"])
    df = pd.DataFrame(columns=["exp", "run", "detector", "event"])
    for i in range(n):
        exp = data["experiment"][i]
        run = data["run"][i]
        det = data["detector"][i]
        parsed = False
        if subset:
            if isinstance(data["subsetEvents"][i], (list,)):
                events = data["subsetEvents"][i]
                parsed = True
            else:
                events = data["subsetEvents"][i].split(":")
        else:
            if isinstance(data["events"][i], (list,)):
                events = data["events"][i]
                parsed = True
            else:
                events = data["events"][i].split(":")
        if parsed:
            pass
        else:
            if len(events) == 3:
                events = list(range( int(events[0]), int(events[2]), int(events[1]) ))
            elif len(events) == 2:
                if len(events[0]) > 0 and len(events[1]) > 0:
                    events = list(range( int(events[0]), int(events[1]) ))
                else:
                    events = list(range(nEvents(str(exp), str(run), str(det))))
            elif len(events) == 1:
                events = [int(events[0])]
        my_data = [ [exp, run, det, e, False] for e in events ]
        my_df = pd.DataFrame(my_data, columns=["exp", "run", "detector", "event", "processed"])
        df = pd.concat( (df, my_df), axis=0 )
    df = df.reset_index(drop=True)
    return df


def psanaRun(exp_name, run, detector="DscCsPad"):
    ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
    det = psana.Detector(detector)
    this_run = ds.runs().next()
    times = this_run.times()
    return (this_run, det, times)


def nEvents(exp_name, run, detector="DscCsPad"):
    ds = psana.DataSource("exp=" + exp_name + ":run=" + str(run) + ":idx")
    det = psana.Detector(detector)
    this_run = ds.runs().next()
    times = this_run.times()
    num_events = len(times)
    return num_events


def loadLabels(path, num):
    data = []
    for u in range(num):
        labels = []
        txt = open( path + str(u).zfill(6) + ".txt", 'r').readlines()
        for v, line in enumerate(txt):
            vals = line.split(" ")
            x = round( w * float(vals[1]))
            y = round( h * float(vals[2]))
            labels.append( (x,y) )
        data.append( labels )
    return data


def visualize( imgs, labels=None, normalized_labels=False, nms_boxes=None, plot_label=False, plot_box=False, 
              box_size=7, indexes=[], vmin=0, vmax=1, ignore_miss=True, img_no=0, verbose=False):

    n, m, h, w = imgs.shape
    
    for i in indexes:
        box = nms_boxes[img_no][i]
        if len(box) == 0 and ignore_miss:
            continue
        fig, ax = plt.subplots(1)
        if verbose:
            print("imgs shape", imgs.shape)
        img = imgs[img_no,i,:,:]
        #h, w = img.shape
        h, w = 192, 392
        # plot image
        im0 = plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cm.gray)
        plt.title( "ASIC " + str(i) ) 
        fig.set_size_inches(12, 6)
        # plot labels
        if plot_label and normalized_labels is not None:
            my_r = normalized_labels[img_no][2][ normalized_labels[img_no][1] == i ] #* h
            my_c = normalized_labels[img_no][3][ normalized_labels[img_no][1] == i ] #* w
            my_h = normalized_labels[img_no][4][ normalized_labels[img_no][1] == i ] #* h
            my_w = normalized_labels[img_no][5][ normalized_labels[img_no][1] == i ] #* w
            my_l = normalized_labels[img_no][0][ normalized_labels[img_no][1] == i ]
            if verbose:
                print(my_r, my_c, my_h, my_w, my_l)
            for j in range(len(my_r)):
                x = my_c[j]# 
                y = my_r[j]# 
                ww = my_w[j]
                hh = my_h[j]
                if my_l[j] == 0:
                    color = "m"
                    rect = pat.Circle( (x, y), box_size/2.0, color=color, fill=False, linewidth=1 )
                else:
                    color = "c"
                    rect = pat.Rectangle( (x-my_w[j]/2.0, y-my_h[j]/2.0), 
                                           ww, hh, color=color, fill=False, linewidth=1 )
                ax.add_patch(rect)
        if plot_label and labels is not None:
            my_r = labels[2][ labels[1] == i ] * h
            my_c = labels[3][ labels[1] == i ] * w
            my_h = labels[4][ labels[1] == i ] * h
            my_w = labels[5][ labels[1] == i ] * w
            my_l = labels[0][ labels[1] == i ]
            print(my_r, my_c, my_h, my_w, my_l)
            for j in range(len(my_r)):
                x = my_c[j] - my_w[j]/2.0
                y = my_r[j] - my_h[j]/2.0
                ww = box_size
                hh = box_size
                if my_l[j] == 0:
                    color = "m"
                else:
                    color = "c"
                rect = pat.Rectangle( (x, y), ww, hh, color=color, fill=False, linewidth=1 )
                ax.add_patch(rect)
        # plot predictions
        if plot_box:
            for peak in nms_boxes[img_no][i]:
                x = w * ( peak[0]-0.5*peak[2] ) - 2
                y = h * ( peak[1]-0.5*peak[3] ) - 4
                ww = w * peak[2]
                hh = h * peak[3]
                if peak[5] > peak[6]:
                    color = "g"
                else:
                    color = "y"
                rect = pat.Rectangle( (x, y), ww, hh, color=color, fill=False, linewidth=1 )
                ax.add_patch(rect)

                
def load_cxi_labels( filename, box_size=7, total_size=-1, shuffle=False ):
    # H5 file IO
    with h5py.File(filename, 'r') as f:
        nPeaks = f["entry_1/result_1/nPeaks"].value
        nStreaks = f["entry_1/result_1/nStreaks"].value
        dataset_hits = len(nPeaks)
        print('hits: ' + str(dataset_hits))
        if total_size == -1:
            n_total = dataset_hits
        else:
            n_total = total_size
        eventIdx = f["LCLS/eventNumber"][:n_total]
        peak_x_label = f['entry_1/result_1/peakXPosRaw'][:n_total,:]
        peak_y_label = f['entry_1/result_1/peakYPosRaw'][:n_total,:]
        streak_p_label = f["entry_1/result_1/streakPanel"][:n_total,:]
        streak_x_label = f["entry_1/result_1/streakXPos"][:n_total,:]
        streak_y_label = f["entry_1/result_1/streakYPos"][:n_total,:]
        streak_w_label = f["entry_1/result_1/streakWidth"][:n_total,:]
        streak_h_label = f["entry_1/result_1/streakHeight"][:n_total,:]
    # image reshaping

    # label formatting
    labels = []
    for i in range(n_total):
        nObj = nPeaks[i] + nStreaks[i]
        offset = nPeaks[i]
        cls = np.zeros( (nObj,) )
        s = np.zeros( (nObj,) )
        r = np.zeros( (nObj,) )
        c = np.zeros( (nObj,) )
        ww = np.zeros( (nObj,) )
        hh = np.zeros( (nObj,) )
        for j in range(nPeaks[i]):
            my_s = (int(peak_y_label[i,j])/185) + (int(peak_x_label[i,j])/388)*8
            my_r = peak_y_label[i,j] % 185
            my_c = peak_x_label[i,j] % 388
            s[j] = my_s
            r[j] = my_r
            c[j] = my_c
            hh[j] = box_size
            ww[j] = box_size
        for j in range(nStreaks[i]):
            my_s = streak_p_label[i,j]
            my_r = streak_x_label[i,j]
            my_c = streak_y_label[i,j]
            my_h = streak_w_label[i,j]
            my_w = streak_h_label[i,j]
            s[offset+j] = my_s
            r[offset+j] = my_r
            c[offset+j] = my_c
            hh[offset+j] = np.minimum( np.minimum( 2*my_r, my_h ), 2*(185-my_r) )
            ww[offset+j] = np.minimum( np.minimum( 2*my_c, my_w ), 2*(388-my_c) )
            cls[offset+j] = 1
        my_label = (cls, s, r, c, hh, ww)
        labels.append( my_label )
    # randomize    
    if shuffle:
        rand_idxs = np.random.permutation( n_total )
        labels = [labels[i] for i in rand_idxs]
        eventIdx = [eventIdx[i] for i in rand_idxs]
    return labels, eventIdx

def psana_img_loader(eventIdxs, startIdx, n, det, this_run, times):
    imgs = np.zeros( (n, 32, 185, 388) )
    for i, event_idx in enumerate(eventIdxs[startIdx:(startIdx+n)]):
        evt = this_run.event(times[event_idx])
        calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                              edges=True, central=True,
                              unbond=True, unbondnbrs=True)
        imgs[i,:,:,:] = calib
    return imgs
    

def build_dataset( filename, dev_size, box_size=7, total_size=-1 ):
    # H5 file IO
    with h5py.File(filename, 'r') as f:
        nPeaks = f["entry_1/result_1/nPeaks"].value
        dataset_hits = len(nPeaks)
        print('hits: ' + str(dataset_hits))
        if total_size == -1:
            n_total = dataset_hits
        else:
            n_total = total_size
        imgs = f["entry_1/data_1/data"][:n_total,:,:]
        masks = f["entry_1/data_1/mask"][:n_total,:,:]
        imgs= imgs * (1-masks)
        x_label = f['entry_1/result_1/peakXPosRaw'][:n_total,:]
        y_label = f['entry_1/result_1/peakYPosRaw'][:n_total,:]
    # image reshaping
    imgs = np.reshape( imgs, (-1, 8, 185, 4, 194*2) )
    imgs = np.transpose( imgs, (0, 1, 3, 2, 4) )
    imgs = np.reshape( imgs, (-1, 32, 185, 388) )
    n, m, h, w = imgs.shape
    # label formatting
    labels = []
    for i in range(n_total):
        cls = np.zeros( (nPeaks[i],) )
        s = np.zeros( (nPeaks[i],) )
        r = np.zeros( (nPeaks[i],) )
        c = np.zeros( (nPeaks[i],) )
        ww = np.zeros( (nPeaks[i],) )
        hh = np.zeros( (nPeaks[i],) )
        for j in range(nPeaks[i]):
            my_s = (int(y_label[i,j])/185)*4 + (int(x_label[i,j])/388)
            my_r = y_label[i,j] % 185
            my_c = x_label[i,j] % 388
            s[j] = my_s
            r[j] = my_r
            c[j] = my_c
            hh[j] = box_size
            ww[j] = box_size
        my_label = (cls, s, r, c, hh, ww)
        labels.append( my_label )
    # randomize    
    rand_idxs = np.random.permutation( n_total )
    imgs = imgs[ rand_idxs, :, :, : ]
    labels = [labels[i] for i in rand_idxs]
    # build dev set
    if dev_size > 0:
        if dev_size < 1:
            dev_size = round( dev_size * dataset_hits )
    dev_imgs = imgs[:dev_size,:,:,:]
    dev_labels = labels[:dev_size]
    train_imgs = imgs[dev_size:,:,:,:]
    train_labels = labels[dev_size:]

    return train_imgs, train_labels, dev_imgs, dev_labels


def load_from_cxi( filename, idx, box_size=7 ):
    f = h5py.File(filename, 'r')
    nPeaks = f["entry_1/result_1/nPeaks"].value
    dataset_hits = len(nPeaks)
    #print('hits: ' + str(dataset_hits))
    dataset_peaks = np.sum(nPeaks)
    #print('peaks: ' + str(dataset_peaks))
    img = f["entry_1/data_1/data"][idx,:,:]
    mask = f["entry_1/data_1/mask"][idx,:,:]
    img = img * (1-mask).astype(float)
    x_label = f['entry_1/result_1/peakXPosRaw'][idx,:]
    y_label = f['entry_1/result_1/peakYPosRaw'][idx,:]
    f.close()

    imgs = np.reshape( img, (8, 185, 4, 194*2) )
    imgs = np.transpose( imgs, (0, 2, 1, 3) )
    imgs = np.reshape( imgs, (1, 32, 185, 388) )
    n, m, h, w = imgs.shape

    cls = np.zeros( (nPeaks[idx],) )
    s = np.zeros( (nPeaks[idx],) )
    r = np.zeros( (nPeaks[idx],) )
    c = np.zeros( (nPeaks[idx],) )
    ww = np.zeros( (nPeaks[idx],) )
    hh = np.zeros( (nPeaks[idx],) )
    for u in range(nPeaks[idx]):
        my_s = (int(y_label[u])/185)*4 + (int(x_label[u])/388)
        my_r = y_label[u] % 185
        my_c = x_label[u] % 388
        s[u] = my_s
        r[u] = my_r
        c[u] = my_c
        hh[u] = box_size
        ww[u] = box_size
    labels = (cls, s, r, c, hh, ww)

    return imgs, labels
