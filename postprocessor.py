from collections import defaultdict
import numpy as np
import cv2
import torch

# from utils.utils import _nms, _top1
# from utils.image import get_affine_transform, affine_transform

# class TracknetV2Postprocessor(object):
#     def __init__(self, cfg):
#         #print(cfg['detector']['postprocessor'])
#         _score_threshold = cfg['detector']['postprocessor']['score_threshold']
#         _model_name      = cfg['model']['name']
#         _scales          = cfg['detector']['postprocessor']['scales']
#         _blob_det_method = cfg['detector']['postprocessor']['blob_det_method']
#         _use_hm_weight   = cfg['detector']['postprocessor']['use_hm_weight']
#         #_xy_comp_method  = cfg['detector']['postprocessor']['xy_comp_method']
#         #print(_score_threshold, _scales)

#         #_hm_type = cfg['target_generator']['type']
#         #_sigmas  = cfg['target_generator']['sigmas']
#         _sigmas = cfg['dataloader']['heatmap']['sigmas']
#         #_mags    = cfg['target_generator']['mags']
#         #_min_values = cfg['target_generator']['min_values']
#         #print(hm_type, sigmas, mags, min_values)

"""
def _detect_blob_center(self, hm):
    xy   = np.array([0., 0.])
    visi = False
    max_blob_score = -1
    if np.max(hm) > _score_threshold:
        visi = True
        th, hm_th        = cv2.threshold(hm, _score_threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
        for m in range(1,n_labels):
            ys, xs = np.where( labels==m )
            score  = xs.shape[0]
            #print(xs, ys)
            if score > max_blob_score:
                max_blob_score = score
                xy = np.array([np.mean(xs), np.mean(ys)])
                #xy = np.array([np.unique(xs).mean(), np.unique(ys).mean()])
    return xy, visi, max_blob_score
"""

def _detect_blob_concomp(hm, _score_threshold = 0.5, _use_hm_weight = False):
    xys, scores = [], []
    if np.max(hm) > _score_threshold:
        visi = True
        th, hm_th        = cv2.threshold(hm, _score_threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
        for m in range(1,n_labels):
            ys, xs = np.where( labels==m )
            ws     = hm[ys, xs]
            if _use_hm_weight:
                score  = ws.sum()
                x      = np.sum( np.array(xs) * ws ) / np.sum(ws)
                y      = np.sum( np.array(ys) * ws ) / np.sum(ws)
            else:
                score  = ws.shape[0]
                x      = np.sum( np.array(xs) ) / ws.shape[0]
                y      = np.sum( np.array(ys) ) / ws.shape[0]
                #print(xs, ys)
                #print(score, x, y)
            xys.append( np.array([x, y]) )
            scores.append( score)
    return xys, scores

def _detect_blob_nms(hm, _score_threshold = 0.5, sigma = 2.5, _use_hm_weight = False):
    xys, scores  = [], []
    hm_ori       = hm.copy()
    hm_h, hm_w   = hm.shape
    map_x, map_y = np.meshgrid(np.linspace(1, hm_w, hm_w), np.linspace(1, hm_h, hm_h))
    while True:
        cy, cx = np.unravel_index(np.argmax(hm), hm.shape)
        if hm[cy,cx] <= _score_threshold:
            break
        #dist_map = ((map_y - cy)**2) + ((map_x - cx)**2)
        dist_map = ((map_y - (cy+1))**2) + ((map_x - (cx+1))**2)
        ys, xs   = np.where( dist_map<=sigma**2)
        ws       = hm_ori[dist_map <= sigma**2]
        if _use_hm_weight:
            score  = ws.sum()
            x      = np.sum( np.array(xs) * ws ) / np.sum(ws)
            y      = np.sum( np.array(ys) * ws ) / np.sum(ws)
        else:
            score  = ws.shape[0]
            x      = np.sum( np.array(xs) ) / ws.shape[0]
            y      = np.sum( np.array(ys) ) / ws.shape[0]
            #print(xs, ys)
            #print(score, x, y)
        xys.append( np.array([x, y]) )
        scores.append( score)
        hm[dist_map<=sigma**2] = 0.
    return xys, scores

def run(preds, _blob_det_method = 'concomp', _sigmas = [2.5]):
    results = defaultdict(lambda: defaultdict(dict))
    preds_       = preds
    hms_         = preds_.sigmoid_().cpu().numpy()           

    b,s,h,w = hms_.shape
    for i in range(b):
        for j in range(s):
            #print(i,j)
            """
            if _xy_comp_method=='center':
                assert 0, 'not yet'
                #xy_, visi_, blob_size_ = _detect_blob_center(hms_[i,j])
                xy_, visi_, blob_score_ = _detect_blob_center(hms_[i,j])
            elif _xy_comp_method=='gravity':
                #xy_, visi_, blob_size_ = _detect_blob_gravity(hms_[i,j])
                #xy_, visi_, blob_score_ = _detect_blob_gravity(hms_[i,j])
                xys_, scores_ = _detect_blob_gravity(hms_[i,j])
            """
            if _blob_det_method=='concomp':
                xys_, scores_ = _detect_blob_concomp(hms_[i,j])
            elif _blob_det_method=='nms':
                xys_, scores_ = _detect_blob_nms(hms_[i,j], _sigmas)
            # xys_t_ = []
            # for xy_ in xys_:
            #     xys_t_.append( affine_transform(xy_, affine_mats_[i]))
            #results[i][j] = {'xy': xy_, 'visi': visi_, 'blob_size': blob_size_}
            #results[i][j] = {'xy': xy_, 'visi': visi_, 'blob_score': blob_score_}
            results[i][j] = {'xys': xys_, 'scores': scores_}
    return results

