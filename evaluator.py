import numpy as np

def eval_single_frame(xy_pred, visi_pred, score_pred, xy_gt, visi_gt):
    tp, fp1, fp2, tn, fn = 0, 0, 0, 0, 0
    se = None
    if visi_gt:
        if visi_pred:
            if np.linalg.norm( np.array(xy_pred)-np.array(xy_gt) ) < 4:
                tp += 1
            else:
                fp1 += 1
            se = np.linalg.norm( np.array(xy_pred)-np.array(xy_gt) )**2
        else:
            fn += 1
    else:
        if visi_pred:
            fp2 += 1
        else:
            tn += 1

    return {'tp': tp, 'tn': tn, 'fp1': fp1, 'fp2': fp2, 'fn': fn, 'se': se}

# @property
# def dist_threshold(self):
#     return _dist_threshold

# @property
# def tp_all(self):
#     return _tp

# @property
# def fp1_all(self):
#     return _fp1

# @property
# def fp2_all(self):
#     return _fp2

# @property
# def fp_all(self):
#     return fp1_all + fp2_all

# @property
# def tn_all(self):
#     return _tn

# @property
# def fn_all(self):
#     return _fn

# @property
# def prec(self):
#     prec = 0.
#     if (tp_all + fp_all) > 0.:
#         prec = tp_all / (tp_all + fp_all)
#     return prec

# @property
# def recall(self):
#     recall = 0.
#     if (tp_all + fn_all) > 0.:
#         recall = tp_all / (tp_all + fn_all)
#     return recall

# @property
# def f1(self):
#     f1 = 0.
#     if prec+recall > 0.:
#         f1 = 2 * prec * recall / (prec + recall)
#     return f1

# @property
# def accuracy(self):
#     accuracy = 0.
#     if tp_all+tn_all+fp_all+fn_all > 0.:
#         accuracy = (tp_all+tn_all) / (tp_all+tn_all+fp_all+fn_all)
#     return accuracy

# @property
# def sq_errs(self):
#     return _ses

# @property
# def ap(self):
#     inds = np.argsort(-1 * np.array(_scores)).tolist()
#     tp   = 0
#     r2p  = {}
#     for i, ind in enumerate(inds, start=1):
#         tp += _ys[ind]
#         p   = tp / i
#         r   = tp / (tp_all + fn_all)
#         if not r in r2p.keys():
#             r2p[r] = p
#         else:
#             if r2p[r] < p:
#                 r2p[r] = p
#     prev_r = 0
#     ap = 0.
#     for r, p in r2p.items():
#         ap += (r-prev_r) * p
#         prev_r = r
#     return ap

# @property
# def rmse(self):
#     _rmse = - np.Inf
#     if len(sq_errs) > 0:
#         _rmse = np.sqrt(np.array(sq_errs).mean())
#     return _rmse

# def print_results(self, txt=None, elapsed_time=0., num_frames=0, with_ap=True):
#     if txt is not None:
#         log.info('{}'.format(txt))
#     if num_frames > 0:
#         log.info('Elapsed time: {}, FPS: {} ({}/{})'.format(elapsed_time, num_frames/elapsed_time, num_frames, elapsed_time))
#     if with_ap:
#         log.info('| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE | AP  |')
#         log.info('| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- | ----- |')
#         log.info('| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) | {ap:.4f} |'.format(tp=tp_all, tn=tn_all, fp1=fp1_all, fp2=fp2_all, fp=fp_all, fn=fn_all, prec=prec, recall=recall, f1=f1, accuracy=accuracy, rmse=rmse, num_ses=len(sq_errs), ap=ap))
#     else:
#         log.info('| TP   | TN   | FP1   | FP2   | FP   | FN   | Prec       | Recall       | F1       | Accuracy       | RMSE |')
#         log.info('| ---- | ---- | ----- | ----- | ---- | ---- | ---------- | ------------ | -------- | -------------- | ---- |')
#         log.info('| {tp} | {tn} | {fp1} | {fp2} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} | {accuracy:.4f} | {rmse:.2f}({num_ses}) |'.format(tp=tp_all, tn=tn_all, fp1=fp1_all, fp2=fp2_all, fp=fp_all, fn=fn_all, prec=prec, recall=recall, f1=f1, accuracy=accuracy, rmse=rmse, num_ses=len(sq_errs)))