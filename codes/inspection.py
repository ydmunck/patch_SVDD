from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(dataset, loader, enc):
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        
        print("Patches: " + str(len(loader)))
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

class Evaluate:
    def __init__(self):
        self.standardized_images_train = None
        self.standardized_images_test = None
        
        self.dataset_train_64 = None
        self.dataset_train_32 = None
        self.dataset_test_64 = None
        self.dataset_test_32 = None
        
        self.loader_train_64 = None
        self.loader_train_32 = None
        self.loader_test_64 = None
        self.loader_test_32 = None

        
    def set_standardized_images_train(self, standardized_images_train):
        self.standardized_images_train = standardized_images_train
        
        transposed = NHWC2NCHW(self.standardized_images_train)
        
        self.dataset_train_64 = PatchDataset_NCHW(transposed, K=64, S=16)
        self.dataset_train_32 = PatchDataset_NCHW(transposed, K=32, S=4)
        
        self.loader_train_64 = DataLoader(dataset_train_64, batch_size=512, shuffle=False, pin_memory=True, num_workers=32)
        self.loader_train_32 = DataLoader(dataset_train_32, batch_size=512, shuffle=False, pin_memory=True, num_workers=32)

    def set_standardized_images_test(self, standardized_images_test):
        self.standardized_images_test = standardized_images_test
        
        transposed = NHWC2NCHW(self.standardized_images_test)
        
        self.dataset_test_64 = PatchDataset_NCHW(transposed, K=64, S=16)
        self.dataset_test_32 = PatchDataset_NCHW(transposed, K=32, S=4)
        
        self.loader_test_64 = DataLoader(dataset_test_64, batch_size=512, shuffle=False, pin_memory=True, num_workers=32)
        self.loader_test_32 = DataLoader(dataset_test_32, batch_size=512, shuffle=False, pin_memory=True, num_workers=32)

    def eval_encoder_NN_multiK(self, enc):
        print("First embeddings")
        embs64_train = infer(self.dataset_train_64, self.loader_train_64, enc)
        embs64_te = infer(self.dataset_test_64, self.loader_test_64, enc)

        print("Last embeddings")
        embs32_tr = infer(self.dataset_train_32, self.loader_train_32, enc.enc)
        embs32_te = infer(self.dataset_test_32, self.loader_test_32, enc.enc)

        embs64 = embs64_train, embs64_te
        embs32 = embs32_train, embs32_te
        return embs64, embs32


    def eval_embeddings_NN_multiK(self, obj, embs64, embs32, NN=1):
        emb_tr, emb_te = embs64
        maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
        maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)

        emb_tr, emb_te = embs32
        maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
        maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)

        maps_sum = maps_64 + maps_32
        maps_mult = maps_64 * maps_32

        return {
            'maps_64': maps_64,
            'maps_32': maps_32,
            'maps_sum': maps_sum,
            'maps_mult': maps_mult,
        }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
