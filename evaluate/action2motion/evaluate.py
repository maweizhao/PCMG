import time
import torch
import numpy as np
from .models import load_classifier, load_classifier_for_fid
from ..P4Transformer.models.PCMG_classfier import load_classifier as load_p4_classifier, load_classifier_for_fid as load_p4_classifier_for_fid

from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality
from .pointcloud_seq_laplace_loss import pointcloud_seq_laplace_loss


class A2MEvaluation:
    def __init__(self,args, dataname, device):
        self.evaluate_model=args.evaluate_model
        if args.evaluate_model == "GRU":
            dataset_opt = {"ntu13": {"joints_num": 18,
                                    "input_size_raw": 54,
                                    "num_classes": 13},
                        'humanact12': {"input_size_raw": 72,
                                        "joints_num": 24,
                                        "num_classes": 12},
                            'uestc': {"input_size_raw": 72,
                                        "joints_num": 24,
                                        "num_classes": 40},
                            'cmu': {"input_size_raw": 60,
                                        "joints_num": 20,
                                        "num_classes": 8}}
        elif args.evaluate_model == "P4Transformer":
            dataset_opt = {"ntu13": {"joints_num": 1024*3,
                                    "input_size_raw": 1024,
                                    "num_classes": 13},
                        'humanact12': {"input_size_raw": 1024*3,
                                        "joints_num": 1024,
                                        "num_classes": 12},
                            'uestc': {"input_size_raw": 1024*3,
                                        "joints_num": 1024,
                                        "num_classes": 40},
                            'cmu': {"input_size_raw": 1024*3,
                                        "joints_num": 1024,
                                        "num_classes": 8}}
            
        
        if dataname != dataset_opt.keys():
            assert NotImplementedError(f"{dataname} is not supported.")
        
        self.dataname = dataname
        self.input_size_raw = dataset_opt[dataname]["input_size_raw"]
        self.num_classes = dataset_opt[dataname]["num_classes"]
        self.device = device
        
        if args.evaluate_model == "GRU":
            self.gru_classifier_for_fid = load_classifier_for_fid(dataname, self.input_size_raw,
                                                                self.num_classes, device).eval()
            self.gru_classifier = load_classifier(dataname, self.input_size_raw,
                                                self.num_classes, device).eval()
        elif args.evaluate_model == "P4Transformer":
            self.gru_classifier_for_fid = load_p4_classifier_for_fid(dataname, self.input_size_raw,
                                                                self.num_classes, device).eval()
            self.gru_classifier = load_p4_classifier(dataname, self.input_size_raw,
                                                self.num_classes, device).eval()
        
    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(motionloader):
                activations.append(self.gru_classifier_for_fid(batch["output_xyz"], lengths=batch["lengths"]))
                labels.append(batch["y"])
            activations = torch.cat(activations, dim=0)
            labels = torch.cat(labels, dim=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.cpu().numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, model, loaders):
        
        def print_logs(metric, key):
            print(f"Computing action2motion {metric} on the {key} loader ...")
            
        #print(loaders["gt"].keys())
        metrics_all = {}
        for sets in loaders["gt"].keys():
            metrics = {}
            computedfeats = {}
            for key, loaderSets in loaders.items():
                loader = loaderSets[sets]
                #print(loader)
                metric = "accuracy"
                print_logs(metric, key)
                mkey = f"{metric}_{key}"
                # accuracy
                metrics[mkey], _ = calculate_accuracy(model, loader,
                                                    self.num_classes,
                                                    self.gru_classifier, self.device)
                
                
                if self.evaluate_model == "P4Transformer" :
                    metric = "laplace_loss"
                    print_logs(metric, key)
                    mkey = f"{metric}_{key}"
                    #laplace_loss
                    metrics[mkey] = pointcloud_seq_laplace_loss(loader,self.device)
            
                # features for diversity
                print_logs("features", key)
                # 用action2motion里面的编码器对输入进行编码
                feats, labels = self.compute_features(model, loader)
                print_logs("stats", key)
                # 计算特征的mu,sigma
                stats = self.calculate_activation_statistics(feats)
                
                computedfeats[key] = {"feats": feats,
                                    "labels": labels,
                                    "stats": stats}

                # diversity,multimodality
                print_logs("diversity", key)
                ret = calculate_diversity_multimodality(feats, labels, self.num_classes)
                metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret
            
            # taking the stats of the ground truth and remove it from the computed feats
            gtstats = computedfeats["gt"]["stats"]
            # computing fid
            for key, loader in computedfeats.items():
                metric = "fid"
                mkey = f"{metric}_{key}"
                
                stats = computedfeats[key]["stats"]
                # fid
                metrics[mkey] = float(calculate_fid(gtstats, stats))
            metrics_all[sets] = metrics
            
        metrics = {}
        for sets in loaders["gt"].keys():
            for key in metrics_all[sets]:
                metrics[f"{key}_{sets}"] = metrics_all[sets][key]
            
        return metrics
