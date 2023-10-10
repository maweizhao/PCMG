import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
#print(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)
from _parser.evaluate import parser



def main():
    args,parameters, folder, checkpointname, niter = parser()
    
    #parameters, args = parser()
    
     

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12","uestc","cmu"] and args.evaluate_model == "GRU":
        from  evaluate.gru_eval import evaluate
        evaluate(args,parameters, folder, checkpointname, niter)
    elif dataset in ["ntu13", "humanact12","uestc","cmu"] and args.evaluate_model == "P4Transformer":
        from  evaluate.p4transformer_eval import evaluate
        evaluate(args,parameters, folder, checkpointname, niter)
    elif dataset in ["uestc"] and args.evaluate_model == "STGCN":
        from evaluate.stgcn_eval import evaluate
        evaluate(args,parameters, folder, checkpointname, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()