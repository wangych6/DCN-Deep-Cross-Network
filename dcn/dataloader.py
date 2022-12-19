
import numpy as np
    
    
def gen_data_dims():
    sparse_features = ['S' + str(i) for i in range(1, 27)]
    dense_features = ['D' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features
    
    real_dims = [   7912889,
                    33823,
                    582469,
                    245828,
                    11,
                    2209,
                    10667,
                    104,
                    4,
                    968,
                    15,
                    8165896,
                    17139,
                    2675940,
                    7156453,
                    302516,
                    12022,
                    97,
                    35,
                    7339,
                    20046,
                    4,
                    7105,
                    1382,
                    63,
                    5554114,
                    ]
    sparse_dims = {sparse_features[i]: real_dims[i] for i in range(26)}
    
    return sparse_dims
        
    
def create_virtual_data(args):
    feature_dims = gen_data_dims()
    # print(" == sparse feature cardinaltity: "*20, feature_dims)
    total_data_size = args.steps * args.batch_size
    feature_columns = [{"feat_name": name, "feat_num": dims, "embed_dim": args.embed_dims} for name, dims in feature_dims.items()]

    data_X = {name: np.random.randint(0, dims, size=(1, total_data_size), dtype=np.int32).squeeze() for name, dims in feature_dims.items()}
    data_y = np.random.randint(0, 2, size=(1, total_data_size), dtype=np.int8).squeeze()
    return feature_columns, (data_X, data_y)