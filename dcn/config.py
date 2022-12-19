
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DCN Benchmark")
    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="generate hou many data for virtual")
    parser.add_argument("--embed_dims", type=int, default=4, help="embedding dims")
    
    return parser.parse_args()