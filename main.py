
import argparse
import handler

parser = argparse.ArgumentParser( prog='Optimizing-Extended-Metro-Timetables')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--station_index', type=int, default=[])
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--num_train', type=int, default=8)
parser.add_argument('--passenger_flow', type=int, default=1000)
parser.add_argument('--fixed_timetable', type=int, default=[70, 190])
parser.add_argument('--peak_time', type=int, default=80)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=500)
parser.add_argument('--exponential_decay_step', type=int, default=400)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)

args = parser.parse_args()

if __name__ == "__main__":

  handler.main(args)