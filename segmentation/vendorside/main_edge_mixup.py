import argparse
import trainer_edge_mixup

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
parser.add_argument('--image_size', type=int, default=256, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=19, help="number of channels in the image")
parser.add_argument('--lr', type=float, default=2.5e-4, help="starting learning rate")
parser.add_argument('--input_representation', type=float, default=19, help="number of channels in the data")
parser.add_argument('--save_every', type=int, default=200, help='saving images after every _ iterations')
parser.add_argument('--aug', type=bool, default=False, help='enable tencrop augmentations')
parser.add_argument('--start_iter', type=int, default=0, help='iteration number to start from')
parser.add_argument('--end_iter', type=int, default=100000, help='iteration number for stopping')

parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum in optimizer")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay param in optimizer")
parser.add_argument('--mixup_lambda', type=float, default=0.005, help="weight decay param in optimizer")


parser.add_argument('--dataset', nargs='*')
parser.add_argument('--model', type=str, default='deeplab', choices=['deeplab', 'fcn8s'])
parser.add_argument('--runs', type=str, default='testrun')
parser.add_argument('--load_prev_model', type=str, default='deeplab_finetune.pth', help="load model path")

parser.add_argument('--save_current_model', type=str, default='phase_1', help="model save for phase 1")
parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")


parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

FLAGS = parser.parse_args()

if __name__ == '__main__':
    trainer_edge_mixup.trainingProcedure(FLAGS)
