import configargparse


def get_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--cfg', is_config_file=True, help='config file path')
    # EXPERIMENT
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='the log saving directory')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--exp_sfx', type=str, default='',
                        help='suffix of the experiment')
    parser.add_argument('--i_save', type=int, default=1,
                        help='save log every i_save epochs')
    parser.add_argument('--i_image', type=int, default=1,
                        help='save validation results every i_image epochs')
    parser.add_argument('--i_test', type=int, default=10,
                        help='test model every i_test epochs')
    parser.add_argument('--N_poses', type=int, default=60,
                        help='number of test poses')
    parser.add_argument('--weight', type=str, default='',
                        help='pretrained checkpoint')
    parser.add_argument('--load_weight', default=False, action="store_true",
                        help='Whether load checkpoint')
    parser.add_argument('--fps', type=int, default=12,
                        help='fps of testing videos')
    # DATASET
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--data_type', type=str, default='llff',
                        help='dataset type', choices=['blender', 'llff'])
    parser.add_argument('--spheric', default=False, action='store_true',
                        help='images are taken in spheric poses (llff)')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[500, 500],
                        help='resolution of the images')
    parser.add_argument('--val_step', type=int, default=8,
                        help='validation step (llff)')
    parser.add_argument('--res_factor', type=int, default=4,
                        help='Help get images scaled *factor* times down')
    parser.add_argument('--white_bg', default=False, action='store_true')
    # MEMORY CONTROLLER
    parser.add_argument('--batch_size', type=int, default=1024 * 2)
    parser.add_argument('--chunk', type=int, default=1024 * 2,
                        help='split input batch into mini-batches avoid OMM')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index. If gpu=-1, using cpu')
    # MODEL
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of fine samples')
    parser.add_argument('--pos_freqs', type=int, default=10,
                        help='position embedding frequency')
    parser.add_argument('--dir_freqs', type=int, default=4,
                        help='direction embedding frequency')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='pos/dir in channels')
    parser.add_argument('--log_scale', default=False, action='store_true',
                        help='log scale the frequency')
    parser.add_argument('--depth', type=int, default=8, help='NeRF depth')
    parser.add_argument('--hid_layers', type=int, default=256,
                        help='number of NeRF hidden layers')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--skips', nargs="+", type=int, default=[4],
                        help='NeRF skip connection at k-th layers')
    # LOSS
    parser.add_argument('--loss', type=str, choices=['mse'], default='mse',
                        help='loss function')
    # TRAINER
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--i_batch_save', type=int, default=-1,
                        help='Save ckpt after i batch')
    return parser.parse_args()
