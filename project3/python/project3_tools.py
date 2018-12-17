import numpy as np

def create_parser():
    """https://codereview.stackexchange.com/questions/79008/parse-a-config-file-and-add-to-command-line-arguments-using-argparse-in-python"""
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Device Targets')
    g.add_argument( '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
    g.add_argument('-T', '--final_time', default=1, type=int)
    g.add_argument('--Nx', default = 10, type=int)
    g.add_argument('--set_Nt', action='store_true')
    g.add_argument('--Nt_min', default = 10, type=int)
    return parser

def parse_args(parser):
    """https://codereview.stackexchange.com/questions/79008/parse-a-config-file-and-add-to-command-line-arguments-using-argparse-in-python"""
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args

