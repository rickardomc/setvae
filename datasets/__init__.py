from . import ShapeNet, SetMNIST, SetMultiMNIST, ArCH


def get_datasets(args):
    if args.dataset_type == 'shapenet15k':
        return ShapeNet.build(args)

    if args.dataset_type == 'mnist':
        return SetMNIST.build(args)

    if args.dataset_type == 'multimnist':
        return SetMultiMNIST.build(args)

    if args.dataset_type == 'arch':
        return ArCH.build(args)

    raise NotImplementedError
