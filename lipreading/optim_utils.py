import math
import torch
import torch.optim as optim


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)


def get_optimizer(args, model):
    # -- define optimizer
    optim_policies = model.parameters()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(optim_policies, lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        if args.transformer:
            avsep_param, other_param = parameter_seperator(model)
            param_groups = [{'params': avsep_param, 'lr' : args.lr_sep},
                            {'params' : other_param, 'lr':args.lr}]
            optimizer = optim.AdamW(param_groups,weight_decay = 1e-2)
            return optimizer
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_policies, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer

def create_optimizer(nets, opt):
        (net_lipreading, net_facial_attribtes, net_unet, net_vocal_attributes) = nets
        param_groups = [{'params': net_lipreading.parameters(), 'lr': opt.lr_lipreading},
                        {'params': net_facial_attribtes.parameters(), 'lr': opt.lr_facial_attributes},
                        {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                        {'params': net_vocal_attributes.parameters(), 'lr': opt.lr_vocal_attributes}]
        if opt.optimizer == 'sgd':
            return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adam':
            return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def parameter_seperator(model):
    avsep_param = []
    other_param=[]
    for name,param in model.named_parameters():
        if 'seperator' in name:
            avsep_param.append(param)
        else:
            other_param.append(param)
    return avsep_param,other_param