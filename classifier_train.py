
import jittor as jt
from jittor import init
from jittor import nn
import jittor.distributions as dist
from jittor.optim import AdamW
import argparse
import os
import datetime
import blobfile as bf
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict, classifier_and_diffusion_defaults, create_classifier_and_diffusion
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

def main():
    jt.flags.use_cuda = 1
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    outdir = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    logf = open('last_output_dir.txt','w')
    logf.write(outdir)
    logf.close()
    outdir = './output/' + outdir
    logger.configure(dir = outdir)
    logger.log('creating model and diffusion...')
    (model, diffusion) = create_classifier_and_diffusion(**args_to_dict(args, classifier_and_diffusion_defaults().keys()))
    # model=model.to('cuda')
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        logger.log(f'loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step')
        print(args.resume_checkpoint)
        model.load(args.resume_checkpoint)
        # if (dist.get_rank() == 0):
        #     logger.log(f'loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step')
        #     model.load_parameters(dist_util.load_parameters(args.resume_checkpoint, map_location=dist_util.dev()))
    # dist_util.sync_params(model.parameters())
    target_step = args.run_iterations + resume_step
    args.tot_iterations = max(args.tot_iterations, target_step)
    logger.log(f'total step: {args.tot_iterations}')
    logger.log(f'target step of this run: {target_step}')
    logger.log(f'saving interval: per {args.save_interval} steps')
    logger.log(f'learning rate: {args.lr}')
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0)
    #model = DDP(model, device_ids=[dist_util.dev()], output_device=dist_util.dev(), broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=False)
    #xjh:DDP
    logger.log('creating data loader...')
    data = load_data(data_dir=args.data_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=True, random_crop=True)
    if args.val_data_dir:
        val_data = load_data(data_dir=args.val_data_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=True)
    else:
        val_data = None
    logger.log(f'creating optimizer...')
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    # if args.resume_checkpoint:
    #     opt_checkpoint = bf.join(bf.dirname(args.resume_checkpoint), f'opt{resume_step:06}.pkl')
    #     logger.log(f'loading optimizer state from checkpoint: {opt_checkpoint}')
    #     opt.load_state_dict(opt_checkpoint)
    #     # opt.load_parameters(dist_util.load_parameters(opt_checkpoint, map_location=dist_util.dev()))
    logger.log('training classifier model...')

    def forward_backward_log(data_loader, opt, prefix='train'):
        (batch, extra) = next(data_loader)
        labels = extra['y']
        # totalloss=0
        # batch = batch.to(dist_util.dev())
        if args.noised:
            (t, _) = schedule_sampler.sample(batch.shape[0], 'cuda')
            batch = diffusion.q_sample(batch, t)
        else:
            t = jt.zeros(batch.shape[0], dtype=jt.long, device='cuda')
        for (i, (sub_batch, sub_labels, sub_t)) in enumerate(split_microbatches(args.microbatch, batch, labels, t)):
            logits = model(sub_batch, timesteps=sub_t)
            loss = nn.cross_entropy_loss(logits, sub_labels, reduction='none')
            # losses = {}
            # losses[f'{prefix}_loss'] = loss.detach()
            # losses[f'{prefix}_acc@1'] = compute_top_k(logits, sub_labels, k=1, reduction='none')
            # losses[f'{prefix}_acc@5'] = compute_top_k(logits, sub_labels, k=5, reduction='none')
            # del losses
            loss = loss.mean()
            totalloss=loss
            acc1 = compute_top_k(logits, sub_labels, k=1, reduction='mean')
            acc5 = compute_top_k(logits, sub_labels, k=5, reduction='mean')
            if loss.requires_grad:
                # if (i == 0):
                    # mp_trainer.zero_grad()
                opt.backward(((loss * len(sub_batch)) / len(batch)))
        return totalloss, acc1, acc5
    # for step in range((args.iterations - resume_step)):
    for step in range((args.run_iterations)):
        logger.logkv('step', (step + resume_step))
        logger.logkv('samples', ((((step + resume_step) + 1) * args.batch_size) ))
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, ((step + resume_step) / args.tot_iterations))
        opt.zero_grad()
        loss, acc1, acc5 = forward_backward_log(data,opt=opt)
        opt.step()
        logger.logkv('loss', loss)
        logger.logkv('acc1', acc1)
        logger.logkv('acc5', acc5)
        # mp_trainer.optimize(opt)
        if ((val_data is not None) and (not (step % args.eval_interval))):
            with jt.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data,None, prefix='val')
                    model.train()
        if (not (step % args.log_interval)):
            logger.dumpkvs()
        if (step and (not ((step + resume_step) % args.save_interval))):
            logger.log('saving model...')
            logf = open('last_model_num.txt','w')
            model_num = f'{step + resume_step:06d}'
            logf.write(model_num)
            logf.close()
            save_model(mp_trainer, opt, (step + resume_step))
    # if (dist.get_rank() == 0):
    logger.log('saving model...')
    logf = open('last_model_num.txt','w')
    model_num = f'{target_step:06d}'
    logf.write(model_num)
    logf.close()
    save_model(mp_trainer, opt, target_step)
    # dist.barrier()
    logger.log('training complete')

def set_annealed_lr(opt, base_lr, frac_done):
    lr = (base_lr * (1 - frac_done))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def save_model(mp_trainer, opt, step):
    # if (dist.get_rank() == 0):
    jt.save(mp_trainer.master_params_to_state_dict(mp_trainer.master_params), os.path.join(logger.get_dir(), f'model{step:06d}.pkl'))
    logger.log(f'model saved: model{step:06d}.pkl')
    # jt.save(opt.state_dict(), os.path.join(logger.get_dir(), f'opt{step:06d}.pkl'))

def compute_top_k(logits, labels, k, reduction='mean'):
    (_, top_ks) = jt.topk(logits, k, dim=(- 1))
    if (reduction == 'mean'):
        return (top_ks == labels[:, None]).float().sum(dim=(- 1)).mean().item()
    elif (reduction == 'none'):
        return (top_ks == labels[:, None]).float().sum(dim=(- 1))

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if ((microbatch == (- 1)) or (microbatch >= bs)):
        (yield tuple(args))
    else:
        for i in range(0, bs, microbatch):
            (yield tuple(((x[i:(i + microbatch)] if (x is not None) else None) for x in args)))

def create_argparser():
    defaults = dict(data_dir='', val_data_dir='', noised=True, tot_iterations=150000, run_iterations=150000, lr=0.0003, weight_decay=0.0, anneal_lr=False, batch_size=4, microbatch=(- 1), schedule_sampler='uniform', resume_checkpoint='', log_interval=10, eval_interval=5, save_interval=10000)
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
if (__name__ == '__main__'):
    main()

