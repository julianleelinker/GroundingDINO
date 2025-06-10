import os
import torch
import torch.multiprocessing as mp
import torch.distributed.nn
import torch.distributed as dist
import fire

os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
os.environ['MASTER_PORT'] = '12355'


def run(rank, world_size, gather_grad, wrap_ddp):
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = torch.nn.Linear(1, 2, bias=False).to(torch.device(f'cuda:{rank}'))
    model.weight = torch.nn.Parameter(torch.tensor([[0.1], [0.2]], dtype=torch.float32).to(torch.device(f'cuda:{rank}')))
    if wrap_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        input = torch.tensor([[1.]], device=f'cuda:{rank}')
    elif rank == 1:
        input = torch.tensor([[2.]], device=f'cuda:{rank}')
    output = model(input)
    image_embed, text_embed = output[:, 0], output[:, 1]

    if gather_grad:
        text_embed_all = torch.distributed.nn.all_gather(text_embed)
        loss = image_embed*text_embed
        for i in range(world_size):
            if i == rank:
                continue
            loss += image_embed*text_embed_all[i]
    else:
        with torch.no_grad():
            text_embed_all = [torch.zeros_like(text_embed) for _ in range(world_size)]
            torch.distributed.all_gather(text_embed_all, text_embed)
        text_embed_all[rank] = text_embed 
        loss = image_embed*text_embed
        for i in range(world_size):
            if i == rank:
                continue
            loss += image_embed*text_embed_all[i]

    loss.backward()

    if wrap_ddp:
        print(f"{rank=}: {model.module.weight.grad}")
    else:
        print(f"{rank=}: {model.weight.grad}")

    torch.distributed.destroy_process_group()


def main(gather_grad=False, wrap_ddp=False):
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, gather_grad, wrap_ddp), nprocs=world_size, join=True)


if __name__ == '__main__':
    fire.Fire(main)