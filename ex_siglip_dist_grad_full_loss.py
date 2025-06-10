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
    # model.weight = torch.nn.Parameter(0.3*torch.eyes(2, 1).to(torch.device(f'cuda:{rank}')))
    model.weight = torch.nn.Parameter(torch.tensor([[0.1], [0.2]], dtype=torch.float32).to(torch.device(f'cuda:{rank}')))
    if wrap_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        input = torch.tensor([[1.]], device=f'cuda:{rank}')
    elif rank == 1:
        input = torch.tensor([[2.]], device=f'cuda:{rank}')
    output = model(input)
    print(output.shape)


    if gather_grad:
        outputs = torch.cat(torch.distributed.nn.all_gather(output), dim=0)
        loss = outputs[0][0]*outputs[0][1] + outputs[1][0]*outputs[1][1] \
             + outputs[0][0]*outputs[1][1] + outputs[1][0]*outputs[0][1]
    else:
        with torch.no_grad():
            outputs = [torch.zeros_like(output) for _ in range(world_size)]
            torch.distributed.all_gather(outputs, output)
        outputs[rank] = output
        loss = outputs[0][0, 0]*outputs[0][0, 1] + outputs[1][0, 0]*outputs[1][0, 1] \
             + outputs[0][0, 0]*outputs[1][0, 1] + outputs[1][0, 0]*outputs[0][0, 1]

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