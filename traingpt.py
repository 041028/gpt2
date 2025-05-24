import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import model, utils
from model import GPT
from trainer import Trainer
from utils import set_seed, setup_logging, CfgNode as CN

def get_config():
    C = CN()
    # 系统配置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './outs/chargpt'
    # 数据配置
    C.data = CharDataset.get_default_config()
    # 模型配置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2'
    # 训练器配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-4
    return C

class CharDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C
    def __init__(self, config, data):
        self.config = config
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f'数据包含 {data_size} 个字符，{vocab_size} 个唯一字符。')

        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 字符到索引
        self.itos = {i: ch for i, ch in enumerate(chars)}  # 索引到字符
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.config.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

if __name__ == '__main__':
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)
    text = open('fiction.txt', 'r', encoding='utf-8').read()
    train_dataset = CharDataset(config.data, text)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset)

    checkpoint_path = os.path.join(config.system.work_dir, "model.pt")
    if os.path.exists(checkpoint_path):
        print(f"从检查点 {checkpoint_path} 加载模型...")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("优化器状态已恢复")

        if 'iter_num' in checkpoint:
            trainer.iter_num = checkpoint['iter_num']
            print(f"从迭代 {trainer.iter_num} 恢复训练")
    else:
        print("未找到检查点，从头开始训练")

    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(
                f"迭代耗时 {trainer.iter_dt * 1000:.2f}ms; 迭代 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                context = "我们家"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(
                    trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])

            checkpoint = {
                'iter_num': trainer.iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
            }
            os.makedirs(config.system.work_dir, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()