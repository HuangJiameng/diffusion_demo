import os
import torch
import matplotlib.pyplot as plt
from time import time
from utils import sample_and_plot
from models import DiT, SimpleMLPAdaLN

class Trainer:
    def __init__(self, model, diffusion, optimizer, device, X, Y, 
                 output_dir, epochs, log_interval, save_interval):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.X = X
        self.Y = Y
        self.output_dir = output_dir
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_interval = save_interval

    def train(self):
        tot_len = len(self.X)
        batch_size = 4096
        tot_len = (tot_len // batch_size) * batch_size
        self.X = self.X[:tot_len]
        self.Y = self.Y[:tot_len]

        train_steps = 0    
        running_loss = 0
        log_steps = 0
        start_time = time()

        for epoch in range(self.epochs):
            for i in range(0, tot_len, batch_size):  
                x = self.X[i:i+batch_size]  
                y = self.Y[i:i+batch_size]
                x = torch.tensor(x, dtype=torch.float)
                if type(self.model) == DiT:
                    x = x.unsqueeze(2).unsqueeze(3)
                    y = torch.tensor(y, dtype=torch.int)
                elif type(self.model) == SimpleMLPAdaLN:
                    y = torch.tensor(y, dtype=torch.float)
                    y = y.unsqueeze(1)
                else:
                    raise NotImplementedError
                x = x.to(self.device)
                y = y.to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
                model_kwargs = dict(y=y)

                loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)

                loss = loss_dict["loss"].mean()
                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_steps += 1
                log_steps += 1

            if epoch % self.log_interval == 0:
                # print(loss.item())
                end_time = time()
                sec_per_step = (end_time - start_time) / self.log_interval
                avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Sec per Epoch: {sec_per_step:.4f}\n")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if epoch % self.save_interval == 0:
                print(f"Sampling on Epoch {epoch}...")
                sample_and_plot(model = self.model, 
                                diffusion = self.diffusion,
                                output_dir = self.output_dir,
                                comment = f"Epoch{epoch}-")
                model_save_path = f"{self.output_dir}/model_{epoch}.pt"
                torch.save(self.model.state_dict(), model_save_path)

        self.model.eval()

        sample_and_plot(model = self.model, 
                        diffusion = self.diffusion,
                        output_dir = self.output_dir,
                        comment = f"Epoch{self.epochs}-")
        model_save_path = f"{self.output_dir}/model_final.pt"
        torch.save(self.model.state_dict(), model_save_path)
