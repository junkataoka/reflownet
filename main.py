# import libraries
from cgi import print_directory
import os
from time import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from data_module import TSDataModule
from lstm_ae import EncoderDecoderConvLSTM
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from torchvision import transforms
from mmd import MMD
import seaborn as sns
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import pandas  as pd
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--n_hidden_dim', type=int, default=8, help='number of hidden dim for ConvLSTM layers')
parser.add_argument('--log_images', action='store_true', help='Whether to log images')
parser.add_argument('--is_distributed', action='store_true', help='Whether to used distributeds dataloader')

parser.add_argument('--root', type=str, default="./dataset")
parser.add_argument('--src_input_file', type=str, default="source_input.pt")
parser.add_argument('--src_target_file', type=str, default="source_target.pt")
parser.add_argument('--tar_input_file', type=str, default="target_input.pt")
parser.add_argument('--tar_target_file', type=str, default="sp_target_target.pt")
parser.add_argument('--time_steps', type=int, default=7)

parser.add_argument('--model_path', type=str, default="checkpoints/lstm_ac.ckpt")
parser.add_argument('--out_model_path', type=str, default="checkpoints/lstm_ac.ckpt")
parser.add_argument('--retrain', action='store_true', help='Whether to retrain the model or not')
parser.add_argument('--neptune_logger', action='store_true', help='Whether to use neptune.ai logger')
parser.add_argument('--api_key', type=str,
                    default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")

parser.add_argument('--run_type', type=str, default="train")
parser.add_argument('--source_only', action="store_true", help="Wether training model using source only or not")


opt = parser.parse_args()
print(opt)

# SEED = 1234
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

class OvenLightningModule(pl.LightningModule):

    def __init__(self, opt):

        super(OvenLightningModule, self).__init__()

        self.save_hyperparameters()
        self.opt = opt
        self.normalize = False
        self.model1 = EncoderDecoderConvLSTM(nf=self.opt.n_hidden_dim, in_chan=4)
        self.model2 = EncoderDecoderConvLSTM(nf=self.opt.n_hidden_dim, in_chan=4)
        self.log_images = self.opt.log_images
        self.criterion = nn.MSELoss()
        self.dcl_criterion = nn.NLLLoss()
        self.pdist = nn.PairwiseDistance(p=2)
        self.batch_size = self.opt.batch_size
        self.time_steps = self.opt.time_steps
        self.epoch = 0
        self.step = 0
        alpha = torch.FloatTensor(6).fill_(1)
        self.register_parameter(name="alpha", param=nn.Parameter(data=alpha, requires_grad=True))
        beta = torch.FloatTensor(6).fill_(0)
        self.register_parameter(name="beta", param=nn.Parameter(data=beta, requires_grad=True))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 100 , gamma = 0.5)

        # return {"optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler":sch,
        #             "monitor": "loss"}}
        return optimizer

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path)
        self.model1.load_state_dict(checkpoint["model1"])
        self.model2.load_state_dict(checkpoint["model1"])

        print('Model Created!')


    def forward(self, x, model, source_only):

        output, f1, f2 = model(x, future_step=self.time_steps, source_only=self.opt.source_only)

        return output, f1, f2

    def RegLoss(self, model1, model2, alpha, beta):
        loss = 0
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if 'encoder_1' in name1:
                loss += torch.norm(alpha[0] * param1 + beta[0] - param2)

            if 'encoder_2' in name1:
                loss += torch.norm(alpha[1] * param1 + beta[1] - param2)

            if 'decoder_1' in name1:
                loss += torch.norm(alpha[2] * param1 + beta[2] - param2)

            if 'decoder_2' in name1:
                loss += torch.norm(alpha[3] * param1 + beta[3] - param2)

            elif 'decoder_CNN' in name1:
                loss += torch.norm(alpha[4] * param1 + beta[4] - param2)

        return loss

    def training_step(self, batch, batch_idx):

        src_batch = batch[0]
        tar_batch = batch[1]
        src_x, src_y = src_batch
        tar_x, tar_y = tar_batch

        src_y_hat, src_ln_1, src_ln_2 = self.forward(src_x, self.model1, self.opt.source_only)
        tar_y_hat, tar_ln_1, tar_ln_2 = self.forward(tar_x, self.model2, self.opt.source_only)

        src_loss = self.criterion(src_y_hat, torch.log(src_y))
        tar_loss = self.criterion(tar_y_hat, torch.log(tar_y))

        dst_loss1 = torch.mean(self.pdist(src_y_hat, tar_y_hat))
        dst_loss2 = torch.mean(self.pdist(src_ln_1, tar_ln_1))
        dst_loss3 = torch.mean(self.pdist(src_ln_2, tar_ln_2))

        # mmd_loss3 = MMD(src_ln_2, tar_ln_2, kernel="multiscale")

        reg_loss = self.RegLoss(self.model1, self.model2, self.alpha, self.beta)

        avg_diff_src_src = torch.mean(torch.abs(src_y_hat - torch.log(src_y)))
        avg_diff_tar_tar = torch.mean(torch.abs(tar_y_hat - torch.log(tar_y)))

        self.log("src_loss", src_loss.item(), on_step=True, on_epoch=False)
        self.log("tar_loss", tar_loss.item(), on_step=True, on_epoch=False)
        # self.log("mmd_loss1", mmd_loss1.item(), on_step=True, on_epoch=True)
        # self.log("mmd_loss2", mmd_loss2.item(), on_step=True, on_epoch=True)
        # self.log("mmd_loss3", mmd_loss3.item(), on_step=True, on_epoch=True)
        self.log("dst_loss1", dst_loss1, on_step=True, on_epoch=False)
        self.log("dst_loss2", dst_loss2, on_step=True, on_epoch=False)

        self.log("alpha", self.alpha[0], on_step=True, on_epoch=False)
        self.log("beta", self.beta[0], on_step=True, on_epoch=False)

        self.log("avg_diff_src_src", avg_diff_src_src.item(), on_step=True, on_epoch=False)

        self.log("avg_diff_tar_tar", avg_diff_tar_tar.item(), on_step=True, on_epoch=False)

        if self.opt.source_only:
            loss = src_loss
            return loss

        else:
            loss = src_loss + dst_loss1 + dst_loss2 + dst_loss3 + reg_loss
            return {"loss": loss, "src_y_hat": src_y_hat, "tar_y_hat": tar_y_hat, "src_ln_1": src_ln_1, "tar_ln_1": tar_ln_1, "src_ln_2": src_ln_2, "tar_ln_2": tar_ln_2}

    def training_epoch_end(self, training_step_outputs):

        if not self.opt.source_only:

            src_ln_1 = torch.cat([i["src_ln_1"] for i in training_step_outputs], axis=0)
            src_ln_2 = torch.cat([i["src_ln_2"] for i in training_step_outputs], axis=0)
            tar_ln_1 = torch.cat([i["tar_ln_1"] for i in training_step_outputs], axis=0)
            tar_ln_2 = torch.cat([i["tar_ln_2"] for i in training_step_outputs], axis=0)
            loss = torch.cat([i["loss"].view(1, -1) for i in training_step_outputs], axis=0)
            torch.save(src_ln_1, f"feature/src_ln_1_epoch{self.epoch}.pt")
            torch.save(src_ln_2, f"feature/src_ln_2_epoch{self.epoch}.pt")
            torch.save(tar_ln_1, f"feature/tar_ln_1_epoch{self.epoch}.pt")
            torch.save(tar_ln_2, f"feature/tar_ln_2_epoch{self.epoch}.pt")
            torch.save(loss, f"feature/loss_epoch{self.epoch}.pt")

            self.epoch += 1

def test_trainer():
    model =OvenLightningModule(opt).cuda()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.test_dataloader()
    model.load_model()
    model.eval()
    for i, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            if opt.source_only:
                predictions, _, _ = model(inp, model.model1, opt.source_only)
            else:
                predictions, _, _ = model(inp, model.model2, opt.source_only)

        predictions = torch.exp(predictions)
        plt.figure(figsize=(4, 3))
        plt.plot([i+1 for i in range(target.shape[1])], target.view(-1).cpu(), ".-", label="Target")
        plt.plot([i+1 for i in range(target.shape[1])], predictions.view(-1).cpu(), ".-", label="Prediction")
        plt.ylabel("Temperature")
        plt.xlabel("Area")
        plt.legend()
        model_name = opt.out_model_path.split("/")[-1]
        model_name = model_name.split(".")[0]
        plt.savefig(f"Figure/sample{i+1}_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close("all")
        err = torch.norm(target-predictions)
        print(f"{model_name} predictions:", predictions.view(-1))
        print(f"{model_name} target:", target.view(-1))
        print(f"{model_name} error:", err)

def val_best_recipes():
    model =OvenLightningModule(opt).cuda()

    model.eval()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.val_dataloader()
    avg_diff = []
    for idx, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            predictions, _ = model(inp)
            avg_diff.append(torch.mean(torch.abs(target - predictions)))
    arr = torch.stack(avg_diff)
    torch.save(arr, f"temp/{opt.tar_input_file}_err.pt")

def run_trainer():
    model =OvenLightningModule(opt).cuda()
    if not opt.source_only:
        oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, opt.batch_size)
    else:
        oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.src_input_file, opt.src_target_file, opt.batch_size)
    if opt.neptune_logger:
        logger = NeptuneLogger(
                api_key=opt.api_key,
                    project='junkataoka/heatmap'
                                )
    else:
        logger = None


    trainer = Trainer(max_epochs=opt.epochs,
                        gpus=opt.n_gpus,
                        logger=logger,
                        accelerator='cuda',
                        num_nodes=opt.num_nodes,
                        # gradient_clip_val=0.5,
                        # multiple_trainloader_mode="min_size"
                      )

    if opt.retrain:
        model.load_model()

    trainer.fit(model, datamodule=oven_data)
    torch.save({"model1":model.model1.state_dict(),
                "model2":model.model2.state_dict()
                }, opt.out_model_path)


def create_input(geom_root, heatmap_root, geom_num, seq_len):

    die_path = f"M{geom_num}_DIE.csv"
    pcb_path = f"M{geom_num}_PCB.csv"
    trate_path = f"M{geom_num}_Substrate.csv"
    out = np.empty((1, len(seq_len), 4, 50, 50))
    die_img = np.genfromtxt(os.path.join(geom_root, die_path), delimiter=",")
    pcb_img = np.genfromtxt(os.path.join(geom_root, pcb_path), delimiter=",")
    trace_img = np.genfromtxt(os.path.join(geom_root, trate_path), delimiter=",")
    for i in range(len(seq_len)):
        heatmap_img = np.genfromtxt(os.path.join(heatmap_root, trate_path), delimiter=",")
        heatmap_path = f"IMG_{geom_num}_1_{i+1}.csv"
        arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                        trace_img[np.newaxis, ...], heatmap_img[np.newaxis, ...]], axis=0)
        out[0, i,  :, :, :] = arr

    inp = torch.tensor(out)
    return inp


def bayesian_ops():
    model =OvenLightningModule(opt).cuda()
    model.load_model()
    # model.load_state_dict(torch.load(opt.model_path, map_location='cuda:0'), strict=False)
    model.eval()

    src_mean = torch.load("dataset/source_mean.pt", map_location="cuda:0")
    src_sd = torch.load("dataset/source_sd.pt", map_location="cuda:0")
    plt.figure(figsize=(4, 3))
    geom_num = 1

    def GetSlope(val1, val2, t1, t2):
        slope = (val2 - val1) / (t2 - t1)
        return slope

    def black_box_function(r1, r2, r3, r4, r5, r6, r7):

        steps = [33, 66, 99, 132, 171, 204, 214, 224,
                    234, 244, 254, 264, 274, 284, 294]
        recipes = [r1, r2, r3, r4, r5, r6, r7]
        inp = create_input(geom_num, recipes)
        inp = inp.cuda()
        inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
        inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
        loss = 0

        target_temp = torch.cuda.FloatTensor( [57, 90, 123, 156, 195, 228, 233, 238, 242, 247, 242, 238, 233, 228, 223]).unsqueeze(1)

        with torch.no_grad():
            pred, _, _, _, _ = model(inp_normalized, model.model1)

        pred_temp = pred[:,:,:, :12, :17].mean((-1,-2)).squeeze(0)
        loss += (target_temp - pred_temp).pow(2).sum() / target_temp.size(0)
        # peak_err = max((max(pred_temp) - max(recipes), 0.0))
        # loss += float(peak_err)


        loss = np.array(loss.cpu())


        return -loss

    # pbounds = {"r1": (100, 120), "r2": (120, 170), "r3": (170, 190),
    #            "r4":(190, 210), "r5":(240, 400), "r6": (270, 400), "r7": (290, 400)}


    pbounds = {"r1": (100, 280), "r2": (100, 280), "r3": (100, 280),
               "r4":(100, 280), "r5":(100, 280), "r6": (100, 280), "r7": (100, 280)}

#recipes = [120, 150, 180, 210, 240, 280, 300]

    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds,
                                     random_state=1)
# logger = JSONLogger(path="./bo_logs.json")
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        acq="ei",
        xi=0.1,
        init_points=100,
        n_iter=100
    )
    print(optimizer.max)
    r1, r2, r3, r4, r5, r6, r7 = optimizer.max["params"]["r1"], \
                                 optimizer.max["params"]["r2"], \
                                 optimizer.max["params"]["r3"], \
                                 optimizer.max["params"]["r4"], \
                                 optimizer.max["params"]["r5"], \
                                 optimizer.max["params"]["r6"], \
                                 optimizer.max["params"]["r7"],

    recipes = [r1, r2, r3, r4, r5, r6, r7]

    recipe_list = { "Experiment1":[105, 130, 160, 190, 230, 270, 290],
               "Experiment2":[110, 140, 170, 200, 240, 280, 290],
                "Experiment3":[120, 150, 180, 220, 260, 280, 300]}

    recipe_list["BO"] = recipes
    df = pd.DataFrame()
    for k, v in recipe_list.items():
        print(f"", black_box_function(v[0], v[1], v[2], v[3], v[4], v[5], v[6]))

        inp = create_input(geom_num, v)
        inp = inp.cuda()
        inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
        inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
        steps = [33, 66, 99, 132, 171, 204, 214, 224,
                    234, 244, 254, 264, 274, 284, 294]

        with torch.no_grad():
            pred, _, _, _, _ = model(inp_normalized, model.model1)

        y = pred[0, :, :, :5, :5].mean((-1, -2)).cpu()
        df[k] = pd.Series(y.squeeze(1))
        plt.plot(steps, y, ".-", label=f"M{geom_num}_{k}")

    target_temp = torch.cuda.FloatTensor( [57, 90, 123, 156, 195, 228, 233, 238, 242, 247, 242, 238, 233, 228, 223]).unsqueeze(1).cpu()

    plt.plot(steps, target_temp, ".-", label="Target")

    df.to_csv("temp/bo_profile.csv")
    plt.ylabel("Temperature")
    plt.xlabel("Time Step")
    plt.legend()
    plt.savefig(f"Figure/BO_pred.png", dpi=300, bbox_inches='tight')
    plt.close("all")


if __name__ == '__main__':
    if opt.run_type=="test":
        test_trainer()
    elif opt.run_type=="validation":
        val_best_recipes()
    elif opt.run_type=="bo":
        bayesian_ops()
    else:
        run_trainer()
