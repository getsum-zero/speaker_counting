import os
import yaml
import argparse
from torch import nn
import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from osdc.utils import BinaryMeter, MultiMeter
from online_data import OnlineFeats
from transformers import AutoProcessor, WavLMModel
from Suconloss import SupConLoss
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="OSDC on AMI")
parser.add_argument("--conf_file", type=str, default="../conf/train.yml")
parser.add_argument("--log_dir", type=str, default="../exp/tcn")
parser.add_argument("--gpus", type=str, default="0")


class PlainModel(nn.Module):

    def __init__(self, masker, hparams):

        super(PlainModel, self).__init__()
        self.model = masker
        self.configs = hparams
        self.processor = AutoProcessor.from_pretrained("wavlm-libri-clean-100h-base-plus")
        self.wavlm = WavLMModel.from_pretrained("wavlm-libri-clean-100h-base-plus")
        self.linear = nn.Linear(in_features=249, out_features=500)
        # for param in self.wavlm.parameters():
        #     param.requires_grad = False

    def forward(self, tf_rep):
        if self.configs["feats"]["type"] == "wavlm":
            res = []
            for item in tf_rep:
                inputs = self.processor(item, sampling_rate=self.configs["data"]["fs"], return_tensors="pt")
                inputs = {k: v.to(torch.device('cuda')) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                inputs["output_hidden_states"] = True 
                item = self.wavlm(**inputs)
                # print(item.extract_features.shape, item.last_hidden_state.shape, item.hidden_states[0].shape, len(item.hidden_states))
                if self.configs["feats"]["layer"] == -1:
                    res.append(item.last_hidden_state.transpose(1, 2))
                else:
                    res.append(item.hidden_states[self.configs["feats"]["layer"]].transpose(1, 2))
            tf_rep = torch.cat(res, dim=0)
            batch_size, seq_len, _ = tf_rep.shape
            flattened_tensor = tf_rep.view(batch_size * seq_len, -1)
            output_tensor = self.linear(flattened_tensor)
            tf_rep = output_tensor.view(batch_size, seq_len, -1)
        feats, mask = self.model(tf_rep)

        return feats, mask


class OSDC_AMI(pl.LightningModule):

    '''
    Plain cycle routine we have 2 discriminators and two generators
    '''

    def __init__(self, hparams):
        super(OSDC_AMI, self).__init__()
        self.configs = hparams # avoid pytorch-lightning hparams logging

        if not self.configs["augmentation"]["probs"]:
            # these are determined by looking at tensoboard statistics: used to fight imbalancing
            cross = nn.CrossEntropyLoss(torch.Tensor([1.74, 1.0, 11.98]).cuda(), reduction="none")
        else:
            cross = nn.CrossEntropyLoss(torch.Tensor([3.54, 1, 9.72]).cuda(), reduction="none")


        self.loss_ce = lambda x, y : cross(x, y) #+ 0.1*dice(1-x, 1-y) # flip positive for focal loss

        config = self.configs["loss"]["contrast"]
        self.loss_con = SupConLoss(temperature=config["temperature"], 
                                    contrast_mode=config["contrast_mode"],
                                    base_temperature=config["base_temperature"])
        self.gamma_loss_con = config["gamma"]


        self.train_count_metrics = MultiMeter()
        self.train_vad_metrics = BinaryMeter()
        self.train_osd_metrics = BinaryMeter()
        self.val_count_metrics = MultiMeter()
        self.val_vad_metrics = BinaryMeter()
        self.val_osd_metrics = BinaryMeter()

        from osdc.models.tcn import TCN
        self.model = PlainModel(TCN(768, 3, 1, 5, 3, 64, 128), hparams)



    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):

        feats, label, mask = batch
        assert torch.max(label) < 3
        feats, preds = self.model(feats)

        loss_ce = self.loss_ce(preds, label)
        loss_ce = (loss_ce * mask.detach()).mean()
        
        feats = feats.transpose(1,2)
        feats = feats.reshape(-1, 1, feats.shape[2])
        feats = F.normalize(feats, dim=-1, p=2)
        loss_con = self.loss_con(feats, labels=label)
        loss = loss_con  # loss_ce + self.gamma_loss_con * loss_con 
        


        preds = torch.softmax(preds, 1)
        self.train_count_metrics.update(torch.argmax(preds, 1), label)
        self.train_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        #self.train_osd_metrics.update(torch.argmax(torch.cat((preds[:, :2], torch.sum(preds[:, 2:], 1, keepdim=True)),1), 1), torch.clamp(label, 0, 2))
        self.train_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)


        tensorboard_logs = {'train_batch_loss': loss,
                            'train_tp_count': self.train_count_metrics.get_tp(),
                            'train_tn_count': self.train_count_metrics.get_tn(),
                            'train_fp_count': self.train_count_metrics.get_fp(),
                            'train_fn_count': self.train_count_metrics.get_fn(),
                            'train_prec_count': self.train_count_metrics.get_precision(),
                            'train_rec_count': self.train_count_metrics.get_recall(),
                            'train_prec_vad': self.train_vad_metrics.get_precision(),
                            'train_rec_vad': self.train_vad_metrics.get_recall(),
                            'train_fa_vad': self.train_vad_metrics.get_fa(),
                            'train_miss_vad': self.train_vad_metrics.get_miss(),
                            'train_der_vad': self.train_vad_metrics.get_der(),
                            'train_prec_osd': self.train_osd_metrics.get_precision(),
                            'train_rec_osd': self.train_osd_metrics.get_recall(),
                            'train_fa_osd': self.train_osd_metrics.get_fa(),
                            'train_miss_osd': self.train_osd_metrics.get_miss(),
                            'train_der_osd': self.train_osd_metrics.get_der(),
                            'train_tot_silence': self.train_count_metrics.get_positive_examples_class(0),
                            'train_tot_1spk': self.train_count_metrics.get_positive_examples_class(1),
                            'train_tot_2spk': self.train_count_metrics.get_positive_examples_class(2),
                            'train_tot_3spk': self.train_count_metrics.get_positive_examples_class(3),
                            'train_tot_4spk': self.train_count_metrics.get_positive_examples_class(4)
                            }

        output = OrderedDict({
                'loss': loss,
                'log': tensorboard_logs
            })
        return output

    def validation_step(self, batch, batch_indx):

        feats, label, _ = batch
        feats, preds = self.model(feats)
        loss_ce = self.loss_ce(preds, label).mean()

        feats = feats.transpose(1,2)
        feats = feats.reshape(-1, 1, feats.shape[2])
        feats = F.normalize(feats, dim=-1, p=2)
        loss_con = self.loss_con(feats, labels=label.flatten())
        loss = loss_con #loss_ce + self.gamma_loss_con * loss_con
        


        preds = torch.softmax(preds, 1)
        self.val_count_metrics.update(torch.argmax(preds, 1), label)
        self.val_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        #self.val_osd_metrics.update(torch.argmax(torch.cat((preds[:, :2], torch.sum(preds[:, 2:], 1, keepdim=True)),1),1), torch.clamp(label, 0, 2))
        self.val_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)
        tqdm_dict = {'val_loss': loss_ce}


        avg_loss = loss.mean()
        tqdm_dict = {'val_loss': avg_loss}
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_tp_count': self.val_count_metrics.get_tp(),
                            'val_tn_count': self.val_count_metrics.get_tn(),
                            'val_fp_count': self.val_count_metrics.get_fp(),
                            'val_fn_count': self.val_count_metrics.get_fn(),
                            'val_prec_count': self.val_count_metrics.get_precision(),
                            'val_rec_count': self.val_count_metrics.get_recall(),
                            'val_prec_vad': self.val_vad_metrics.get_precision(),
                            'val_rec_vad': self.val_vad_metrics.get_recall(),
                            'val_fa_vad': self.val_vad_metrics.get_fa(),
                            'val_miss_vad': self.val_vad_metrics.get_miss(),
                            'val_der_vad': self.val_vad_metrics.get_der(),
                            'val_prec_osd': self.val_osd_metrics.get_precision(),
                            'val_rec_osd': self.val_osd_metrics.get_recall(),
                            'val_fa_osd': self.val_osd_metrics.get_fa(),
                            'val_miss_osd': self.val_osd_metrics.get_miss(),
                            'val_der_osd': self.val_osd_metrics.get_der(),
                            }
        self.log("val_loss", avg_loss)
        self.train_count_metrics.reset()
        self.train_vad_metrics.reset()
        self.train_osd_metrics.reset()
        self.val_count_metrics.reset()
        self.val_vad_metrics.reset()
        self.val_osd_metrics.reset()
        output = OrderedDict({
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': tensorboard_logs
        })

        return output

    def configure_optimizers(self):

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    self.configs["opt"]["lr"], weight_decay=self.configs["opt"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


    def train_dataloader(self):
        dataset = OnlineFeats(self.configs["data"]["chime6_root"], self.configs["data"]["label_train"],
                              self.configs, probs=self.configs["augmentation"]["probs"], segment=self.configs["data"]["segment"])
        dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
                                shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)
        return dataloader


    def val_dataloader(self):

        dataset = OnlineFeats(self.configs["data"]["chime6_root"], self.configs["data"]["label_val"],
                                    self.configs, segment=self.configs["data"]["segment"])
        dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
                                shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)

        return dataloader

if __name__ == "__main__":

    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)

    # test if compatible with lightning
    confs.update(args.__dict__)
    a = OSDC_AMI(confs)

    if confs["training"]["resume_from"]:
        print(confs["training"]["resume_from"])
        state_dict = torch.load(confs["training"]["resume_from"],
                            map_location='cpu')["state_dict"]
        a.load_state_dict(state_dict)

    checkpoint_dir = os.path.join(confs["log_dir"], 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 filename='ft-only-con-{epoch}-{step}-layer' + str(confs["feats"]["layer"]),
                                 mode='min',  verbose=True, save_top_k=1)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    with open(os.path.join(confs["log_dir"], "confs.yml"), "w") as f:
        yaml.dump(confs, f)

    logger = TensorBoardLogger(os.path.dirname(confs["log_dir"]), confs["log_dir"].split("/")[-1])

    trainer = pl.Trainer(max_epochs=confs["training"]["n_epochs"],
                         accumulate_grad_batches=confs["training"]["accumulate_batches"], callbacks=[checkpoint, early_stop_callback],
                         # limit_train_batches = 1, limit_val_batches = 3,
                         logger = logger,
                         gradient_clip_val=confs["training"]["gradient_clip"],
                         accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                         strategy = "auto",
                         devices = -1,
                         # resume_from_checkpoint=confs["training"]["resume_from"]
                         )
    trainer.fit(a)