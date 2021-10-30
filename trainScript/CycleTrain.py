import argparse
import itertools

import torch
from torch.utils.data import DataLoader
from torchvision import utils
import os
from dataset import ImageDataSet
from dataset import TestImageDataSet
from networks.Pix2PixHD.Generator import GlobalGenerator
from networks.Pix2PixHD.Discriminator import MultiscaleDiscriminator
from utils import get_norm_layer as get_norm_layer
from utils import weights_init as weights_init
from Loss import GANLoss
from Loss import VGGLoss
from Loss import SemiLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sketch')
    parser.add_argument('--data_root', type=str, default='/data/haokang/Semi-Supervisied-DataSet/sketch')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=256)

    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--save_img_epoch', type=int, default=1)
    parser.add_argument('--save_model_epoch', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.5)
    parser.add_argument('--lambda_feat', type=float, default=10.0)

    parser.add_argument('--distributed', type=bool, default=False)

    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsampling_G', type=int, default=4)
    parser.add_argument('--n_blocks_G', type=int, default=9)

    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--use_sigmoid', type=bool, default=False)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--getIntermFeat', type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = 'cuda:0'

    assert args.name
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(os.path.join('./result', args.name)):
        os.mkdir(os.path.join('./result', args.name))
        os.mkdir(os.path.join('./result', args.name, 'imgs'))
        os.mkdir(os.path.join('./result', args.name, 'checkpoints'))
        os.mkdir(os.path.join('./result', args.name, 'loss_txt'))
        os.mkdir(os.path.join('./result', args.name, 'figure'))
        os.mkdir(os.path.join('./result', args.name, 'train_data'))

    args.result_imgs = os.path.join('./result', args.name, 'imgs')
    args.result_checkpoints = os.path.join('./result', args.name, 'checkpoints')
    args.result_loss_txt = os.path.join('./result', args.name, 'loss_txt')
    args.result_train_data = os.path.join('./result', args.name, 'train_data')

    dataset = ImageDataSet(data_root=args.data_root,
                           input_size=256,
                           open_Method='RGB')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)

    # The Generator of Pix2PixHD, x -> y
    G = GlobalGenerator(input_nc=args.input_nc,
                        output_nc=args.output_nc,
                        ngf=args.ngf,
                        n_downsampling=args.n_downsampling_G,
                        n_blocks=args.n_blocks_G,
                        norm_layer=get_norm_layer(norm_type=args.norm)).to(device)
    # The Generator of Pix2PixHD, y -> x
    F = GlobalGenerator(input_nc=args.input_nc,
                        output_nc=args.output_nc,
                        ngf=args.ngf,
                        n_downsampling=args.n_downsampling_G,
                        n_blocks=args.n_blocks_G,
                        norm_layer=get_norm_layer(norm_type=args.norm)).to(device)
    G.apply(weights_init)
    F.apply(weights_init)

    # The Discriminator of Pix2PixHD is used to discriminate domainX
    DB = MultiscaleDiscriminator(input_nc=args.input_nc + args.output_nc,
                                 ndf=args.ndf,
                                 n_layers=args.n_layers_D,
                                 norm_layer=get_norm_layer(norm_type=args.norm),
                                 use_sigmoid=args.use_sigmoid,
                                 num_D=args.num_D,
                                 getIntermFeat=args.getIntermFeat).to(device)
    # The Discriminator of Pix2PixHD is used to discriminate domainX
    DA = MultiscaleDiscriminator(input_nc=args.input_nc + args.output_nc,
                                 ndf=args.ndf,
                                 n_layers=args.n_layers_D,
                                 norm_layer=get_norm_layer(norm_type=args.norm),
                                 use_sigmoid=args.use_sigmoid,
                                 num_D=args.num_D,
                                 getIntermFeat=args.getIntermFeat).to(device)
    DB.apply(weights_init)
    DA.apply(weights_init)

    # Loss items
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
    criterionFeat = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device=device)
    criterionSemi = SemiLoss().to(device)

    Gen_optim = torch.optim.Adam(itertools.chain(G.parameters(),
                                               F.parameters()),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))
    Dis_optim = torch.optim.Adam(itertools.chain(DB.parameters(),
                                                 DA.parameters()),
                               lr=args.lr,
                               betas=(args.beta1, args.beta2))

    for epoch in range(args.epoch):
        if epoch % args.save_img_epoch == 0:
            pass
        if epoch % args.save_model_epoch == 0:
            pass
        count = 0
        for idx, data in enumerate(dataloader):
            loss_dict = {}
            # 输入数据
            supervisied_img_A = data['supervisied_img_A'].to(device)
            supervisied_img_B = data['supervisied_img_B'].to(device)
            unsupervised_img = data['unsupervised_img'].to(device)

            supervisied_A2B,xlFeatures = G(supervisied_img_A)
            unsupervised_A2B,xuFeatures = G(unsupervised_img)
            supervisied_B2A,ylFeatures = F(supervisied_A2B)
            unsupervised_B2A,yuFeatures = F(unsupervised_A2B)

            # update D
            requires_grad(G, False)
            requires_grad(F, False)
            requires_grad(DB, True)
            requires_grad(DA, True)

            DB_real_pred = 2 * DB(torch.cat([supervisied_img_A, supervisied_img_B], dim=1))
            DB_fake_pred = DB(torch.cat([supervisied_img_A, supervisied_A2B], dim=1)) +\
                           DB(torch.cat([unsupervised_img, unsupervised_A2B], dim=1))

            D_loss_B = 0.5 * criterionGAN(DB_fake_pred, False) +\
                       0.5 * criterionGAN(DB_real_pred, True)

            DA_real_pred = DA(torch.cat([supervisied_A2B, supervisied_img_A], dim=1)) +\
                           DA(torch.cat([unsupervised_A2B, unsupervised_img], dim=1))
            DA_fake_pred = DA(torch.cat([supervisied_A2B, supervisied_B2A], dim=1)) +\
                           DA(torch.cat([unsupervised_A2B, unsupervised_B2A], dim=1))
            D_loss_A = 0.5 * criterionGAN(DA_fake_pred, False) +\
                       0.5 * criterionGAN(DA_real_pred, True)

            loss_dict['D_loss_A'] = D_loss_A
            loss_dict['D_loss_B'] = D_loss_B

            Discriminator_loss = D_loss_A + D_loss_B

            DB.zero_grad()
            DA.zero_grad()
            Discriminator_loss.backward(retain_graph=True)
            Dis_optim.step()

            # update G
            requires_grad(G, True)
            requires_grad(F, True)
            requires_grad(DB, False)
            requires_grad(DA, False)

            GDB_real_pred = 2 * DB(torch.cat([supervisied_img_A, supervisied_img_B], dim=1))
            GDB_fake_pred = DB(torch.cat([supervisied_img_A, supervisied_A2B], dim=1)) + \
                            DB(torch.cat([unsupervised_img, unsupervised_A2B], dim=1))

            GDA_real_pred = DA(torch.cat([supervisied_A2B, supervisied_img_A], dim=1)) + \
                            DA(torch.cat([unsupervised_A2B, unsupervised_img], dim=1))
            GDA_fake_pred = DA(torch.cat([supervisied_A2B, supervisied_B2A], dim=1)) + \
                            DA(torch.cat([unsupervised_A2B, unsupervised_B2A], dim=1))

            G_loss_B = criterionGAN(GDB_fake_pred, True)
            G_loss_A = criterionGAN(GDA_fake_pred, True)

            A2B_GAN_Feat = 0.0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(GDB_fake_pred[i])):
                    A2B_GAN_Feat += D_weights * feat_weights * \
                                       criterionFeat(GDB_fake_pred[i][j],
                                                     GDB_real_pred[i][j].detach()) * args.lambda_feat

            B2A_GAN_Feat = 0.0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(GDA_fake_pred[i])):
                    B2A_GAN_Feat += D_weights * feat_weights * \
                                       criterionFeat(GDA_fake_pred[i][j],
                                                     GDA_real_pred[i][j].detach()) * args.lambda_feat

            A2B_loss_VGG = criterionVGG(supervisied_A2B, supervisied_img_B) * args.lambda_feat
            B2A_loss_VGG = 0.5 * criterionVGG(supervisied_B2A, supervisied_img_A) * args.lambda_feat +\
                           0.5 * criterionVGG(unsupervised_B2A, unsupervised_img) * args.lambda_feat

            # SemiLoss = criterionSemi(xlFeatures, xuFeatures, ylFeatures, yuFeatures)

            loss_dict['G_loss_B'] = G_loss_B
            loss_dict['G_loss_A'] = G_loss_A
            loss_dict['A2B_GAN_Feat'] = A2B_GAN_Feat
            loss_dict['B2A_GAN_Feat'] = B2A_GAN_Feat
            loss_dict['A2B_loss_VGG'] = A2B_loss_VGG
            loss_dict['B2A_loss_VGG'] = B2A_loss_VGG
            # loss_dict['Semi_loss'] = SemiLoss

            Generator_loss = G_loss_B + G_loss_A + A2B_GAN_Feat + B2A_GAN_Feat +\
                             A2B_loss_VGG + B2A_loss_VGG

            G.zero_grad()
            F.zero_grad()
            Generator_loss.backward()
            Gen_optim.step()

            loss_log = ''
            for l in loss_dict:
                loss_log += l + ":" + str(round(loss_dict[l].item(), 4)) + '; '

            print(('epoch_{}_{}:' + loss_log).format(epoch,idx))
            epoch_loss_dir = os.path.join(args.result_loss_txt, 'epoch_' + str(epoch) + '.txt')
            with open(epoch_loss_dir, 'a+') as f:
                f.write(loss_log + '\n')

            if idx >= 0 and idx < 50:
                this_epoch = os.path.join(args.result_train_data, 'epoch_' + str(epoch))
                if not os.path.exists(this_epoch):
                    os.mkdir(this_epoch)

                supres = torch.cat([supervisied_img_A, supervisied_img_B, supervisied_A2B, supervisied_B2A], dim=3)
                unsupres = torch.cat([unsupervised_img, unsupervised_A2B,unsupervised_B2A], dim=3)


                utils.save_image(
                    supres,
                    os.path.join(this_epoch, 'supervisied_' + str(count) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    unsupres,
                    os.path.join(this_epoch, 'unsupervisied_' + str(count) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                count += 1
