import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import utils
import os
import torch.nn as nn
import random
from dataset import ImageDataSet
from dataset import TestImageDataSet
from networks.Pix2PixHD.Generator import GlobalGenerator
from networks.Pix2PixHD.Discriminator import MultiscaleDiscriminator
from networks.StyleGAN_Discriminator import D_basic
from utils import get_norm_layer as get_norm_layer
from utils import weights_init as weights_init
from Loss import GANLoss
from Loss import VGGLoss
from Loss import GANLogisticLoss
from Loss import IDLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def save_train_data():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='sketch')
    parser.add_argument('--data_root',type=str,default='/data/haokang/Semi-Supervisied-DataSet/sketch')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--resolution',type=int,default=256)

    parser.add_argument('--epoch',type=int,default=800)
    parser.add_argument('--save_img_epoch',type=int,default=1)
    parser.add_argument('--save_model_epoch',type=int,default=10)

    parser.add_argument('--lr',type=float,default=0.002)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--beta2',type=float,default=0.999)
    parser.add_argument('--lambda_feat',type=float,default=10.0)

    parser.add_argument('--distributed',type=bool,default=False)

    parser.add_argument('--norm',type=str,default='instance')
    parser.add_argument('--input_nc',type=int,default=3)
    parser.add_argument('--output_nc',type=int,default=1)

    parser.add_argument('--ngf',type=int,default=64)
    parser.add_argument('--n_downsampling_G',type=int,default=4)
    parser.add_argument('--n_blocks_G',type=int,default=9)

    parser.add_argument('--ndf',type=int,default=64)
    parser.add_argument('--n_layers_D',type=int,default=3)
    parser.add_argument('--use_sigmoid',type=bool,default=False)
    parser.add_argument('--num_D',type=int,default=2)
    parser.add_argument('--getIntermFeat',type=bool,default=False)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = 'cuda'

    assert args.name
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(os.path.join('./result',args.name)):
        os.mkdir(os.path.join('./result',args.name))
        os.mkdir(os.path.join('./result', args.name,'imgs'))
        os.mkdir(os.path.join('./result', args.name,'checkpoints'))
        os.mkdir(os.path.join('./result', args.name, 'loss_txt'))
        os.mkdir(os.path.join('./result', args.name, 'figure'))
        os.mkdir(os.path.join('./result', args.name, 'train_data'))

    args.result_imgs = os.path.join('./result', args.name, 'imgs')
    args.result_checkpoints = os.path.join('./result', args.name, 'checkpoints')
    args.result_loss_txt = os.path.join('./result', args.name, 'loss_txt')
    args.result_train_data = os.path.join('./result', args.name, 'train_data')

    dataset = ImageDataSet(data_root=args.data_root,
                           input_size=256,
                           open_Method='L')
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)
    # The Generator of Pix2PixHD
    G = GlobalGenerator(input_nc = args.input_nc,
                        output_nc = args.output_nc,
                        ngf = args.ngf,
                        n_downsampling = args.n_downsampling_G,
                        n_blocks = args.n_blocks_G,
                        norm_layer = get_norm_layer(norm_type = args.norm)).to(device)
    G.apply(weights_init)

    # The Discriminator of Pix2PixHD
    D1 = MultiscaleDiscriminator(input_nc = args.input_nc + args.output_nc,
                                ndf = args.ndf,
                                n_layers = args.n_layers_D,
                                norm_layer = get_norm_layer(norm_type = args.norm),
                                use_sigmoid = args.use_sigmoid,
                                num_D = args.num_D,
                                getIntermFeat = args.getIntermFeat).to(device)
    D1.apply(weights_init)

    # # The Discriminator of StyleGAN
    # D2 = D_basic(resolution = args.resolution).to(device)
    # The Discriminator of Pix2PixHD
    D2 = MultiscaleDiscriminator(input_nc=args.input_nc,
                                 ndf=args.ndf,
                                 n_layers=args.n_layers_D,
                                 norm_layer=get_norm_layer(norm_type=args.norm),
                                 use_sigmoid=args.use_sigmoid,
                                 num_D=args.num_D,
                                 getIntermFeat=args.getIntermFeat).to(device)
    D2.apply(weights_init)

    # Loss items
    criterionGAN = GANLoss(use_lsgan=True,tensor=torch.cuda.FloatTensor)
    # criterionLogistic_D = GANLogisticLoss(mode='D')
    # criterionLogistic_G = GANLogisticLoss(mode='G')
    criterionVGG = VGGLoss(device=device)
    criterionFeat = torch.nn.L1Loss()
    criterionID = IDLoss().to(device)

    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        D1 = nn.DataParallel(D1)
        D2 = nn.DataParallel(D2)
        criterionID = nn.DataParallel(criterionID)

    optimizerG = torch.optim.Adam(G.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    optimizerD1 = torch.optim.Adam(D1.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    optimizerD2 = torch.optim.Adam(D2.parameters(),lr = args.lr, betas = (args.beta1, args.beta2))

    for epoch in range(args.epoch):
        if epoch % args.save_img_epoch == 0:
            this_epoch = os.path.join(args.result_imgs,'epoch_'+str(epoch))
            if not os.path.exists(this_epoch):
                os.mkdir(this_epoch)
            with torch.no_grad():
                G.eval()
                test_dataset = TestImageDataSet(args.data_root)
                test_dataloader = DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=4)
                for i, data in enumerate(test_dataloader):
                    if i > 10:
                        break
                    name = data['name'][0]
                    img = data['img'].to(device)
                    style_img = data['style_img'].to(device)

                    img_ = G(img)
                    cat_img = torch.cat([img,img_],dim=3)

                    epoch_dir = os.path.join(this_epoch,name)
                    utils.save_image(
                        cat_img,
                        epoch_dir,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

        if epoch % args.save_model_epoch == 0:
            torch.save(
                {
                    "g": G.state_dict(),
                    "d_1": D1.state_dict(),
                    "d_2": D2.state_dict(),
                },
                os.path.join(args.result_checkpoints,'epoch_' + str(epoch)+'.pt'),
            )

        count = 0 # 表示存储的训练数据个数
        for i,data in enumerate(dataloader):
            loss_dict = {}
            supervisied_img_A = data['supervisied_img_A'].to(device)
            supervisied_img_B = data['supervisied_img_B'].to(device)
            unsupervised_img = data['unsupervised_img'].to(device)
            # style_img = data['style_img'].to(device)

            supervisied_A2B = G(supervisied_img_A)
            unsupervised_A2B = G(unsupervised_img)

            # update D
            requires_grad(G, False)
            requires_grad(D1, True)
            requires_grad(D2, True)

            D1.zero_grad()
            D2.zero_grad()

            D1_real_pred = D1(torch.cat([supervisied_img_A,supervisied_img_B],dim=1))
            D1_fake_pred = D1(torch.cat([supervisied_img_A,supervisied_A2B],dim=1))
            D2_real_pred = D2(supervisied_img_B)
            D2_fake_pred = D2(unsupervised_A2B)

            loss_d1_adv = 0.5 * criterionGAN(D1_fake_pred, False) + \
                          0.5 * criterionGAN(D1_real_pred, True)
            # loss_d2_adv = criterionLogistic_D([D2_real_pred,D2_fake_pred])
            loss_d2_adv = 0.5 * criterionGAN(D2_fake_pred, False) +\
                          0.5 * criterionGAN(D2_real_pred, True)
            loss_dict['d1_adv'] = loss_d1_adv
            loss_dict['d2_adv'] = loss_d2_adv
            loss_D = loss_d1_adv + loss_d2_adv

            loss_D.backward(retain_graph=True)
            optimizerD1.step()
            optimizerD2.step()

            # update G
            requires_grad(G, True)
            requires_grad(D1, False)
            requires_grad(D2, False)

            G.zero_grad()
            G_fake_pred_d1 = D1(torch.cat([supervisied_img_A, supervisied_A2B], dim=1))
            G_real_pred_d1 = D1(torch.cat([supervisied_img_A, supervisied_img_B], dim=1))
            G_fake_pred_d2 = D2(unsupervised_A2B)

            loss_g_d1 = criterionGAN(G_fake_pred_d1, True)
            loss_g_d2 = criterionGAN(G_fake_pred_d2, True)
            loss_G_GAN_Feat = 0.0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(G_fake_pred_d1[i])):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       criterionFeat(G_fake_pred_d1[i][j],
                                                     G_real_pred_d1[i][j].detach()) * args.lambda_feat
            loss_VGG = criterionVGG(supervisied_A2B, supervisied_img_B) * args.lambda_feat
            loss_l2 = F.mse_loss(supervisied_A2B,supervisied_img_B)
            if torch.cuda.device_count() > 1:
                loss_ID_all = criterionID(supervisied_img_A,supervisied_img_B,supervisied_A2B)
                loss_ID = 0.5 * loss_ID_all[0] + 0.5 * loss_ID_all[1]
            else:
                loss_ID = criterionID(supervisied_img_A,supervisied_img_B,supervisied_A2B)

            loss_dict['g_adv_d1'] = loss_g_d1
            loss_dict['g_adv_d2'] = loss_g_d2
            loss_dict['g_gan_feat'] = loss_G_GAN_Feat
            loss_dict['g_vgg'] = loss_VGG
            loss_dict['g_l2'] = loss_l2
            loss_dict['g_ID'] = loss_ID
            loss_G = loss_g_d1 + loss_g_d2 + loss_G_GAN_Feat + loss_VGG + loss_l2 + loss_ID
            loss_G.backward(retain_graph=True)
            optimizerG.step()

            loss_log = ('g_adv_d1:{:.4f};'
                        'g_adv_d2:{:.4f};'
                        'g_gan_feat:{:.4f};'
                        'g_vgg:{:.4f};'
                        'g_l2:{:.4f};'
                        'g_id:{:.4f};'
                        'd1_adv:{:.4f};'
                        'd2_adv:{:.4f};'.format(loss_dict['g_adv_d1'].item(),
                                                loss_dict['g_adv_d2'].item(),
                                                loss_dict['g_gan_feat'].item(),
                                                loss_dict['g_vgg'].item(),
                                                loss_dict['g_l2'].item(),
                                                loss_dict['g_ID'].item(),
                                                loss_dict['d1_adv'].item(),
                                                loss_dict['d2_adv'].item()))



            print(('epoch_{}:' + loss_log).format(epoch))
            epoch_loss_dir = os.path.join(args.result_loss_txt,'epoch_' + str(epoch) + '.txt')
            with open(epoch_loss_dir,'a+') as f:
                f.write(loss_log + '\n')

            if((loss_dict['g_ID'].item() < 0.5 or loss_dict['g_vgg'].item() < 0.7) and epoch >= 3):
                this_epoch = os.path.join(args.result_train_data,'epoch_'+str(epoch))
                if not os.path.exists(this_epoch):
                    os.mkdir(this_epoch)

                supervisied_data = torch.cat([supervisied_img_A,supervisied_img_B,supervisied_A2B],dim = 3)
                unsupervised_data = torch.cat([unsupervised_img,unsupervised_A2B],dim = 3)
                utils.save_image(
                    supervisied_data,
                    os.path.join(this_epoch,'supervisied_' + str(count) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    unsupervised_data,
                    os.path.join(this_epoch, 'unsupervisied_' + str(count) + '.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                count += 1




