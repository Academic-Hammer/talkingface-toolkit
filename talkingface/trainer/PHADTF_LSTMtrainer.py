###�1�7�1�7�1�7�1�7�1�7�1�7�1�7�1�7�1�7
import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse

from dataset import LRW_1D_lstm_3dmm, LRW_1D_lstm_3dmm_pose
from dataset import News_1D_lstm_3dmm_pose

from models import ATC_net

from torch.nn import init
import pdb

class PHADTF_LSTMTrainer(Trainer):
    def __init__(self, config, model):
        super(PHADTF_LSTMTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx,epoch_niter=100,epoch_niter_decay=10, loss_func=None, show_progress=False):
        start = time.time()
        dataset = train_data  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        model = self.model
        visualizer = Visualizer()
        total_iters = 0                # the total number of training iterations

        for epoch in range(epoch_idx, epoch_niter + epoch_niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % 20 == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += train_data.batch_size
                epoch_iter += train_data.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if total_iters % 20 == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % 10 == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                    #sys.exit(-1)

                if total_iters % 20 == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / train_data.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if show_progress:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    if 'memory' in model:
                        print('replace %d, update %d' % (model.replace, model.update))

                if total_iters % 20 == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = '%d_iter_%d' % (epoch,total_iters) if 0 else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % 20 == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            if 'memory' in model:
                print('End of epoch %d / %d \t Time Taken: %d sec, replace %d, update %d' % (epoch, epoch_niter + epoch_niter_decay, time.time() - epoch_start_time, model.replace, model.update))
            else:
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, epoch_niter + epoch_niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()

        return model.get_current_losses
    
    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        # hard-code some parameters for test
        dataset = valid_data  # create a dataset given opt.dataset_mode and other options
        model = self.model      # create a model given opt.model and other options
        # create a website
        #web_dir = os.path.join()  # define the website directory
        #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch), refresh=0, folder=opt.imagefolder)
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        model.eval()
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        #webpage.save()  # save the HTML
        return visuals
    



class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0, folder='images'):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        #self.img_dir = os.path.join(self.web_dir, 'images')
        self.img_dir = os.path.join(self.web_dir, folder)
        self.folder = folder
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                #img(style="width:%dpx" % width, src=os.path.join('images', im))
                                img(style="width:%dpx" % width, src=os.path.join(self.folder, im))

                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        #html_file = '%s/index.html' % self.web_dir
        name = self.folder[6:] if self.folder[:6] == 'images' else self.folder
        html_file = '%s/index%s.html' % (self.web_dir, name)
        if len(name.split('/')) > 1:
            html_file = '%s/%s/index%s.html' % (self.web_dir,os.path.dirname(name),os.path.basename(name)[6:])
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()

# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    short_path1 = ntpath.basename(ntpath.dirname(image_path[0]))
    short_path = short_path1 + '-' + short_path
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)#tensor to numpy array [-1,1]->[0,1]->[0,255]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            # im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio))))

        if aspect_ratio < 1.0:
            # im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            im = np.array(Image.fromarray(im).resize((int(h / aspect_ratio), w)))
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

class Trainer():
    def __init__(self, config):
        if config.lstm == True:
            if config.pose == 0:
                self.generator = ATC_net(config.para_dim)
            else:
                self.generator = ATC_net(config.para_dim+6)
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.generator.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % ( num_params / 1e6))
        print('-----------------------------------------------')
        #pdb.set_trace()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            if len(device_ids) > 1:
                self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            else:
                self.generator     = self.generator.cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()
        initialize_weights(self.generator)
        if config.continue_train:
            state_dict = multi2single(config.model_name, 0)
            self.generator.load_state_dict(state_dict)
            print('load pretrained [{}]'.format(config.model_name))
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        if config.lstm:
            if config.pose == 0:
                self.dataset = LRW_1D_lstm_3dmm(config.dataset_dir, train=config.is_train, indexes=config.indexes)
            else:
                if config.dataset == 'lrw':
                    self.dataset = LRW_1D_lstm_3dmm_pose(config.dataset_dir, train=config.is_train, indexes=config.indexes, relativeframe=config.relativeframe)
                    self.dataset2 = LRW_1D_lstm_3dmm_pose(config.dataset_dir, train='test', indexes=config.indexes, relativeframe=config.relativeframe)
                elif config.dataset == 'news':
                    self.dataset = News_1D_lstm_3dmm_pose(config.dataset_dir, train=config.is_train, indexes=config.indexes, relativeframe=config.relativeframe,
                                newsname=config.newsname, start=config.start, trainN=config.trainN, testN=config.testN)
        
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        if config.dataset == 'lrw':
            self.data_loader_val = DataLoader(self.dataset2,
                                      batch_size=config.batch_size,
                                      num_workers= config.num_thread,
                                      shuffle=False, drop_last=True)
        

    def fit(self):
        config = self.config
        L = config.para_dim

        num_steps_per_epoch = len(self.data_loader)
        print("num_steps_per_epoch", num_steps_per_epoch)
        cc = 0
        t00 = time.time()
        t0 = time.time()
        

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (coeff, audio, coeff2) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    coeff = Variable(coeff.float()).cuda()
                    audio = Variable(audio.float()).cuda()
                else:
                    coeff = Variable(coeff.float())
                    audio = Variable(audio.float())

                #print(audio.shape, coeff.shape) # torch.Size([16, 16, 28, 12]) torch.Size([16, 16, 70])
                fake_coeff= self.generator(audio)

                loss =  self.mse_loss_fn(fake_coeff , coeff)
                
                if config.less_constrain:
                    loss =  self.mse_loss_fn(fake_coeff[:,:,:L], coeff[:,:,:L]) + config.lambda_pose * self.mse_loss_fn(fake_coeff[:,:,L:], coeff[:,:,L:])
                
                # put smooth on pose
                # tidu ermo pingfang
                if config.smooth_loss:
                    loss1 = loss.clone()
                    frame_dif = fake_coeff[:,1:,L:] - fake_coeff[:,:-1,L:] # [16, 15, 6]
                    #norm2 = torch.norm(frame_dif, dim = 1) # default 2-norm, [16, 6]
                    #norm2_ss1 = torch.sum(torch.mul(norm2, norm2), dim=1) # [16, 1]
                    norm2_ss = torch.sum(torch.mul(frame_dif,frame_dif), dim=[1,2])
                    loss2 = torch.mean(norm2_ss)
                    #pdb.set_trace()
                    loss = loss1 + loss2 * config.lambda_smooth
                
                # put smooth on expression
                if config.smooth_loss2:
                    loss3 = loss.clone()
                    frame_dif2 = fake_coeff[:,1:,:L] - fake_coeff[:,:-1,:L]
                    norm2_ss2 = torch.sum(torch.mul(frame_dif2,frame_dif2), dim=[1,2])
                    loss4 = torch.mean(norm2_ss2)
                    loss = loss3 + loss4 * config.lambda_smooth2


                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()


                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    if not config.smooth_loss and not config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                    elif config.smooth_loss and not config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss1, loss2*config.lambda_smooth, t1-t0,  time.time() - t1))
                    elif not config.smooth_loss and config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss3, loss4*config.lambda_smooth2, t1-t0,  time.time() - t1))
                    else:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss1, loss2*config.lambda_smooth, loss4*config.lambda_smooth2, t1-t0,  time.time() - t1))
                
                if (num_steps_per_epoch > 100 and (step) % (int(num_steps_per_epoch  / 10 )) == 0 and step != 0) or (num_steps_per_epoch <= 100 and (step+1) == num_steps_per_epoch):
                    if config.lstm:
                        for indx in range(3):
                            for jj in range(16):
                                name = "{}/real_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                coeff2 = coeff.data.cpu().numpy()
                                np.save(name, coeff2[indx,jj])
                                if config.relativeframe:
                                    name = "{}/real2_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                    np.save(name, coeff2[indx,jj])
                                name = "{}/fake_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                fake_coeff2 = fake_coeff.data.cpu().numpy()
                                np.save(name, fake_coeff2[indx,jj]) 
                        # check val set loss
                        vloss = 0
                        if config.dataset == 'lrw':
                            for step,  (coeff, audio, coeff2) in enumerate(self.data_loader_val):
                                with torch.no_grad():
                                    if step == 100:
                                        break
                                    if config.cuda:
                                        coeff = Variable(coeff.float()).cuda()
                                        audio = Variable(audio.float()).cuda()
                                    fake_coeff= self.generator(audio)
                                    valloss = self.mse_loss_fn(fake_coeff,coeff)
                                    if config.less_constrain:
                                        valloss =  self.mse_loss_fn(fake_coeff[:,:,:L], coeff[:,:,:L]) + config.lambda_pose * self.mse_loss_fn(fake_coeff[:,:,L:], coeff[:,:,L:])
                                    vloss += valloss.cpu().numpy()
                            print("[{}/{}][{}/{}]   val loss:{}".format(epoch+1, config.max_epochs,
                                    step+1, num_steps_per_epoch, vloss/100.))
                        # save model
                        print("[{}/{}][{}/{}]   save model".format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch))
                        torch.save(self.generator.state_dict(),
                                   "{}/atcnet_lstm_{}.pth"
                                   .format(config.model_dir,cc))
                    cc += 1
                 
                t0 = time.time()
        print("total time: {} second".format(time.time()-t00))
 
    def _reset_gradients(self):
        self.generator.zero_grad()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="../model/atcnet/")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="../sample/atcnet/")
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--lstm', type=bool, default= True)
    parser.add_argument('--num_thread', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)
    parser.add_argument('--para_dim', type=int, default=64)
    parser.add_argument('--index', type=str, default='80,144', help='index ranges')
    parser.add_argument('--pose', type=int, default=0, help='whether predict pose')
    parser.add_argument('--relativeframe', type=int, default=0, help='whether use relative frame value for pose')
    # for personalized data
    parser.add_argument('--newsname', type=str, default='Learn_English')
    parser.add_argument('--start', type=int, default=357)
    parser.add_argument('--trainN', type=int, default=300)
    parser.add_argument('--testN', type=int, default=100)
    # for continnue train
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument("--model_name", type=str, default='../model/atcnet_pose0/atcnet_lstm_24.pth')
    parser.add_argument('--preserve_mouth', type=bool, default=False)
    # for remove jittering
    parser.add_argument('--smooth_loss', type=bool, default=False) # smooth in time, similar to total variation
    parser.add_argument('--smooth_loss2', type=bool, default=False) # smooth in time, for expression
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--lambda_smooth2', type=float, default=0.0001)
    # for less constrain for pose
    parser.add_argument('--less_constrain', type=bool, default=False)
    parser.add_argument('--lambda_pose', type=float, default=0.2)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    str_ids = config.index.split(',')
    config.indexes = []
    for i in range(int(len(str_ids)/2)):
        start = int(str_ids[2*i])
        end = int(str_ids[2*i+1])
        if end > start:
            config.indexes += range(start, end)
    #print('indexes', config.indexes)
    print('device', config.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'train'
    import atcnet as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)

