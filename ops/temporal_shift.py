# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
######


import matplotlib.pyplot as plt

import numpy as np

###########

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))
        ###############
        self.sigmoid = nn.Sigmoid()
        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        '''
        self.action_p2_squeeze = nn.Conv2d(x.size(1), x.size(1)//16 , kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
                                         bias=False, padding=1,
                                         groups=1)
        
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))
        '''
        #####3

    def forward(self, x):

        print('xxx:', x.shape)
        ##########

        '''
        if x.shape[3]==56:
            #selected_channels = [0, 1, 2]  # Replace with the channel indices you want to visualize
            for channel in range(0,18):#selected_channels:
                original_image=x[channel,0,:,:].detach().cpu().numpy()
                #plt.imshow(original_image, cmap='gray')  # Plot the average activation map
                #plt.title(f'original_image {channel} Activation Map')
                original_imagename='./size_'+str(x.shape[3])+'/'+str(channel)+'original.jpg'
                plt.imsave(original_imagename, original_image, cmap='gray')
                #plt.savefig(original_imagename, bbox_inches='tight', pad_inches=0)
                #plt.axis('off')
                #plt.show()

                xx=x[channel,:,:,:]
                feature_map = torch.mean(xx, dim=0)
                feature_map=feature_map.detach().cpu().numpy()
                #feature_map = x[0, channel,:, :].detach().cpu().numpy()
                print('feature_map:',feature_map.shape)
                #################hot
                # 归一化特征图，使数值范围在（0，1）之间
                normalized_feature_map = (feature_map - np.min(feature_map)) / (
                            np.max(feature_map) - np.min(feature_map))
                #print('normalized_feature_map:',normalized_feature_map)
                # 使用Matplotlib创建热力图
                #plt.imshow(normalized_feature_map, cmap='hot', interpolation='nearest')
                #plt.colorbar()  # 添加颜色条以显示值与颜色之间的映射
                #plt.title("Normalized Feature Map")
                #plt.clim(0.3, 1.0)
                # 保存结果为图片文件
                # 设置阈值范围
                threshold_min = 0.7
                threshold_max = 1.0

                # 创建一个阈值化的特征图，将超出阈值范围的值设为0
                normalized_feature_map = np.where(
                    (normalized_feature_map >= threshold_min) & (normalized_feature_map <= threshold_max),
                    normalized_feature_map, 0)
                hot_imagename='./size_'+str(x.shape[3])+'/'+str(channel)+'hot_image.jpg'
                plt.imsave(hot_imagename, normalized_feature_map, cmap='hot')
                #plt.savefig(hot_imagename, bbox_inches='tight', pad_inches=0)
                #plt.savefig(hot_imagename, bbox_inches='tight', pad_inches=0)
                #plt.show()
                #breakpoint()

                ##################
            breakpoint()
        '''
        #############
        #x = self.global_shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = x#self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        #print(x.size())
        #breakpoint()
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    def global_shift(self, x, n_segment, fold_div=None, inplace=False):
        #print(x.shape)#torch.Size([80, 64, 56, 56])

        x0=x
        nt, c1, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c1, h, w)
        # print(x.shape)
        # x1,x2,x3,x4,x5=x[:,0:1,:,:,:],x[:,1:2,:,:,:],x[:,2:3,:,:,:],x[:,3:4,:,:,:],x[:,4:5,:,:,:]
        x_shift = x0
        # print('x_shift',x_shift.shape)#x_shift torch.Size([16, 64, 56, 56])

        x_p2 = self.avg_pool(x_shift)
        # print('x_p2',x_p2.shape)

        action_p2_squeeze = nn.Conv2d(c1, c1 // 16, kernel_size=(1, 1), stride=(1, 1),
                                      bias=False, padding=(0, 0)).to(x.device)
        x_p2 = action_p2_squeeze(x_p2).to(x.device)
        # print('x_p2', x_p2.shape)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        # print('x_p2', x_p2.shape)
        action_p2_conv1 = nn.Conv1d(c, c, kernel_size=3, stride=1, bias=False, padding=1, groups=1).to(x.device)
        x_p2 = action_p2_conv1(x_p2).to(x.device)
        # print('x_p2', x_p2.shape)
        x_p2 = self.relu(x_p2)
        # print('x_p2', x_p2.shape)
        x_p2 = x_p2.transpose(2, 1).contiguous().view(-1, c, 1, 1)
        # print('x_p2', x_p2.shape)
        action_p2_expand = nn.Conv2d(c, c1, kernel_size=(1, 1), stride=(1, 1),
                                     bias=False, padding=(0, 0)).to(x.device)
        x_p2 = action_p2_expand(x_p2).to(x.device)
        # print('x_p2', x_p2.shape)
        x_p2 = self.sigmoid(x_p2)
        # print('x_p2', x_p2.shape)#x_p2 torch.Size([16, 64, 1, 1])
        # print('x_shift', x_shift.shape)
        output1 = x_shift * x_p2 #+ x_shift  # x_p2 torch.Size([16, 64, 56, 56])
        # print('x_p2', x_p2.shape)
        # breakpoint()
        ##########################
        x_p3 = self.avg_pool(x_shift)
        #print('x_p3', x_p3.shape)
        nt, c, h, w = x_p3.size()
        action_p3_expand = nn.Conv2d(c, c, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0)).to(x.device)
        x_p3 = action_p3_expand(x_p3).to(x.device)
        #print('x_p3', x_p3.shape)
        x_p3 = self.sigmoid(x_p3)
        #print('x_p3', x_p3.shape)#x_p3 torch.Size([64, 64, 1, 1])
        output2 = x_shift * x_p3 #+ x_cat
        #print('x_p3', x_p3.shape)  #x_p3 torch.Size([64, 64, 56, 56])
        output=x_shift+output2
        return output  # out.view(nt, c, h, w)
    '''
    def global_shift(self, x, n_segment, fold_div=None, inplace=False):
        #print(x.shape)#torch.Size([80, 64, 56, 56])
        nt, c1, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c1, h, w)
        #print(x.shape)
        #x1,x2,x3,x4,x5=x[:,0:1,:,:,:],x[:,1:2,:,:,:],x[:,2:3,:,:,:],x[:,3:4,:,:,:],x[:,4:5,:,:,:]
        x1, x2, x3, x4, x5 = x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :], x[:, 3, :, :, :], x[:, 4,:, :, :]
        ##############3

        x3=x3.view(-1, c1, h, w)#.transpose(2,1).contiguous()
        #print(x3.shape)#torch.Size([16, 64, 1, 56, 56])
        x=x3
        n_segment=1
        nt, c1, h, w = x.size()
        n_batch = nt // n_segment
        # 2D convolution: c*T*1*1, channel excitation
        x_shift=x
        #print('x_shift',x_shift.shape)#x_shift torch.Size([16, 64, 56, 56])
        x_p2 = self.avg_pool(x_shift)
        #print('x_p2',x_p2.shape)

        action_p2_squeeze = nn.Conv2d(c1, c1 // 16, kernel_size=(1, 1), stride=(1, 1),
                                                                         bias=False, padding=(0, 0)).to(x.device)
        x_p2 = action_p2_squeeze(x_p2).to(x.device)
        #print('x_p2', x_p2.shape)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        #print('x_p2', x_p2.shape)
        action_p2_conv1 = nn.Conv1d(c , c , kernel_size=3, stride=1,bias=False, padding=1,groups=1).to(x.device)
        x_p2 = action_p2_conv1(x_p2).to(x.device)
        #print('x_p2', x_p2.shape)
        x_p2 = self.relu(x_p2)
        #print('x_p2', x_p2.shape)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        #print('x_p2', x_p2.shape)
        action_p2_expand = nn.Conv2d(c, c1, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0)).to(x.device)
        x_p2 = action_p2_expand(x_p2).to(x.device)
        #print('x_p2', x_p2.shape)
        x_p2 = self.sigmoid(x_p2)
        x_p8=x_p2
        #print('x_p2', x_p2.shape)#x_p2 torch.Size([16, 64, 1, 1])
        #print('x_shift', x_shift.shape)
        x_p2 = x_shift * x_p2 + x_shift#x_p2 torch.Size([16, 64, 56, 56])
        #print('x_p2', x_p2.shape)
        #breakpoint()
        ###########


        x_cat=torch.cat([x1, x2, x4, x5], 1).view(-1, c1, x1.size(3), x1.size(4))#x_cat torch.Size([64, 64, 56, 56])
        #print('x_cat', x_cat.shape)
        x_p3 = self.avg_pool(x_cat)
        #print('x_p3', x_p3.shape)
        action_p3_expand = nn.Conv2d(c1, c1, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0)).to(x.device)
        x_p3 = action_p3_expand(x_p3).to(x.device)
        #print('x_p3', x_p3.shape)
        x_p3 = self.sigmoid(x_p3)
        #print('x_p3', x_p3.shape)#x_p3 torch.Size([64, 64, 1, 1])
        x_p3 = x_cat * x_p3 + x_cat
        #print('x_p3', x_p3.shape)  #x_p3 torch.Size([64, 64, 56, 56])

        #########################
        x_p3=x_p3.view(-1,4, c1, x_p3.size(2), x_p3.size(3))#x_p3 torch.Size([16, 4, 64, 1, 1])
        #print('x_p3', x_p3.shape)  # x_p3 torch.Size([64, 64, 1, 1])
        x_p3_1, x_p3_2, x_p3_30, x_p3_4=x_p3[:,0,:,:,:],x_p3[:,1,:,:,:],x_p3[:,2,:,:,:],x_p3[:,3,:,:,:]
        x_p3_1, x_p3_2, x_p3_30, x_p3_4 = x_p8*x_p3_1+x1, x_p8*x_p3_2+x2, x_p8*x_p3_30+x4, x_p8*x_p3_4+x5
        #print('x_p3_1', x_p3_2.shape)  # x_p3 torch.Size([64, 64, 1, 1])
        output=torch.cat([x_p3_1, x_p3_2,x_p2 ,x_p3_30, x_p3_4], 0)
        #print('output', output.shape)


        return output#out.view(nt, c, h, w)
        '''

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')




