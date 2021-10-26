import torch
import torch.nn as nn
import random

class DeepConvNet(nn.Module):
    def __init__(self, n_classes,input_ch,input_time,
                 batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                    nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                    nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                    nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                    nn.Dropout(p=0.5),
                    
                    nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                    nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                    nn.Dropout(p=0.5),

                    nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                    nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                    # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                    nn.Dropout(p=0.5),

                    nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                    nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), # maxpool 뒤에 mixstyle 추가
                    # MixStyle(p=0.5, alpha=0.1, mix="random"), # MixStyle
                    )
                
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # out = torch.zeros(1, 1, input_ch, input_time)
        #
        # for i, module in enumerate(self.convnet):
        #     print(module)
        #     out = module(out)
        #     print(out.size())
        #

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output=self.clf(output) 

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)



class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        # print(x)
        B = x.size(0)

        # resnet1d이므로 dim=[1,2], deepconvnet은 dim=[2,3]
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1,1)) # resnet1d는 (B, 1, 1), deepconvnet은 (B,1,1,1)
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B-1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2) # tensor 자르는 용도

            # perm의 갯수가 홀수 일때
            if len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2+1)]
                perm_a = perm_a[torch.randperm(B // 2)]
            elif len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2+1)]
            elif len(perm_a)==len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2)]
            
            # print(len(perm), len(perm_a),len(perm_b))
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default='/deep4net_origin_result_ch20_raw')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    args = parser.parse_args()
    model = DeepConvNet(2,30,750,[])
    # model = FcClfNet(embedding_net)
    
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros((1, 1, 30,750)), show_input=False))