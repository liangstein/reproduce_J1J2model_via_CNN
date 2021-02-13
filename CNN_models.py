import jax,jax.numpy as jp
import flax.linen as nn

L=10
site_number=int(L**2)

#build network
class singleCNN(nn.Module):
    @nn.compact
    def basic_module(self,x):
        x = nn.Conv(features=128, kernel_size=(9, 9), padding='VALID', dtype=jp.float64)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.ConvTranspose(features=1, kernel_size=(2, 2), strides=(2, 2), dtype=jp.float64)(x)
        x = x.reshape(x.shape[0], -1)
        x = jp.prod(x, 1)
        return x
    def __call__(self, x1, x2, x3, x4):
        y1=self.basic_module(x1)
        y2=self.basic_module(x2)
        y3=self.basic_module(x3)
        y4=self.basic_module(x4)
        y=y1+y2+y3+y4
        return y


class MPSR_singleCNN(nn.Module):
    @nn.compact
    def basic_module(self,x):
        x = nn.Conv(features=90, kernel_size=(9, 9), padding='VALID', dtype=jp.float64)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.ConvTranspose(features=1, kernel_size=(2, 2), strides=(2, 2), dtype=jp.float64)(x)
        x = x.reshape(x.shape[0], -1)
        x = jp.prod(x, 1)
        return x
    def __call__(self, x1, x2, x3, x4):
        y1=self.basic_module(x1)
        y2=self.basic_module(x2)
        y3=self.basic_module(x3)
        y4=self.basic_module(x4)
        y=y1+y2+y3+y4
        return y