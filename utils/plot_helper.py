from sys import platform
import matplotlib.pyplot as plt

if platform == "linux" or platform == "linux2":
    import matplotlib
    if matplotlib.get_backend() != "TkAgg":
        matplotlib.use("Agg")


class PlotHelper(object):
    @classmethod
    def plot_img(cls, img, name=None):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        if name is not None:
            ax.set_title(name)
        ax.imshow(img, interpolation="none")
        plt.show()

    @classmethod
    def plot_imgs(cls, imgs, names=None, title=None, return_fig=False):
        fig = plt.figure(figsize=(14, 6))    
        if title is not None:
            ax = plt.gca()
            ax.set_axis_off()
            ax.set_title(title)

        num = len(imgs)
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(1, num, i + 1)

            if names is not None:
                ax.set_title(names[i])
            if img is not None:
                ax.imshow(img, interpolation="none")
        
        if return_fig:
            return fig
        else:
            plt.show()

    @classmethod
    def plot_imgs_vertical(cls, imgs, names=None, title=None, horizontal=True):
        fig = plt.figure(figsize=(6, 12))
        if title is not None:
            ax = plt.gca()
            ax.set_axis_off()
            ax.set_title(title)

        num = len(imgs)
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(num, 1, i + 1)

            if names is not None:
                ax.set_title(names[i])
            if img is not None:
                ax.imshow(img, interpolation="none")
        plt.show()    

    @classmethod
    def plot_imgs_grid(cls, imgs, names=None, title=None, mod_num=4, figsize=(10, 8), set_axis_off=False):
        if mod_num == 0 or len(imgs) == 0:
            return
        if names is not None:
            if len(names) != len(imgs):
                return
        
        fig = plt.figure(figsize=figsize)    
        if title is not None:
            ax = plt.gca()
            ax.set_axis_off()
            ax.set_title(title)

        num = len(imgs)
        row = int(num / mod_num + (mod_num - 1 )/mod_num)

        for i, img in enumerate(imgs):
            ax = fig.add_subplot(row, mod_num, i + 1)

            if set_axis_off:
                ax.set_axis_off()
            if names is not None:
                ax.set_title(names[i])
            if img is not None:
                ax.imshow(img, interpolation="none")
        plt.show()    

    @classmethod
    def plot_imgs_grid_2(cls, imgs, imgs_x, names=None, title=None, mod_num=4, figsize=(10, 8), set_axis_off=False):
        if mod_num == 0 or len(imgs) == 0:
            return
        if names is not None:
            if len(names) != len(imgs):
                return
        
        num = len(imgs) 
        num_div = num - num % mod_num
        if num_div == 0:
            num_div = num 
        
        imgs_div = imgs[0:num_div]
        imgs_x_div = imgs_x[0:num_div]
        imgs_cmb = []
        imgs_cmb.extend(imgs_div)
        imgs_cmb.extend(imgs_x_div)
        cls.plot_imgs_grid(imgs_cmb, mod_num=mod_num, figsize=figsize, set_axis_off=set_axis_off)
