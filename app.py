
from flask import Flask,render_template,request,json
import base64
import numpy as np
import cv2
from math import *
from skimage.filters import gaussian
from PIL import Image
from models import FFA
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

app = Flask(__name__)
#---------------图片转换模块----------------
def Base64ToMat(base):#base64转mat
    imgData = base64.b64decode(base)
    nparr = np.fromstring(imgData, np.uint8)
    imgData = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # base64转mat
    return imgData

def MatToBase64(mat):#mat转base64
    data = cv2.imencode('.jpg', mat)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4

#---------------图片处理模块----------------
#亮度对比度调节
def updateAlpha(img, alpha, beta):
    alpha = 1.00 / (1.00 + exp(-alpha)) + 0.5
    img = np.uint8(np.clip((alpha * img + beta), 0, 255))
    return img

#直方图均衡化
def pictuer_eq(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result

# 曝光调节
def gamma_trans(img, gamma_values):  # gamma函数处理
    if gamma_values < 0:
        gamma = float(-gamma_values * 0.1)  # gamma取值
    else:
        gamma = float((100-gamma_values) * 0.01)  # gamma取值
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表-(Color/255)^R*255
    # 当gamma > 1时,颜色空间的数值在变换后整体下降，R越大，颜色数值下降得越明显，宏观的表现为图片亮度下降增加
    # 当0 < R < 1时，颜色空间的数值在变换后整体上升，R越小，颜色数值上升得越明显，宏观的表现为图片亮度上升增加

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
    # 通过LUT，我们可以将一组RGB值输出为另一组RGB值，从而改变画面的曝光与色彩
    img = cv2.LUT(img, gamma_table)
    print(type(img))

    return img

# 饱和度调节
def Image_saturation_adjustment(img,Increment):     #图片饱和度调整和
    img = img * 1.0
    img_out = img * 1.0

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0

    mask_1 = L < 0.5

    s1 = Delta/(value + 0.001)
    s2 = Delta/(2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    if Increment >= 0 :
        temp = Increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - Increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1/(alpha + 0.001) -1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = Increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)

    img_out = img_out/255.0

    # 饱和处理
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1

    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    img_out = img_out*255
    print(type(img_out))
    return img_out

# 锐化调节
def Image_sharpening(img,Increment):     #图片锐化
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)


    img_out = (img - gauss_out) * Increment + img

    img_out = img_out/255.0

    # 饱和处理
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1

    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    img_out = img_out * 255
    return img_out

# 平滑调节
def Image_smoothing(image):     #图像平滑处理
    noise_img = noise(image)
    # 均值滤波
    result1 = cv2.blur(noise_img, (5, 5))
    # 方框滤波
    result2 = cv2.boxFilter(noise_img, -1, (5, 5), normalize=1)
    # 高斯滤波
    result3 = cv2.GaussianBlur(noise_img, (3, 3), 0)
    # 中值滤波
    result4 = cv2.medianBlur(noise_img, 3)

    return result4
def noise(img):
    out = img
    rows, cols, chn = img.shape
    for i in range(5000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        out[x, y, :] = 255
    return out

# 雾化操作
def Picture_defogging(haze):               #图片去雾
    path_save = '../image01.jpg'
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--task',type=str,default='its',help='its or ots')
    # parser.add_argument('--test_imgs',type=str,default='test_imgs',help='Test imgs folder')
    # opt=parser.parse_args()
    # dataset=opt.task
    dataset='ots'      #模型选择its或者ots
    #print("dataset:",dataset)
    gps=3
    blocks=19
    #img_dir='C:/Users/ASUS/Desktop/cl数字信号任务/FFA-Net-master/'
    #print("img_dir:",img_dir)
    #output_dir=abs+f'pred_FFA_{dataset}/'
    #print("pred_dir:",output_dir)
    #if not os.path.exists(output_dir):
        #os.mkdir(output_dir)
    # model_dir='./'+f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
    model_dir='D:/University Life/大三上/数字信号/课设/PictureTest/defogging/trained_models/ots_train_ffa_3_19.pk'
    #print(model_dir)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckp=torch.load(model_dir,map_location=device)
    net=FFA(gps=gps,blocks=blocks)
    net=nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()
    # for im in os.listdir(img_dir):
    #     print(f'\r {im}',end='',flush=True)
    #     haze = Image.open(img_dir+im)
    #     haze1= tfs.Compose([
    #         tfs.ToTensor(),
    #         tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    #     ])(haze)[None,::]
    #     haze_no=tfs.ToTensor()(haze)[None,::]
    #     with torch.no_grad():
    #         pred = net(haze1)
    #     ts=torch.squeeze(pred.clamp(0,1).cpu())
    #     tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    #     vutils.save_image(ts,output_dir+im.split('.')[0]+'_FFA.png')

    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
            pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())

    vutils.save_image(ts,path_save)
    img = cv2.imread(path_save)
    print(type(img))
    return img
def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256
def tensorShow(tensors, titles=['haze']):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


if __name__ == "__main__":
    net = FFA(gps=3, blocks=19)
    print(net)


#---------------路由模块----------------
#亮度与对比度调节
@app.route('/bright_contrast',methods=['POST'])
def bright_contrast():
    request_img = str(json.loads(request.values.get("data")))
    alpha = int(json.loads(request.values.get("alpha")))
    beta = int(json.loads(request.values.get("beta")))
    if request.method == 'POST':
        imgData = Base64ToMat(request_img)#base64转mat
        result = updateAlpha(imgData,alpha*2,beta*2)#直方图均衡化
        image_base4 = MatToBase64(result)#mat转base64
    return json.dumps(image_base4)


# 曝光
@app.route('/exposure',methods=['POST'])
def exposure():
    request_img = str(json.loads(request.values.get("data")))
    bright_value = int(json.loads(request.values.get("opvalue")))
    if request.method == 'POST':
        imgData = Base64ToMat(request_img)#base64转mat
        result = gamma_trans(imgData, bright_value)  # 曝光调节
        image_base4 = MatToBase64(result)#mat转base64
    return json.dumps(image_base4)


# 直方图均衡化
@app.route('/hist',methods=['POST'])
def hist():
    request_img = str(json.loads(request.values.get("data")))
    if request.method == 'POST':
        imgData = Base64ToMat(request_img)  # base64转mat
        result = pictuer_eq(imgData)  # 直方图均衡化
        image_base4 = MatToBase64(result)  # mat转base64
        print(image_base4)
    return json.dumps(image_base4)


# 饱和度调整
@app.route('/saturation',methods=['POST'])
def saturation():
    request_img = str(json.loads(request.values.get("data")))
    bright_value = int(json.loads(request.values.get("opvalue")))
    if request.method == 'POST':
        imgData = Base64ToMat(request_img)#base64转mat
        result = Image_saturation_adjustment(imgData, bright_value * 0.02)
        image_base4 = MatToBase64(result)#mat转base64
    return json.dumps(image_base4)


# 锐化
@app.route('/sharpness',methods=['POST'])
def sharpness():
    if request.method == 'POST':
        request_img = str(json.loads(request.values.get("data")))
        bright_value = int(json.loads(request.values.get("opvalue")))
        imgData = Base64ToMat(request_img)  # base64转mat
        result = Image_sharpening(imgData,bright_value*0.05)#锐化

        image_base4 = MatToBase64(result)  # mat转base64
        print(image_base4)
    return json.dumps(image_base4)


# 平滑
@app.route('/smooth',methods=['POST'])
def smooth():
    if request.method == 'POST':
        request_img = str(json.loads(request.values.get("data")))
        imgData = Base64ToMat(request_img)  # base64转mat
        result = Image_smoothing(imgData)  # 直方图均衡化
        image_base4 = MatToBase64(result)  # mat转base64
    return json.dumps(image_base4)


# 雾化
@app.route('/defogging',methods=['POST'])
def defogging():
    if request.method == 'POST':
        request_img = str(json.loads(request.values.get("data")))
        imgdata = base64.b64decode(request_img)#转为mat格式
        file = open('去雾前.jpg', 'wb')
        file.write(imgdata)
        imgData = Image.open('去雾前.jpg')
        result = Picture_defogging(imgData)  # 去雾
        image_base4 = MatToBase64(result)  # mat转base64
    return json.dumps(image_base4)



if __name__ == '__main__':
    app.run(port=5001,debug=True)

