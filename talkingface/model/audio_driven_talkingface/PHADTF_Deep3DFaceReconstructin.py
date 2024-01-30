import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat
import sys
from array import array
import math
import pdb
import time

#===============================================================================================================================================================
#demo_19news.py
def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def demo(image_path):
	#输出路径
	save_path = 'output/coeff/19_news'	
	save_path2 = 'output/render/19_news'
	if image_path[-1] == '/':#检查输入路径最后是否有斜杠，并获取路径中的基本名称（不带路径的目录名）
		image_path = image_path[:-1]
	name = os.path.basename(image_path)
	print(image_path, name)
	#列出指定路径下所有的.txt文件，并将其转换为对应的.png文件，并进行排序。
	img_list = glob.glob(image_path + '/' + '*.txt')
	img_list = [e[:-4]+'.png' for e in img_list]
	img_list = sorted(img_list)
	print('img_list len:', len(img_list))
	#接着，它检查输出文件夹是否存在，如果不存在则创建。
	if not os.path.exists(os.path.join(save_path,name)):
		os.makedirs(os.path.join(save_path,name))
	if not os.path.exists(os.path.join(save_path2,name)):
		os.makedirs(os.path.join(save_path2,name))

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()
  
	#加载人脸模型数据和标准的预处理图片的地标点信息。
	#在这里，代码创建了一个BFM类的实例facemodel，并调用load_lm3d函数加载标准的人脸三维地标点。
	# read face model
	facemodel = BFM()
	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	n = 0
	t1 = time.time()

	# build reconstruction model
	#with tf.Graph().as_default() as graph,tf.device('/cpu:0'):
	#创建了一个新的 TensorFlow 图作为默认图。
	with tf.Graph().as_default() as graph:
		#定义了一个 TensorFlow 占位符images，用于接收输入的图像数据。然后通过load_graph函数加载了一个预训练的神经网络图谱，并将其导入到当前的 TensorFlow 图中。
		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257) 
		#获取了通过神经网络输出的系数（代表人脸形状和颜色）的张量coeff。
		coeff = graph.get_tensor_by_name('resnet/coeff:0')
		#定义了用于渲染面部的计算图。
		faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
		facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
		facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
		rendered = Render_layer(faceshaper,facenormr,facecolor,facemodel,1)# 训练

		#这里创建了一个 TensorFlow 占位符rstimg，用于接收渲染后的图像数据。然后使用tf.image.encode_png函数对rstimg进行 PNG 格式编码。
		rstimg = tf.placeholder(name = 'rstimg', shape = [224,224,4], dtype=tf.uint8)
		encode_png = tf.image.encode_png(rstimg)

		#在 TensorFlow 会话中，对输入图像列表中的每个文件进行处理。同时，将计数器n加一，并加载图像及其对应的五个面部地标点。然后更新文件路径。
		with tf.Session() as sess:
			print('reconstructing...')
			for file in img_list:
				n += 1
				# load images and corresponding 5 facial landmarks
				img,lm = load_img(file,file[:-4]+'.txt')
				file = file.replace(image_path, name)
				# preprocess input image
				#对输入图像进行预处理，并获取变换参数。如果是第一次处理（n==1），则记录下第一帧的变换参数。
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)
				if n==1:
					transform_firstflame=transform_params
				input_img2,lm_new2 = Preprocess2(img,lm,transform_firstflame)
				#使用 TensorFlow 会话执行神经网络模型，获取系数coef。
				coef = sess.run(coeff,feed_dict = {images: input_img})
				#根据系数进行面部重建，并生成最终的渲染图像。
				#将渲染后的图像保存为.png文件。
				face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,facemodel)
				final_images = sess.run(rendered, feed_dict={faceshaper: face_shape_r.astype('float32'), facenormr: face_norm_r.astype('float32'), facecolor: face_color.astype('float32')})
				result_image = final_images[0, :, :, :]
				result_image = np.clip(result_image, 0., 1.).copy(order='C')
				result_bytes = sess.run(encode_png,{rstimg: result_image*255.0})
				result_output_path = os.path.join(save_path2,file[:-4]+'_render.png')
				with open(result_output_path, 'wb') as output_file:
					output_file.write(result_bytes)

				# reshape outputs
				#对预处理后的输入图像进行处理，将其保存为.png文件。
				input_img = np.squeeze(input_img)
				im = Image.fromarray(input_img[:,:,::-1])
				cropped_output_path = os.path.join(save_path2,file[:-4]+'.png')
				im.save(cropped_output_path)

				input_img2 = np.squeeze(input_img2)
				im = Image.fromarray(input_img2[:,:,::-1])
				cropped_output_path = os.path.join(save_path2,file[:-4]+'_input2.png')
				im.save(cropped_output_path)

				# save output files
				savemat(os.path.join(save_path,file[:-4]+'.mat'),{'coeff':coef,'lm_5p':lm_new2-lm_new})
	t2 = time.time()
	print('Total n:', n, 'Time:', t2-t1)

if __name__ == '__main__':
	demo(sys.argv[1])
 
 
#===============================================================================================================================================================
#demo_gettex.py
class RenderObject(object):
    def __init__(self, sess):
        #通过检查是否存在名为'./BFM/BFM_model_front.mat'的文件来决定是否调用transferBFM09函数转换数据模型
        if not os.path.isfile('./BFM/BFM_model_front.mat'):
            transferBFM09()
	    # read face model
        self.facemodel = BFM()

        self.faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
        self.facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
        self.facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
        self.rendered = Render_layer(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)
        self.rendered2 = Render_layer2(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)
        #self.project = Project_layer(self.faceshaper)

        self.rstimg = tf.placeholder(name = 'rstimg', dtype=tf.uint8)
        self.encode_png = tf.image.encode_png(self.rstimg)
        
        self.sess = sess
    
    def save_image(self, final_images, result_output_path):
        result_image = final_images[0, :, :, :]
        result_image = np.clip(result_image, 0., 1.).copy(order='C')
        #result_bytes = sess.run(tf.image.encode_png(result_image*255.0))
        result_bytes = self.sess.run(self.encode_png, {self.rstimg: result_image*255.0})
        with open(result_output_path, 'wb') as output_file:
            output_file.write(result_bytes)

    def save_image2(self, final_images, result_output_path, tx=0, ty=0):
        result_image = final_images[0, :, :, :]
        result_image = np.clip(result_image, 0., 1.) * 255.0
        result_image = np.round(result_image).astype(np.uint8)
        im = Image.fromarray(result_image,'RGBA')
        if tx != 0 or ty != 0:
            im = im.transform(im.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))
        im.save(result_output_path)

    #定义了一个名为show_clip_vertices的方法，其作用是在图像上绘制裁剪顶点的位置。
    def show_clip_vertices(self, coef_path, clip_vertices, image_width=224, image_height=224):
        half_image_width = 0.5 * image_width
        half_image_height = 0.5 * image_height
        im = cv2.imread(coef_path.replace('coeff','render')[:-4]+'.png')
        for i in range(clip_vertices.shape[1]):
            if clip_vertices.shape[2] == 4:
                v0x = clip_vertices[0,i,0]
                v0y = clip_vertices[0,i,1]
                v0w = clip_vertices[0,i,3]
                px = int(round((v0x / v0w + 1.0) * half_image_width))
                py = int(image_height -1 - round((v0y / v0w + 1.0) * half_image_height))
            elif clip_vertices.shape[2] == 2:
                px = int(round(clip_vertices[0,i,0]))
                py = int(round(clip_vertices[0,i,1]))
            if px >= 0 and px < image_width and py >= 0 and py < image_height:
                cv2.circle(im, (px, py), 1, (0, 255, 0), -1)
        cv2.imwrite('show_clip_vertices.png',im)
    
    #这两个方法都用于获取面部纹理数据，并将其保存到文件中。
    #这两个方法的区别在于gettexture方法使用了文件路径的替换操作(coef_path.replace('coeff','render'))来生成图片路径，而gettexture2方法则直接使用原始的图片路径。
    def gettexture(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        img_path = coef_path.replace('coeff','render')[:-4]+'.png'
        face_shape_r,face_norm_r,face_color,face_color2,face_texture2,tri,face_projection = Reconstruction_for_render_new(coef,self.facemodel,img_path)
        np.save(coef_path[:-4]+'_tex2.npy',face_texture2)
        return coef_path[:-4]+'_tex2.npy', face_texture2

    def gettexture2(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        img_path = coef_path[:-4]+'.jpg'
        face_shape_r,face_norm_r,face_color,face_color2,face_texture2,tri,face_projection = Reconstruction_for_render_new(coef,self.facemodel,img_path)
        np.save(coef_path[:-4]+'_tex2.npy',face_texture2)
        return coef_path[:-4]+'_tex2.npy'

#进行面部渲染并保存结果图像。
#这三个方法的共同点是都利用了加载的系数数据和人脸模型进行面部重建和渲染，并调用了save_image来保存渲染结果图像。区别在于render224_new和render224_new2使用了额外的纹理路径参数来进行渲染。
    def render224_new(self, coef_path, result_output_path, tex2_path):
        #首先检查是否存在指定的coef_path和result_output_path，若不存在则直接返回。
        if not os.path.exists(coef_path):
            return
        if os.path.exists(result_output_path):
            return
        data = loadmat(coef_path)
        coef = data['coeff']
        #创建保存结果图像的目录（如果不存在）。
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        #结合加载的系数和给定的纹理路径进行面部重建和渲染操作，得到最终图像数据final_images。使用save_image方法保存渲染结果图像。
        face_shape_r,face_norm_r,face_color2,tri = Reconstruction_for_render_new_given(coef,self.facemodel,tex2_path)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color2.astype('float32')})
        self.save_image(final_images, result_output_path)

    def render224_new2(self, coef_path, result_output_path, tex2):
        if not os.path.exists(coef_path):
            return
        #if os.path.exists(result_output_path):
        #    return
        data = loadmat(coef_path)
        coef = data['coeff']
        
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color2,tri = Reconstruction_for_render_new_given2(coef,self.facemodel,tex2)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color2.astype('float32')})
        self.save_image(final_images, result_output_path)
    
    def render224(self, coef_path, result_output_path):
        if not os.path.exists(coef_path):
            return
        if os.path.exists(result_output_path):
            return
        data = loadmat(coef_path)
        coef = data['coeff']
        
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        #t00 = time.time()
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        #t01 = time.time()
        self.save_image(final_images, result_output_path)
        #print(t01-t00,time.time()-t01)

    def render256(self, coef_path, savedir):
        data = loadmat(coef_path)
        coef = data['coeff']
        
        basen = os.path.basename(coef_path)[:-4]
        result_output_path = os.path.join(savedir,basen+'_render256.png')
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered2, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        self.save_image(final_images, result_output_path)

if __name__ == '__main__':
    with tf.Session() as sess:
        render_object = RenderObject(sess)
        coef_path = sys.argv[1]
        tex2_path,face_texture2 = render_object.gettexture(coef_path)
        result_output_path = coef_path.replace('output/coeff','output/render')[:-4]+'_rendernew.png'
        rp = render_object.render224_new2(coef_path,result_output_path,face_texture2)
        
        

#===============================================================================================================================================================
#load_data.py
# define facemodel for reconstruction
class BFM():
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'#设置了3D面部模型文件的路径。
		model = loadmat(model_path)#使用loadmat函数加载.mat格式的模型文件。
		self.meanshape = model['meanshape'] #存储平均面部形状数据。
		self.idBase = model['idBase'] # 存储用于面部身份特征的基础数据。
		self.exBase = model['exBase'] # 存储用于面部表情的基础数据。
		self.meantex = model['meantex'] # 存储平均面部纹理数据。
		self.texBase = model['texBase'] # 存储面部纹理的基础数据。
		self.point_buf = model['point_buf'] # 存储每个顶点的相邻面索引，这通常用于计算面部法线。
		self.tri = model['tri'] # 存储每个三角面的顶点索引。
		self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 存储68个面部关键点的索引，索引从0开始。
  #总体来说，这个BFM类负责加载和存储3D面部模型的各种参数，包括形状、纹理、表情基础数据，以及面部关键点的索引等信息，这些数据通常用于面部重建和分析。

# 加载面部表情的基础数据
# 从二进制文件中加载面部表情的PCA模型（包括平均表情和主成分），并从文本文件中加载表情数据的标准偏差，这些数据用于面部表情的分析和重建。
def LoadExpBasis():
	n_vertex = 53215#定义变量 n_vertex，表示模型中的顶点数目，这里设为53215。
	Expbin = open('BFM/Exp_Pca.bin','rb')#以二进制读取模式打开文件 BFM/Exp_Pca.bin，这个文件包含面部表情的PCA模型数据。
	exp_dim = array('i')
	exp_dim.fromfile(Expbin,1)#从文件中读取表情的维度，存储在 exp_dim 数组中。
	expMU = array('f')
	expPC = array('f')
	expMU.fromfile(Expbin,3*n_vertex)#从文件中读取平均表情向量（expMU）和表情主成分（expPC）。
	expPC.fromfile(Expbin,3*exp_dim[0]*n_vertex)

	expPC = np.array(expPC)#将 expPC 转换为NumPy数组。
	expPC = np.reshape(expPC,[exp_dim[0],-1])#重塑 expPC 以便每行代表一个主成分。
	expPC = np.transpose(expPC)#转置 expPC 数组，使每列代表一个主成分。

	expEV = np.loadtxt('BFM/std_exp.txt')#从文本文件中加载表情数据的标准偏差。

	return expPC,expEV

# transfer original BFM09 to our face model
def transferBFM09():
	original_BFM = loadmat('BFM/01_MorphableModel.mat')#使用 loadmat 函数从 .mat 文件中加载原始的 BFM 2009 模型数据。
	#提取模型的不同部分：形状基础 (shapePC)、对应的特征值 (shapeEV)、平均面部形状 (shapeMU)
	shapePC = original_BFM['shapePC'] # shape basis
	shapeEV = original_BFM['shapeEV'] # corresponding eigen value
	shapeMU = original_BFM['shapeMU'] # mean face
	#纹理基础 (texPC)、特征值 (texEV)、平均纹理 (texMU)
	texPC = original_BFM['texPC'] # texture basis
	texEV = original_BFM['texEV'] # eigen value
	texMU = original_BFM['texMU'] # mean texture

	expPC,expEV = LoadExpBasis()#加载表情基础和特征值。

	# transfer BFM09 to our face model
	#转换模型数据
 # 转换身份基础: 这几行代码将原始模型的形状基础 (shapePC) 与形状特征值 (shapeEV) 相乘，并对结果进行缩放和截断，以创建新的身份基础 (idBase)。
	idBase = shapePC*np.reshape(shapeEV,[-1,199])
	idBase = idBase/1e5 # unify the scale to decimeter
	idBase = idBase[:,:80] # use only first 80 basis
#转换表情基础: 这几行代码类似地处理表情基础 (expPC) 和特征值 (expEV)，创建新的表情基础 (exBase)。
	exBase = expPC*np.reshape(expEV,[-1,79])
	exBase = exBase/1e5 # unify the scale to decimeter
	exBase = exBase[:,:64] # use only first 64 basis
#转换纹理基础: 这行代码将纹理基础 (texPC) 与特征值 (texEV) 相乘，并截断结果以创建新的纹理基础 (texBase)。
	texBase = texPC*np.reshape(texEV,[-1,199])
	texBase = texBase[:,:80] # use only first 80 basis

	# our face model is cropped align face landmarks which contains only 35709 vertex.
	# original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
	# thus we select corresponding vertex to get our face model.
#加载表情顶点索引: 这两行代码加载并处理表情模型的顶点索引，用于从原始模型中选择特定的顶点。
	index_exp = loadmat('BFM/BFM_front_idx.mat')
	index_exp = index_exp['idx'].astype(np.int32) - 1 #starts from 0 (to 53215)

	index_shape = loadmat('BFM/BFM_exp_idx.mat')
	index_shape = index_shape['trimIndex'].astype(np.int32) - 1 #starts from 0 (to 53490)
	index_shape = index_shape[index_exp]


	idBase = np.reshape(idBase,[-1,3,80])
	idBase = idBase[index_shape,:,:]
	idBase = np.reshape(idBase,[-1,80])

	texBase = np.reshape(texBase,[-1,3,80])
	texBase = texBase[index_shape,:,:]
	texBase = np.reshape(texBase,[-1,80])

	exBase = np.reshape(exBase,[-1,3,64])
	exBase = exBase[index_exp,:,:]
	exBase = np.reshape(exBase,[-1,64])

	meanshape = np.reshape(shapeMU,[-1,3])/1e5
	meanshape = meanshape[index_shape,:]
	meanshape = np.reshape(meanshape,[1,-1])

	meantex = np.reshape(texMU,[-1,3])
	meantex = meantex[index_shape,:]
	meantex = np.reshape(meantex,[1,-1])

	# other info contains triangles, region used for computing photometric loss,
	# region used for skin texture regularization, and 68 landmarks index etc.
	other_info = loadmat('BFM/facemodel_info.mat')
	frontmask2_idx = other_info['frontmask2_idx']
	skinmask = other_info['skinmask']
	keypoints = other_info['keypoints']
	point_buf = other_info['point_buf']
	tri = other_info['tri']
	tri_mask2 = other_info['tri_mask2']

	# save our face model
	savemat('BFM/BFM_model_front.mat',{'meanshape':meanshape,'meantex':meantex,'idBase':idBase,'exBase':exBase,'texBase':texBase,'tri':tri,'point_buf':point_buf,'tri_mask2':tri_mask2\
		,'keypoints':keypoints,'frontmask2_idx':frontmask2_idx,'skinmask':skinmask})

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

	Lm3D = loadmat('./BFM/similarity_Lm3D_all.mat')
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

# load input images and corresponding 5 landmarks
def load_img(img_path,lm_path):

	image = Image.open(img_path)
	lm = np.loadtxt(lm_path)

	return image,lm

# save 3D face to obj file
def save_obj(path,v,f,c):
	with open(path,'w') as file:
		for i in range(len(v)):
			file.write('v %f %f %f %f %f %f\n'%(v[i,0],v[i,1],v[i,2],c[i,0],c[i,1],c[i,2]))

		file.write('\n')

		for i in range(len(f)):
			file.write('f %d %d %d\n'%(f[i,0],f[i,1],f[i,2]))

	file.close()



#===============================================================================================================================================================
#preprocess_img.py
def POS(xp,x):
	npts = xp.shape[1]#获取 xp 的列数，即二维点的数量。xp 是二维图像中的点，而 x 是相应的三维对象中的点。

 #构建矩阵 A 和向量 b：
	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b,rcond=None)#使用最小二乘法解线性方程组 A*k = b。结果 k 包含了正交摄影变换的参数。

#提取和计算正交摄影变换的参数：
	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

def process_img(img,lm,t,s):
	w0,h0 = img.size#获取了输入图像的宽度和高度。
	img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0/2, 0, 1, h0/2 - t[1]))#对图像进行仿射变换，通过调整偏移量t，将图像中心平移到指定位置。
	#计算缩放后的新宽度和高度。
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	#使用LANCZOS插值方法对图像进行缩放。
	img = img.resize((w,h),resample = Image.LANCZOS)
	#对特征点lm进行相同的仿射变换和缩放。
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102

	# 计算裁剪区域的左上角和右下角坐标。
	left = (w/2 - 112).astype(np.int32)
	right = left + 224
	up = (h/2 - 112).astype(np.int32)
	below = up + 224

	#对图像进行裁剪、颜色通道顺序调整、扩展维度，并对特征点进行偏移。
	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1]
	img = np.expand_dims(img,0)
	lm = lm - np.reshape(np.array([(w/2 - 112),(h/2-112)]),[1,2])

	return img,lm


# resize and crop input images before sending to the R-Net
def Preprocess(img,lm,lm3D):

	w0,h0 = img.size

	# 将二维标记点从图像平面坐标转换为三维空间坐标，根据图像高度调整Y坐标。
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# 使用正交姿势估计（POS）算法基于二维和三维标记点计算平移（t）和比例因子（s）。
	t,s = POS(lm.transpose(),lm3D.transpose())

	# 使用计算得到的变换参数处理图像和标记点，得到新图像（img_new）和调整后的标记点（lm_new）。
	img_new,lm_new = process_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])#构造了一个包含宽度、高度、比例因子和平移参数的数组（trans_params），用于进一步使用。

	return img_new,lm_new,trans_params


def Preprocess2(img,lm,trans_params):

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# 不重新计算平移和比例因子，而是直接使用提供的trans_params中的比例因子和平移值。
	s = 102./trans_params[2]
	t = np.stack([trans_params[3],trans_params[4]],axis = 0)

	# processing the image
	img_new,lm_new = process_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)

	return img_new,lm_new



#===============================================================================================================================================================
#reconstruct_mesh.py

def Split_coeff(coeff):#coeff是输入的系数数组，假设是一个二维数组。
	id_coeff = coeff[:,:80] # 包含了前80个系数，用于表示身份（形状）的系数，每个样本的系数为80维。
	ex_coeff = coeff[:,80:144] # expression coeff of dim 64
	tex_coeff = coeff[:,144:224] # texture(albedo) coeff of dim 80
	angles = coeff[:,224:227] # ruler angles(x,y,z) for rotation of dim 3
	gamma = coeff[:,227:254] # lighting coeff for 3 channel SH function of dim 27
	translation = coeff[:,254:] # translation coeff of dim 3

	return id_coeff,ex_coeff,tex_coeff,angles,gamma,translation
#函数返回了上述分割得到的系数数组，依次对应身份系数、表情系数、纹理系数、旋转角度、光照系数和平移系数。

# compute face shape with identity and expression coeff, based on BFM model
# input: id_coeff with shape [1,80]
#		 ex_coeff with shape [1,64]
# output: face_shape with shape [1,N,3], N is number of vertices
#用于计算生成人脸形状
def Shape_formation(id_coeff,ex_coeff,facemodel):#id_coeff是身份（形状）系数，ex_coeff是表情系数，facemodel包含了人脸模型相关的数据。
	face_shape = np.einsum('ij,aj->ai',facemodel.idBase,id_coeff) + \
				np.einsum('ij,aj->ai',facemodel.exBase,ex_coeff) + \
				facemodel.meanshape

	face_shape = np.reshape(face_shape,[1,-1,3])
	# re-center face shape
	face_shape = face_shape - np.mean(np.reshape(facemodel.meanshape,[1,-1,3]), axis = 1, keepdims = True)

	return face_shape#返回生成的人脸形状。

# 用tex_coeff计算顶点纹理(反照率)
# input: tex_coeff with shape [1,N,3]
# output: face_texture with shape [1,N,3], RGB order, range from 0-255
def Texture_formation(tex_coeff,facemodel):

	face_texture = np.einsum('ij,aj->ai',facemodel.texBase,tex_coeff) + facemodel.meantex
	face_texture = np.reshape(face_texture,[1,-1,3])

	return face_texture

# 使用单环邻域计算顶点法线
# input: face_shape with shape [1,N,3]
# output: v_norm with shape [1,N,3]
def Compute_norm(face_shape,facemodel):

	face_id = facemodel.tri # vertex index for each triangle face, with shape [F,3], F is number of faces
	point_id = facemodel.point_buf # adjacent face index for each vertex, with shape [N,8], N is number of vertex
	shape = face_shape
	face_id = (face_id - 1).astype(np.int32)
	point_id = (point_id - 1).astype(np.int32)
	v1 = shape[:,face_id[:,0],:]
	v2 = shape[:,face_id[:,1],:]
	v3 = shape[:,face_id[:,2],:]
	e1 = v1 - v2
	e2 = v2 - v3
	face_norm = np.cross(e1,e2) # compute normal for each face
	face_norm = np.concatenate([face_norm,np.zeros([1,1,3])], axis = 1) # concat face_normal with a zero vector at the end
	v_norm = np.sum(face_norm[:,point_id,:], axis = 2) # compute vertex normal using one-ring neighborhood
	v_norm = v_norm/np.expand_dims(np.linalg.norm(v_norm,axis = 2),2) # normalize normal vectors

	return v_norm

# 计算基于3个直尺角度的旋转矩阵
# input: angles with shape [1,3]
# output: rotation matrix with shape [1,3,3]
def Compute_rotation_matrix(angles):

	angle_x = angles[:,0][0]
	angle_y = angles[:,1][0]
	angle_z = angles[:,2][0]

	# compute rotation matrix for X,Y,Z axis respectively
	rotation_X = np.array([1.0,0,0,\
		0,np.cos(angle_x),-np.sin(angle_x),\
		0,np.sin(angle_x),np.cos(angle_x)])
	rotation_Y = np.array([np.cos(angle_y),0,np.sin(angle_y),\
		0,1,0,\
		-np.sin(angle_y),0,np.cos(angle_y)])
	rotation_Z = np.array([np.cos(angle_z),-np.sin(angle_z),0,\
		np.sin(angle_z),np.cos(angle_z),0,\
		0,0,1])

	rotation_X = np.reshape(rotation_X,[1,3,3])
	rotation_Y = np.reshape(rotation_Y,[1,3,3])
	rotation_Z = np.reshape(rotation_Z,[1,3,3])

	rotation = np.matmul(np.matmul(rotation_Z,rotation_Y),rotation_X)
	rotation = np.transpose(rotation, axes = [0,2,1])  #transpose row and column (dimension 1 and 2)

	return rotation

# 使用face_texture和SH函数光照近似计算顶点颜色
# input: face_texture with shape [1,N,3]
# 	     norm with shape [1,N,3]
#		 gamma with shape [1,27]
# output: face_color with shape [1,N,3], RGB order, range from 0-255
#		  lighting with shape [1,N,3], color under uniform texture
def Illumination_layer(face_texture,norm,gamma):

	num_vertex = np.shape(face_texture)[1]

	init_lit = np.array([0.8,0,0,0,0,0,0,0,0])
	gamma = np.reshape(gamma,[-1,3,9])
	gamma = gamma + np.reshape(init_lit,[1,1,9])

	# parameter of 9 SH function
	a0 = np.pi 
	a1 = 2*np.pi/np.sqrt(3.0)
	a2 = 2*np.pi/np.sqrt(8.0)
	c0 = 1/np.sqrt(4*np.pi)
	c1 = np.sqrt(3.0)/np.sqrt(4*np.pi)
	c2 = 3*np.sqrt(5.0)/np.sqrt(12*np.pi)

	Y0 = np.tile(np.reshape(a0*c0,[1,1,1]),[1,num_vertex,1]) 
	Y1 = np.reshape(-a1*c1*norm[:,:,1],[1,num_vertex,1]) 
	Y2 = np.reshape(a1*c1*norm[:,:,2],[1,num_vertex,1])
	Y3 = np.reshape(-a1*c1*norm[:,:,0],[1,num_vertex,1])
	Y4 = np.reshape(a2*c2*norm[:,:,0]*norm[:,:,1],[1,num_vertex,1])
	Y5 = np.reshape(-a2*c2*norm[:,:,1]*norm[:,:,2],[1,num_vertex,1])
	Y6 = np.reshape(a2*c2*0.5/np.sqrt(3.0)*(3*np.square(norm[:,:,2])-1),[1,num_vertex,1])
	Y7 = np.reshape(-a2*c2*norm[:,:,0]*norm[:,:,2],[1,num_vertex,1])
	Y8 = np.reshape(a2*c2*0.5*(np.square(norm[:,:,0])-np.square(norm[:,:,1])),[1,num_vertex,1])

	Y = np.concatenate([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],axis=2)

	# Y shape:[batch,N,9].

	lit_r = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,0,:],2)),2) #[batch,N,9] * [batch,9,1] = [batch,N]
	lit_g = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,1,:],2)),2)
	lit_b = np.squeeze(np.matmul(Y,np.expand_dims(gamma[:,2,:],2)),2)

	# shape:[batch,N,3]
	face_color = np.stack([lit_r*face_texture[:,:,0],lit_g*face_texture[:,:,1],lit_b*face_texture[:,:,2]],axis = 2)
	lighting = np.stack([lit_r,lit_g,lit_b],axis = 2)*128

	return face_color,lighting

def Reconstruction_for_render(coeff,facemodel):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture = Texture_formation(tex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
	tri = facemodel.tri
	
	return face_shape_r,face_norm_r,face_color,tri


# 将3D人脸投影到图像平面上
# input: face_shape with shape [1,N,3]
# 		 rotation with shape [1,3,3]
#		 translation with shape [1,3]
# output: face_projection with shape [1,N,2]
# 		  z_buffer with shape [1,N,1]
def Projection_layer(face_shape,rotation,translation,focal=1015.0,center=112.0): # we choose the focal length and camera position empirically

	camera_pos = np.reshape(np.array([0.0,0.0,10.0]),[1,1,3]) # camera position
	reverse_z = np.reshape(np.array([1.0,0,0,0,1,0,0,0,-1.0]),[1,3,3])


	p_matrix = np.concatenate([[focal],[0.0],[center],[0.0],[focal],[center],[0.0],[0.0],[1.0]],axis = 0) # projection matrix
	p_matrix = np.reshape(p_matrix,[1,3,3])

	# calculate face position in camera space
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_t = face_shape_r + np.reshape(translation,[1,1,3])
	face_shape_t = np.matmul(face_shape_t,reverse_z) + camera_pos

	# calculate projection of face vertex using perspective projection
	aug_projection = np.matmul(face_shape_t,np.transpose(p_matrix,[0,2,1]))
	face_projection = aug_projection[:,:,0:2]/np.reshape(aug_projection[:,:,2],[1,np.shape(aug_projection)[1],1])
	z_buffer = np.reshape(aug_projection[:,:,2],[1,-1,1])

	return face_projection,z_buffer

def Illumination_inv_layer(face_color,lighting):
	face_texture = np.stack([face_color[:,:,0]/lighting[:,:,0],face_color[:,:,1]/lighting[:,:,1],face_color[:,:,2]/lighting[:,:,2]],axis=2)*128
	return face_texture

def Reconstruction_for_render_new(coeff,facemodel,imgpath):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture = Texture_formation(tex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color,lighting = Illumination_layer(face_texture, face_norm_r, gamma)
	tri = facemodel.tri
	# compute vertex projection on image plane (with image sized 224*224)
	face_projection,z_buffer = Projection_layer(face_shape,rotation,translation)
	face_projection = np.stack([face_projection[:,:,0],224 - face_projection[:,:,1]], axis = 2)
	imcolor = cv2.imread(imgpath)
	fp = face_projection.astype('int')
	face_color2 = imcolor[fp[0,:,1],fp[0,:,0],::-1]
	face_color2 = np.expand_dims(face_color2,0)
	face_texture2 = Illumination_inv_layer(face_color2,lighting)
	
	return face_shape_r,face_norm_r,face_color,face_color2,face_texture2,tri,face_projection

def Render_layer(face_shape,face_norm,face_color,facemodel,batchsize):

	camera_position = tf.constant([0,0,10.0])
	camera_lookat = tf.constant([0,0,0.0])
	camera_up = tf.constant([0,1.0,0])
	light_positions = tf.tile(tf.reshape(tf.constant([0,0,1e5]),[1,1,3]),[batchsize,1,1])
	light_intensities = tf.tile(tf.reshape(tf.constant([0.0,0.0,0.0]),[1,1,3]),[batchsize,1,1])
	ambient_color = tf.tile(tf.reshape(tf.constant([1.0,1,1]),[1,3]),[batchsize,1])

	#pdb.set_trace()
	render = mesh_renderer(face_shape,
		tf.cast(facemodel.tri-1,tf.int32),
		face_norm,
		face_color/255,
		camera_position = camera_position,
		camera_lookat = camera_lookat,
		camera_up = camera_up,
		light_positions = light_positions,
		light_intensities = light_intensities,
		image_width = 224,
		image_height = 224,
		fov_y = 12.5936,
		ambient_color = ambient_color)

	return render

def Render_layer2(face_shape,face_norm,face_color,facemodel,batchsize):

	camera_position = tf.constant([0,0,10.0])
	camera_lookat = tf.constant([0,0,0.0])
	camera_up = tf.constant([0,1.0,0])
	light_positions = tf.tile(tf.reshape(tf.constant([0,0,1e5]),[1,1,3]),[batchsize,1,1])
	light_intensities = tf.tile(tf.reshape(tf.constant([0.0,0.0,0.0]),[1,1,3]),[batchsize,1,1])
	ambient_color = tf.tile(tf.reshape(tf.constant([1.0,1,1]),[1,3]),[batchsize,1])

	#pdb.set_trace()
	render = mesh_renderer(face_shape,
		tf.cast(facemodel.tri-1,tf.int32),
		face_norm,
		face_color/255,
		camera_position = camera_position,
		camera_lookat = camera_lookat,
		camera_up = camera_up,
		light_positions = light_positions,
		light_intensities = light_intensities,
		image_width = 256,
		image_height = 256,
		fov_y = 12.5936,
		ambient_color = ambient_color)

	return render

def Reconstruction_for_render_new_given(coeff,facemodel,tex2_path):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_texture2 = np.load(tex2_path)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color2,lighting = Illumination_layer(face_texture2, face_norm_r, gamma)
	tri = facemodel.tri
	
	return face_shape_r,face_norm_r,face_color2,tri

def Reconstruction_for_render_new_given2(coeff,facemodel,face_texture2):
	id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coeff)
	face_shape = Shape_formation(id_coeff, ex_coeff, facemodel)
	face_norm = Compute_norm(face_shape,facemodel)
	rotation = Compute_rotation_matrix(angles)
	face_shape_r = np.matmul(face_shape,rotation)
	face_shape_r = face_shape_r + np.reshape(translation,[1,1,3])
	face_norm_r = np.matmul(face_norm,rotation)
	face_color2,lighting = Illumination_layer(face_texture2, face_norm_r, gamma)
	tri = facemodel.tri
	
	return face_shape_r,face_norm_r,face_color2,tri


#===============================================================================================================================================================
#mesh_renderer.py

def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
  batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
  light_count = light_positions.shape[1].value
  pixel_count = image_height * image_width
  # Reshape all values to easily do pixelwise computations:
  normals = tf.reshape(normals, [batch_size, -1, 3])
  alphas = tf.reshape(alphas, [batch_size, -1, 1])
  diffuse_colors = tf.reshape(diffuse_colors, [batch_size, -1, 3])
  if camera_position is not None:
    specular_colors = tf.reshape(specular_colors, [batch_size, -1, 3])

  # Ambient component
  output_colors = tf.zeros([batch_size, image_height * image_width, 3])
  if ambient_color is not None:
    ambient_reshaped = tf.expand_dims(ambient_color, axis=1)
    output_colors = tf.add(output_colors, ambient_reshaped * diffuse_colors)

  # Diffuse component
  pixel_positions = tf.reshape(pixel_positions, [batch_size, -1, 3])
  per_light_pixel_positions = tf.stack(
      [pixel_positions] * light_count,
      axis=1)  # [batch_size, light_count, pixel_count, 3]
  directions_to_lights = tf.nn.l2_normalize(
      tf.expand_dims(light_positions, axis=2) - per_light_pixel_positions,
      axis=3)  # [batch_size, light_count, pixel_count, 3]
  # The specular component should only contribute when the light and normal
  # face one another (i.e. the dot product is nonnegative):
  normals_dot_lights = tf.clip_by_value(
      tf.reduce_sum(
          tf.expand_dims(normals, axis=1) * directions_to_lights, axis=3), 0.0,
      1.0)  # [batch_size, light_count, pixel_count]
  diffuse_output = tf.expand_dims(
      diffuse_colors, axis=1) * tf.expand_dims(
          normals_dot_lights, axis=3) * tf.expand_dims(
              light_intensities, axis=2)
  diffuse_output = tf.reduce_sum(
      diffuse_output, axis=1)  # [batch_size, pixel_count, 3]
  output_colors = tf.add(output_colors, diffuse_output)

  # Specular component
  if camera_position is not None:
    camera_position = tf.reshape(camera_position, [batch_size, 1, 3])
    mirror_reflection_direction = tf.nn.l2_normalize(
        2.0 * tf.expand_dims(normals_dot_lights, axis=3) * tf.expand_dims(
            normals, axis=1) - directions_to_lights,
        dim=3)
    direction_to_camera = tf.nn.l2_normalize(
        camera_position - pixel_positions, dim=2)
    reflection_direction_dot_camera_direction = tf.reduce_sum(
        tf.expand_dims(direction_to_camera, axis=1) *
        mirror_reflection_direction,
        axis=3)
    # The specular component should only contribute when the reflection is
    # external:
    reflection_direction_dot_camera_direction = tf.clip_by_value(
        tf.nn.l2_normalize(reflection_direction_dot_camera_direction, dim=2),
        0.0, 1.0)
    # The specular component should also only contribute when the diffuse
    # component contributes:
    reflection_direction_dot_camera_direction = tf.where(
        normals_dot_lights != 0.0, reflection_direction_dot_camera_direction,
        tf.zeros_like(
            reflection_direction_dot_camera_direction, dtype=tf.float32))
    # Reshape to support broadcasting the shininess coefficient, which rarely
    # varies per-vertex:
    reflection_direction_dot_camera_direction = tf.reshape(
        reflection_direction_dot_camera_direction,
        [batch_size, light_count, image_height, image_width])
    shininess_coefficients = tf.expand_dims(shininess_coefficients, axis=1)
    specularity = tf.reshape(
        tf.pow(reflection_direction_dot_camera_direction,
               shininess_coefficients),
        [batch_size, light_count, pixel_count, 1])
    specular_output = tf.expand_dims(
        specular_colors, axis=1) * specularity * tf.expand_dims(
            light_intensities, axis=2)
    specular_output = tf.reduce_sum(specular_output, axis=1)
    output_colors = tf.add(output_colors, specular_output)
  rgb_images = tf.reshape(output_colors,
                          [batch_size, image_height, image_width, 3])
  alpha_images = tf.reshape(alphas, [batch_size, image_height, image_width, 1])
  valid_rgb_values = tf.concat(3 * [alpha_images > 0.5], axis=3)
  rgb_images = tf.where(valid_rgb_values, rgb_images,
                        tf.zeros_like(rgb_images, dtype=tf.float32))
  return tf.reverse(tf.concat([rgb_images, alpha_images], axis=3), axis=[1])


def mesh_renderer(vertices,
                  triangles,
                  normals,
                  diffuse_colors,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  light_positions,
                  light_intensities,
                  image_width,
                  image_height,
                  specular_colors=None,
                  shininess_coefficients=None,
                  ambient_color=None,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if (ambient_color is not None and
      ambient_color.get_shape().as_list() != [batch_size, 3]):
    raise ValueError('Ambient_color must have shape [batch_size, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if specular_colors is not None and shininess_coefficients is None:
    raise ValueError(
        'Specular colors were supplied without shininess coefficients.')
  if shininess_coefficients is not None and specular_colors is None:
    raise ValueError(
        'Shininess coefficients were supplied without specular colors.')
  if specular_colors is not None:
    # Since a 0-D float32 tensor is accepted, also accept a float.
    if isinstance(shininess_coefficients, float):
      shininess_coefficients = tf.constant(
          shininess_coefficients, dtype=tf.float32)
    if len(specular_colors.shape) != 3:
      raise ValueError('The specular colors must have shape [batch_size, '
                       'vertex_count, 3].')
    if len(shininess_coefficients.shape) > 2:
      raise ValueError('The shininess coefficients must have shape at most'
                       '[batch_size, vertex_count].')
    # If we don't have per-vertex coefficients, we can just reshape the
    # input shininess to broadcast later, rather than interpolating an
    # additional vertex attribute:
    if len(shininess_coefficients.shape) < 2:
      vertex_attributes = tf.concat(
          [normals, vertices, diffuse_colors, specular_colors], axis=2)
    else:
      vertex_attributes = tf.concat(
          [
              normals, vertices, diffuse_colors, specular_colors,
              tf.expand_dims(shininess_coefficients, axis=2)
          ],
          axis=2)
  else:
    vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  camera_matrices = look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  pixel_attributes = rasterize(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], axis=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  if specular_colors is not None:
    specular_colors = pixel_attributes[:, :, :, 9:12]
    # Retrieve the interpolated shininess coefficients if necessary, or just
    # reshape our input for broadcasting:
    if len(shininess_coefficients.shape) == 2:
      shininess_coefficients = pixel_attributes[:, :, :, 12]
    else:
      shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

  pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

  renders = phong_shader(
      normals=pixel_normals,
      alphas=pixel_mask,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=camera_position if specular_colors is not None else None,
      specular_colors=specular_colors,
      shininess_coefficients=shininess_coefficients,
      ambient_color=ambient_color)
  return renders


#===============================================================================================================================================================
#camera_utils.py

def perspective(aspect_ratio, fov_y, near_clip, far_clip):
  # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
  # and adds the half-angle factor of .5.
  focal_lengths_y = 1.0 / tf.tan(fov_y * (math.pi / 360.0))
  depth_range = far_clip - near_clip
  p_22 = -(far_clip + near_clip) / depth_range
  p_23 = -2.0 * (far_clip * near_clip / depth_range)

  zeros = tf.zeros_like(p_23, dtype=tf.float32)
  # pyformat: disable
  perspective_transform = tf.concat(
      [
          focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
          zeros, focal_lengths_y, zeros, zeros,
          zeros, zeros, p_22, p_23,
          zeros, zeros, -tf.ones_like(p_23, dtype=tf.float32), zeros
      ], axis=0)
  # pyformat: enable
  perspective_transform = tf.reshape(perspective_transform, [4, 4, -1])
  return tf.transpose(perspective_transform, [2, 0, 1])

def look_at(eye, center, world_up):
  batch_size = center.shape[0].value
  vector_degeneracy_cutoff = 1e-6
  forward = center - eye
  forward_norm = tf.norm(forward, ord='euclidean', axis=1, keepdims=True)
  #tf.assert_greater(
  #    forward_norm,
  #    vector_degeneracy_cutoff,
  #    message='Camera matrix is degenerate because eye and center are close.')
  forward = tf.divide(forward, forward_norm)

  to_side = tf.linalg.cross(forward, world_up)
  to_side_norm = tf.norm(to_side, ord='euclidean', axis=1, keepdims=True)
  #tf.assert_greater(
  #    to_side_norm,
  #    vector_degeneracy_cutoff,
  #    message='Camera matrix is degenerate because up and gaze are close or'
  #    'because up is degenerate.')
  to_side = tf.divide(to_side, to_side_norm)
  cam_up = tf.linalg.cross(to_side, forward)

  w_column = tf.constant(
      batch_size * [[0., 0., 0., 1.]], dtype=tf.float32)  # [batch_size, 4]
  w_column = tf.reshape(w_column, [batch_size, 4, 1])
  view_rotation = tf.stack(
      [to_side, cam_up, -forward,
       tf.zeros_like(to_side, dtype=tf.float32)],
      axis=1)  # [batch_size, 4, 3] matrix
  view_rotation = tf.concat(
      [view_rotation, w_column], axis=2)  # [batch_size, 4, 4]

  identity_batch = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
  view_translation = tf.concat([identity_batch, tf.expand_dims(-eye, 2)], 2)
  view_translation = tf.concat(
      [view_translation,
       tf.reshape(w_column, [batch_size, 1, 4])], 1)
  camera_matrices = tf.matmul(view_rotation, view_translation)
  return camera_matrices

def transform_homogeneous(matrices, vertices):
  if len(matrices.shape) != 3:
    raise ValueError(
        'matrices must have 3 dimensions (missing batch dimension?)')
  if len(vertices.shape) != 3:
    raise ValueError(
        'vertices must have 3 dimensions (missing batch dimension?)')
  homogeneous_coord = tf.ones(
      [tf.shape(vertices)[0], tf.shape(vertices)[1], 1], dtype=tf.float32)
  vertices_homogeneous = tf.concat([vertices, homogeneous_coord], 2)

  return tf.matmul(vertices_homogeneous, matrices, transpose_b=True)


#===============================================================================================================================================================
#rasterize_triangles.py
rasterize_triangles_module = tf.load_op_library(
    #os.path.join(os.environ['TEST_SRCDIR'],
    os.path.join('/home4/yiran/TalkingFace/Pipeline/Deep3DFaceReconstruction',
    'tf_mesh_renderer/mesh_renderer/kernels/rasterize_triangles_kernel.so'))


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
  clip_space_vertices = transform_homogeneous(
      camera_matrices, world_space_vertices)
  return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                              image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if len(clip_space_vertices.shape) != 3:
    raise ValueError('The vertex buffer must be 3D.')

  vertex_count = clip_space_vertices.shape[1].value

  batch_size = tf.shape(clip_space_vertices)[0]
  
  per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
    size=batch_size)
  per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

  def batch_loop_condition(b, *args):
    return b < batch_size

  def batch_loop_iteration(b, per_image_barycentric_coordinates,
    per_image_vertex_ids):
    barycentric_coords, triangle_ids, _ = (
        rasterize_triangles_module.rasterize_triangles(
            clip_space_vertices[b, :, :], triangles, image_width,
            image_height))
    per_image_barycentric_coordinates = \
      per_image_barycentric_coordinates.write(
        b, tf.reshape(barycentric_coords, [-1, 3]))

    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
    per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

    return b+1, per_image_barycentric_coordinates, per_image_vertex_ids

  b = tf.constant(0)
  _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
    batch_loop_condition, batch_loop_iteration,
    [b, per_image_barycentric_coordinates, per_image_vertex_ids])

  barycentric_coordinates = tf.reshape(
    per_image_barycentric_coordinates.stack(), [-1, 3])
  vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
  summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(summed_attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
  alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

  attributes_with_background = (
      alphas * attribute_images + (1.0 - alphas) * background_value)

  return attributes_with_background