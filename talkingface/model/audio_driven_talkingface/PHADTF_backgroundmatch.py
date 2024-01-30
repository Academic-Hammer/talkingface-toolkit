# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import os

import numpy as np
import tensorflow as tf
import math


import test_utils
import camera_utils
import rasterize_triangles

import numpy as np 
from PIL import Image
from scipy.io import loadmat,savemat
from array import array


class RenderTest(tf.test.TestCase):

  def setUp(self):
    self.test_data_directory = 'mesh_renderer/test_data/'

    tf.reset_default_graph()
    self.cube_vertex_positions = tf.constant(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
         [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=tf.float32)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

    self.tf_float = lambda x: tf.constant(x, dtype=tf.float32)

    self.image_width = 640
    self.image_height = 480

    self.perspective = camera_utils.perspective(
        self.image_width / self.image_height,
        self.tf_float([40.0]), self.tf_float([0.01]),
        self.tf_float([10.0]))

  def runTriangleTest(self, w_vector, target_image_name):
    """Directly renders a rasterized triangle's barycentric coordinates.

    Tests only the kernel (rasterize_triangles_module).

    Args:
      w_vector: 3 element vector of w components to scale triangle vertices.
      target_image_name: image file name to compare result against.
    """
    clip_init = np.array(
        [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
        dtype=np.float32)
    clip_init = clip_init * np.reshape(
        np.array(w_vector, dtype=np.float32), [3, 1])

    clip_coordinates = tf.constant(clip_init)
    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    rendered_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, triangles, self.image_width, self.image_height))
    rendered_coordinates = tf.concat(
        [rendered_coordinates,
         tf.ones([self.image_height, self.image_width, 1])], axis=2)
    with self.test_session() as sess:
      image = rendered_coordinates.eval()
      baseline_image_path = os.path.join(self.test_data_directory,
                                         target_image_name)
      test_utils.expect_image_file_and_render_are_near(
          self, sess, baseline_image_path, image)

  def testRendersSimpleTriangle(self):
    self.runTriangleTest((1.0, 1.0, 1.0), 'Simple_Triangle.png')

  def testRendersPerspectiveCorrectTriangle(self):
    self.runTriangleTest((0.2, 0.5, 2.0), 'Perspective_Corrected_Triangle.png')

  def testRendersTwoCubesInBatch(self):
      """Renders a simple cube in two viewpoints to test the python wrapper."""

      vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
      vertex_rgba = tf.concat([vertex_rgb, tf.ones([8, 1])], axis=1)

      center = self.tf_float([[0.0, 0.0, 0.0]])
      world_up = self.tf_float([[0.0, 1.0, 0.0]])
      look_at_1 = camera_utils.look_at(self.tf_float([[2.0, 3.0, 6.0]]),
          center, world_up)
      look_at_2 = camera_utils.look_at(self.tf_float([[-3.0, 1.0, 6.0]]),
          center, world_up)
      projection_1 = tf.matmul(self.perspective, look_at_1)
      projection_2 = tf.matmul(self.perspective, look_at_2)
      projection = tf.concat([projection_1, projection_2], axis=0)
      background_value = [0.0, 0.0, 0.0, 0.0]

      rendered = rasterize_triangles.rasterize(
          tf.stack([self.cube_vertex_positions, self.cube_vertex_positions]),
          tf.stack([vertex_rgba, vertex_rgba]), self.cube_triangles, projection,
          self.image_width, self.image_height, background_value)

      with self.test_session() as sess:
        images = sess.run(rendered, feed_dict={})
        for i in (0, 1):
          image = images[i, :, :, :]
          baseline_image_name = 'Unlit_Cube_{}.png'.format(i)
          baseline_image_path = os.path.join(self.test_data_directory,
                                            baseline_image_name)
          test_utils.expect_image_file_and_render_are_near(
            self, sess, baseline_image_path, image)

  def testSimpleTriangleGradientComputation(self):
    """Verifies the Jacobian matrix for a single pixel.

    The pixel is in the center of a triangle facing the camera. This makes it
    easy to check which entries of the Jacobian might not make sense without
    worrying about corner cases.
    """
    test_pixel_x = 325
    test_pixel_y = 245

    clip_coordinates = tf.placeholder(tf.float32, shape=[3, 4])

    triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, triangles, self.image_width, self.image_height))

    pixels_to_compare = barycentric_coordinates[
        test_pixel_y:test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]

    with self.test_session():
      ndc_init = np.array(
          [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
          dtype=np.float32)
      theoretical, numerical = tf.test.compute_gradient(
          clip_coordinates, (3, 4),
          pixels_to_compare, (1, 1, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.0, True))
      self.assertTrue(jacobians_match, message)

  def testInternalRenderGradientComputation(self):
    """Isolates and verifies the Jacobian matrix for the custom kernel."""
    image_height = 21
    image_width = 28

    clip_coordinates = tf.placeholder(tf.float32, shape=[8, 4])

    barycentric_coordinates, _, _ = (
        rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
            clip_coordinates, self.cube_triangles, image_width, image_height))

    with self.test_session():
      # Precomputed transformation of the simple cube to normalized device
      # coordinates, in order to isolate the rasterization gradient.
      # pyformat: disable
      ndc_init = np.array(
          [[-0.43889722, -0.53184521, 0.85293502, 1.0],
           [-0.37635487, 0.22206162, 0.90555805, 1.0],
           [-0.22849123, 0.76811147, 0.80993629, 1.0],
           [-0.2805393, -0.14092168, 0.71602166, 1.0],
           [0.18631913, -0.62634289, 0.88603103, 1.0],
           [0.16183566, 0.08129397, 0.93020856, 1.0],
           [0.44147962, 0.53497446, 0.85076219, 1.0],
           [0.53008741, -0.31276882, 0.77620775, 1.0]],
          dtype=np.float32)
      # pyformat: enable
      theoretical, numerical = tf.test.compute_gradient(
          clip_coordinates, (8, 4),
          barycentric_coordinates, (image_height, image_width, 3),
          x_init_value=ndc_init,
          delta=4e-2)
      jacobians_match, message = (
          test_utils.check_jacobians_are_nearly_equal(
              theoretical, numerical, 0.01, 0.01))
      self.assertTrue(jacobians_match, message)
# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of TF functions for managing 3D camera matrices."""




def perspective(aspect_ratio, fov_y, near_clip, far_clip):
  """Computes perspective transformation matrices.

  Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

  Args:
    aspect_ratio: float value specifying the image aspect ratio (width/height).
    fov_y: 1-D float32 Tensor with shape [batch_size] specifying output vertical
        field of views in degrees.
    near_clip: 1-D float32 Tensor with shape [batch_size] specifying near
        clipping plane distance.
    far_clip: 1-D float32 Tensor with shape [batch_size] specifying far clipping
        plane distance.

  Returns:
    A [batch_size, 4, 4] float tensor that maps from right-handed points in eye
    space to left-handed points in clip space.
  """
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
  """Computes camera viewing matrices.

  Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

  Args:
    eye: 2-D float32 tensor with shape [batch_size, 3] containing the XYZ world
        space position of the camera.
    center: 2-D float32 tensor with shape [batch_size, 3] containing a position
        along the center of the camera's gaze.
    world_up: 2-D float32 tensor with shape [batch_size, 3] specifying the
        world's up direction; the output camera will have no tilt with respect
        to this direction.

  Returns:
    A [batch_size, 4, 4] float tensor containing a right-handed camera
    extrinsics matrix that maps points from world space to points in eye space.
  """
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


def euler_matrices(angles):
  """Computes a XYZ Tait-Bryan (improper Euler angle) rotation.

  Returns 4x4 matrices for convenient multiplication with other transformations.

  Args:
    angles: a [batch_size, 3] tensor containing X, Y, and Z angles in radians.

  Returns:
    a [batch_size, 4, 4] tensor of matrices.
  """
  s = tf.sin(angles)
  c = tf.cos(angles)
  # Rename variables for readability in the matrix definition below.
  c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
  s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

  zeros = tf.zeros_like(s[:, 0])
  ones = tf.ones_like(s[:, 0])

  # pyformat: disable
  flattened = tf.concat(
      [
          c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
          c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
          -s1, c1 * s0, c1 * c0, zeros,
          zeros, zeros, zeros, ones
      ],
      axis=0)
  # pyformat: enable
  reshaped = tf.reshape(flattened, [4, 4, -1])
  return tf.transpose(reshaped, [2, 0, 1])


def transform_homogeneous(matrices, vertices):
  """Applies batched 4x4 homogenous matrix transformations to 3-D vertices.

  The vertices are input and output as as row-major, but are interpreted as
  column vectors multiplied on the right-hand side of the matrices. More
  explicitly, this function computes (MV^T)^T.
  Vertices are assumed to be xyz, and are extended to xyzw with w=1.

  Args:
    matrices: a [batch_size, 4, 4] tensor of matrices.
    vertices: a [batch_size, N, 3] tensor of xyz vertices.

  Returns:
    a [batch_size, N, 4] tensor of xyzw vertices.

  Raises:
    ValueError: if matrices or vertices have the wrong number of dimensions.
  """
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


# define facemodel for reconstruction
class BFM():
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'
		model = loadmat(model_path)
		self.meanshape = model['meanshape'] # mean face shape 
		self.idBase = model['idBase'] # identity basis
		self.exBase = model['exBase'] # expression basis
		self.meantex = model['meantex'] # mean face texture
		self.texBase = model['texBase'] # texture basis
		self.point_buf = model['point_buf'] # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
		self.tri = model['tri'] # vertex index for each triangle face, starts from 1
		self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 68 face landmark index, starts from 0

# load expression basis
def LoadExpBasis():
	n_vertex = 53215
	Expbin = open('BFM/Exp_Pca.bin','rb')
	exp_dim = array('i')
	exp_dim.fromfile(Expbin,1)
	expMU = array('f')
	expPC = array('f')
	expMU.fromfile(Expbin,3*n_vertex)
	expPC.fromfile(Expbin,3*exp_dim[0]*n_vertex)

	expPC = np.array(expPC)
	expPC = np.reshape(expPC,[exp_dim[0],-1])
	expPC = np.transpose(expPC)

	expEV = np.loadtxt('BFM/std_exp.txt')

	return expPC,expEV

# transfer original BFM09 to our face model
def transferBFM09():
	original_BFM = loadmat('BFM/01_MorphableModel.mat')
	shapePC = original_BFM['shapePC'] # shape basis
	shapeEV = original_BFM['shapeEV'] # corresponding eigen value
	shapeMU = original_BFM['shapeMU'] # mean face
	texPC = original_BFM['texPC'] # texture basis
	texEV = original_BFM['texEV'] # eigen value
	texMU = original_BFM['texMU'] # mean texture

	expPC,expEV = LoadExpBasis()

	# transfer BFM09 to our face model

	idBase = shapePC*np.reshape(shapeEV,[-1,199])
	idBase = idBase/1e5 # unify the scale to decimeter
	idBase = idBase[:,:80] # use only first 80 basis

	exBase = expPC*np.reshape(expEV,[-1,79])
	exBase = exBase/1e5 # unify the scale to decimeter
	exBase = exBase[:,:64] # use only first 64 basis

	texBase = texPC*np.reshape(texEV,[-1,199])
	texBase = texBase[:,:80] # use only first 80 basis

	# our face model is cropped align face landmarks which contains only 35709 vertex.
	# original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
	# thus we select corresponding vertex to get our face model.

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

if __name__ == '__main__':
  tf.test.main()
